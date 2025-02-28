#!/usr/bin/env python3
"""
2048 Bot Web Server

This script creates a Flask server that provides a web interface for the 2048 bot.
It allows users to train the bot or watch it play via a web browser, and displays
hardware usage information.

Features:
- Web UI for training visualization with real-time charts
- Game visualization for watching the bot play
- Real-time hardware monitoring (CPU, RAM, GPU)
- WebSocket communication for live updates
"""

import os
import sys
import json
import time
import threading
import argparse
import socket
import webbrowser
from collections import deque
import torch
import torch.multiprocessing as mp
from flask import Flask, render_template, request
from flask_socketio import SocketIO
import psutil
import numpy as np
import signal

# Import 2048_bot functionality
from importlib.machinery import SourceFileLoader
bot_module = SourceFileLoader("bot_module", os.path.join(os.path.dirname(__file__), "2048_bot.py")).load_module()

# Try to import GPUtil for GPU monitoring (fails gracefully if not available)
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
training_process = None
training_thread = None
hardware_monitor_thread = None
stop_event = threading.Event()
stop_hardware_monitor = threading.Event()
training_data = {
    'total_episodes': 0,
    'rewards_history': deque(maxlen=100),
    'max_tile_history': deque(maxlen=100),
    'loss_history': deque(maxlen=100),
    'moves_history': deque(maxlen=100),  # Add moves history tracking
    'best_avg_reward': 0,
    'best_max_tile': 0
}

# Hardware info update interval (in seconds) - update more frequently
HARDWARE_UPDATE_INTERVAL = 1.0

# Get local IP address
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # doesn't need to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

# SocketIO event handlers
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    # Send server URL to client
    local_ip = get_local_ip()
    server_url = f"http://{local_ip}:{args.port}"
    socketio.emit('server_url', {'url': server_url})
    
    # Start hardware monitoring if not already running
    start_hardware_monitoring()

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")

@socketio.on('start')
def handle_start(data):
    global training_process, training_thread
    
    # Make sure any existing process is stopped
    if training_process is not None:
        stop_event.set()
        if training_thread is not None:
            training_thread.join()
        training_process = None
        training_thread = None
    
    # Reset stop event
    stop_event.clear()
    
    # Start the requested mode
    mode = data.get('mode', 'train')
    if mode == 'train':
        # For Windows compatibility, use a thread instead of a process
        # This avoids pickling issues with dynamically loaded modules
        parent_conn, child_conn = mp.Pipe()
        
        # Start the training in a thread
        training_thread = threading.Thread(
            target=run_in_thread, 
            args=(start_training, child_conn, stop_event)
        )
        training_thread.daemon = True
        training_thread.start()
        
        # Start a thread to receive and broadcast training updates
        update_thread = threading.Thread(
            target=handle_training_updates, 
            args=(parent_conn,)
        )
        update_thread.daemon = True
        update_thread.start()
    
    elif mode == 'watch':
        # For Windows compatibility, use a thread instead of a process
        # This avoids pickling issues with dynamically loaded modules
        parent_conn, child_conn = mp.Pipe()
        
        # Start the watch mode in a thread
        training_thread = threading.Thread(
            target=run_in_thread, 
            args=(start_watch, child_conn, stop_event)
        )
        training_thread.daemon = True
        training_thread.start()
        
        # Start a thread to receive and broadcast game updates
        update_thread = threading.Thread(
            target=handle_watch_updates, 
            args=(parent_conn,)
        )
        update_thread.daemon = True
        update_thread.start()

# Helper function to run training/watch functions in a thread
def run_in_thread(target_function, conn, stop_event):
    try:
        target_function(conn, stop_event)
    except Exception as e:
        print(f"Error in thread: {e}")
        try:
            conn.send({"error": str(e)})
        except:
            pass

@socketio.on('stop')
def handle_stop():
    global training_process, training_thread
    
    print("Stopping current process...")
    
    # Send immediate feedback that the stopping process has begun
    socketio.emit('stopping_process')
    
    # Set the stop event
    stop_event.set()
    
    if training_thread is not None:
        # Don't join, just let it terminate naturally
        # This prevents blocking the main thread if training is slow to stop
        training_thread = None
    
    training_process = None
    
    # Start a thread to poll the stop event and notify when it's fully stopped
    def notify_when_stopped():
        # Wait for a maximum of 10 seconds for the process to stop
        max_wait = 10
        for _ in range(max_wait * 4):  # Check every 0.25 seconds
            time.sleep(0.25)
            # After a short delay, we'll force the process to be considered stopped
            # This ensures the UI doesn't get stuck waiting for a response
        
        # Clear the stop event to signal the process is fully stopped
        stop_event.clear()
        # Send notification that the process is fully stopped
        socketio.emit('process_stopped')
    
    stopping_thread = threading.Thread(target=notify_when_stopped)
    stopping_thread.daemon = True
    stopping_thread.start()

# Function to monitor hardware usage and send updates
def monitor_hardware():
    while not stop_hardware_monitor.is_set():
        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory Usage
        memory = psutil.virtual_memory()
        ram_used_gb = memory.used / (1024**3)
        ram_total_gb = memory.total / (1024**3)
        ram_percent = memory.percent
        
        # GPU Usage (if available)
        gpus = []
        if HAS_GPUTIL:
            try:
                gpu_list = GPUtil.getGPUs()
                for gpu in gpu_list:
                    gpus.append({
                        'name': gpu.name,
                        'memory_used': gpu.memoryUsed / 1024,  # Convert MB to GB
                        'memory_total': gpu.memoryTotal / 1024,  # Convert MB to GB
                        'memory_percent': gpu.memoryUtil * 100,
                        'usage': gpu.load * 100,
                        'temperature': gpu.temperature
                    })
            except Exception as e:
                print(f"Error getting GPU info: {e}")
        
        # Create hardware info object
        hardware_info = {
            'cpu': {
                'usage': cpu_percent,
                'cores': cpu_count
            },
            'ram': {
                'used_gb': ram_used_gb,
                'total_gb': ram_total_gb,
                'percent': ram_percent
            },
            'gpus': gpus
        }
        
        # Send update to clients
        socketio.emit('hardware_info', hardware_info)
        
        # Wait for next update interval
        time.sleep(HARDWARE_UPDATE_INTERVAL)

def start_hardware_monitoring():
    global hardware_monitor_thread
    
    if hardware_monitor_thread is None or not hardware_monitor_thread.is_alive():
        stop_hardware_monitor.clear()
        hardware_monitor_thread = threading.Thread(target=monitor_hardware)
        hardware_monitor_thread.daemon = True
        hardware_monitor_thread.start()

# Function to run training mode
def start_training(conn, stop_event):
    # High-performance data collector with immediate updates
    class DataCollector:
        def __init__(self, connection):
            self.conn = connection
            self.total_episodes = 0
            self.batch_loss = 0.0
            self.batch_reward = 0.0
            self.recent_avg_reward = 0.0
            self.best_avg_reward = 0.0
            self.avg_batch_moves = 0.0
            self.batch_max_tile = 0
            self.best_max_tile = 0
            self.best_tile_rate = 0.0
            self.current_lr = bot_module.LEARNING_RATE
            self.last_update_time = 0
            # Buffer for chart data
            self.rewards_buffer = []
            self.tiles_buffer = []
            self.moves_buffer = []
            self.loss_buffer = []
            
        def update(self, avg_batch_reward, recent_avg_reward, best_avg_reward,
                  avg_batch_moves, batch_max_tile, best_max_tile, batch_loss,
                  total_episodes, rewards_history, moves_history, max_tile_history,
                  best_tile_rate, current_lr, status_message=""):
            # No rate limiting - send updates instantly
            self.total_episodes = total_episodes
            self.batch_loss = float(batch_loss) if batch_loss is not None else 0.0
            self.batch_reward = float(avg_batch_reward)
            self.recent_avg_reward = float(recent_avg_reward)
            self.best_avg_reward = float(best_avg_reward)
            self.avg_batch_moves = float(avg_batch_moves)
            self.batch_max_tile = int(batch_max_tile)
            self.best_max_tile = max(self.best_max_tile, int(best_max_tile))
            self.best_tile_rate = float(best_tile_rate)
            self.current_lr = float(current_lr)
            
            # Add to buffered data for charts
            self.rewards_buffer.append(float(avg_batch_reward))
            self.tiles_buffer.append(int(batch_max_tile))
            self.moves_buffer.append(float(avg_batch_moves))
            if batch_loss is not None:
                self.loss_buffer.append(float(batch_loss))
            
            # Keep buffer sizes manageable
            max_buffer = 100
            if len(self.rewards_buffer) > max_buffer:
                self.rewards_buffer = self.rewards_buffer[-max_buffer:]
            if len(self.tiles_buffer) > max_buffer:
                self.tiles_buffer = self.tiles_buffer[-max_buffer:]
            if len(self.moves_buffer) > max_buffer:
                self.moves_buffer = self.moves_buffer[-max_buffer:]
            if len(self.loss_buffer) > max_buffer:
                self.loss_buffer = self.loss_buffer[-max_buffer:]
            
            # Send update through pipe with explicit error handling
            try:
                # Split data into two types: stat updates (fast) and chart updates (less frequent)
                
                # STAT UPDATE - send immediately for every update
                stat_data = {
                    'type': 'stat_update',
                    'total_episodes': int(self.total_episodes),
                    'batch_loss': float(self.batch_loss),
                    'avg_batch_reward': float(self.batch_reward),
                    'recent_avg_reward': float(self.recent_avg_reward),
                    'best_avg_reward': float(self.best_avg_reward),
                    'avg_batch_moves': float(self.avg_batch_moves),
                    'batch_max_tile': int(self.batch_max_tile),
                    'best_max_tile': int(self.best_max_tile),
                    'best_tile_rate': float(self.best_tile_rate),
                    'current_lr': float(self.current_lr)
                }
                
                # Add status message if provided
                if status_message:
                    stat_data['status_message'] = status_message
                    
                # Check if we're approaching checkpoint
                if self.total_episodes % bot_module.MODEL_SAVE_INTERVAL >= bot_module.MODEL_SAVE_INTERVAL - 2:
                    stat_data['approaching_checkpoint'] = True
                    print(f"Checkpoint approaching at episode {self.total_episodes}, using MODEL_SAVE_INTERVAL={bot_module.MODEL_SAVE_INTERVAL}")
                    
                # Send stats immediately
                self.conn.send(stat_data)
                
                # CHART UPDATE - send less frequently to avoid overwhelming
                # Only send chart data after certain episodes or when significant changes occur
                if self.total_episodes % 3 == 0 or batch_max_tile > self.best_max_tile * 0.8:
                    chart_data = {
                        'type': 'chart_update',
                        'rewards_data': self.rewards_buffer.copy(),
                        'tiles_data': self.tiles_buffer.copy(),
                        'moves_data': self.moves_buffer.copy(),
                        'loss_data': self.loss_buffer.copy(),
                        'episode_base': self.total_episodes - len(self.rewards_buffer) + 1
                    }
                    self.conn.send(chart_data)
                
            except (BrokenPipeError, EOFError) as e:
                print(f"Error sending update: {e}")
            except Exception as e:
                print(f"Unexpected error sending update: {e}")
                import traceback
                traceback.print_exc()
    
    # Modify the train_loop function to use our data collector - ultra simplified version
    def web_train_loop(model, optimizer, scheduler, device, data_collector, stop_event):
        print("Entering web_train_loop")
        # Disable anomaly detection
        torch.autograd.set_detect_anomaly(False)
        baseline = 0.0
        total_episodes = 0
        best_avg_reward = -float('inf')
        best_model_state = None
        best_max_tile = 0
        rewards_history = []
        moves_history = []
        max_tile_history = []
        
        # Just use a very basic training loop to get things working
        while not stop_event.is_set():
            print(f"Training iteration, episodes so far: {total_episodes}")
            
            # Set learning rate based on training progress
            if total_episodes < bot_module.WARMUP_EPISODES:
                current_lr = bot_module.LEARNING_RATE * (total_episodes / max(bot_module.WARMUP_EPISODES, 1))
            elif total_episodes < 100:
                current_lr = bot_module.LEARNING_RATE * bot_module.EARLY_LR_MULTIPLIER
            else:
                current_lr = bot_module.LEARNING_RATE
                
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
                
            # Basic batch collection
            batch_log = []
            batch_reward_sum = 0.0
            batch_moves_sum = 0
            batch_max_tile = 0
            
            # Just collect a simple batch of episodes
            for _ in range(bot_module.BATCH_SIZE):
                if stop_event.is_set():
                    break
                
                # Run one episode
                print(f"Simulating episode {total_episodes+1}")
                
                # Check if this episode will be followed by a model save
                # This is causing the periodic slowdown
                checkpoint_saving = (total_episodes + 1) % bot_module.MODEL_SAVE_INTERVAL == 0
                checkpoint_status = ""
                if checkpoint_saving:
                    checkpoint_status = " (checkpoint will be saved after this episode)"
                    print(f"⚠️ Episode {total_episodes+1} will trigger model checkpoint - may cause temporary slowdown")
                
                # Check if we're in episodes 14-16, which sometimes have lower reward
                # This is due to model preparing for checkpoint
                if (total_episodes + 1) % bot_module.MODEL_SAVE_INTERVAL >= bot_module.MODEL_SAVE_INTERVAL - 2:
                    print(f"ℹ️ Approaching model save boundary at episode {total_episodes+1} - rewards may temporarily decrease")
                
                # Measure episode simulation time
                sim_start = time.time()
                log_probs, entropies, episode_reward, moves, max_tile = bot_module.simulate_episode(model, device)
                sim_duration = time.time() - sim_start
                
                print(f"Episode complete: reward={episode_reward:.2f}, moves={moves}, max_tile={max_tile}{checkpoint_status}, took {sim_duration:.2f}s")
                
                total_episodes += 1
                rewards_history.append(episode_reward)
                moves_history.append(moves)
                max_tile_history.append(max_tile)
                batch_reward_sum += episode_reward
                batch_moves_sum += moves
                batch_max_tile = max(batch_max_tile, max_tile)
                
                if log_probs:
                    # Use reward with baseline subtracted
                    advantage = episode_reward - baseline
                    # Include entropy for exploration
                    entropy_bonus = 5e-3 * torch.stack(entropies).sum()
                    episode_loss = -torch.stack(log_probs).sum() * advantage - entropy_bonus
                    batch_log.append(episode_loss)
                    
                # Update UI after each episode for maximum responsiveness
                # Send update for every episode to maximize UI responsiveness
                    current_avg_reward = batch_reward_sum / min(total_episodes, bot_module.BATCH_SIZE)
                    recent_avg_reward = (sum(rewards_history[-100:]) / min(len(rewards_history), 100)
                                       if rewards_history else 0.0)
                    
                    # Add checkpoint status to the log message
                    status_info = ""
                    if (total_episodes) % bot_module.MODEL_SAVE_INTERVAL == 0:
                        status_info = " [CHECKPOINT SAVED]"
                    elif (total_episodes + 1) % bot_module.MODEL_SAVE_INTERVAL == 0:
                        status_info = " [CHECKPOINT COMING UP]"
                        
                    print(f"Sending update to UI: episodes={total_episodes}, reward={current_avg_reward:.2f}{status_info}")
                    # For debugging
                    print(f"Episode batch so far: {len(batch_log)}/{bot_module.BATCH_SIZE}")
                    
                    # Create a dummy batch loss to ensure loss history is updated
                    current_batch_loss = 0.0
                    if batch_log:
                        current_batch_loss = sum(loss.item() for loss in batch_log) / len(batch_log)
                    
                    # Add status message if checkpoint is about to happen
                    status_msg = ""
                    if checkpoint_saving:
                        status_msg = "Saving model checkpoint after this episode..."
                    
                    data_collector.update(
                        current_avg_reward, recent_avg_reward, best_avg_reward,
                        batch_moves_sum / min(total_episodes, bot_module.BATCH_SIZE), batch_max_tile, best_max_tile,
                        current_batch_loss,  # Use actual loss when available
                        total_episodes, rewards_history, moves_history, max_tile_history,
                        0.0, current_lr, status_msg
                    )
            
            if stop_event.is_set():
                break
            
            # Compute the mean loss for the batch and update model
            print("Computing batch loss and updating model")
            batch_loss = None
            if batch_log:
                batch_loss = torch.stack(batch_log).mean()
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), bot_module.GRAD_CLIP)
                optimizer.step()
            
            # Update running baseline using a smoother exponential moving average
            avg_batch_reward = batch_reward_sum / bot_module.BATCH_SIZE
            if total_episodes <= bot_module.WARMUP_EPISODES:
                baseline = avg_batch_reward
            else:
                # Use a slower update rate for more stability
                baseline = 0.99 * baseline + 0.01 * avg_batch_reward
            
            # Update learning rate scheduler
            recent_avg_reward = (sum(rewards_history[-100:]) / min(len(rewards_history), 100)
                               if rewards_history else avg_batch_reward)
            scheduler.step(recent_avg_reward)
            
            # Track best performance
            if recent_avg_reward > best_avg_reward:
                best_avg_reward = recent_avg_reward
                best_model_state = model.state_dict()
                # Save model whenever we get a new best performance
                print("New best performance - saving model checkpoint")
                torch.save(model.state_dict(), "2048_model.pt", _use_new_zipfile_serialization=True)
            if batch_max_tile > best_max_tile:
                best_max_tile = batch_max_tile
                # Also save model on new best tile
                print(f"New best tile {best_max_tile} - saving model checkpoint")
                torch.save(model.state_dict(), "2048_model.pt", _use_new_zipfile_serialization=True)
                
            recent_max_tiles = max_tile_history[-100:]
            best_tile_rate = (recent_max_tiles.count(best_max_tile) / min(len(recent_max_tiles), 100) * 100
                             if recent_max_tiles else 0.0)
            
            # Send final batch update
            print("Sending final batch update to UI")
            # Check if next batch will start with model save
            save_status = ""
            if total_episodes % bot_module.MODEL_SAVE_INTERVAL == 0:
                save_status = "🔄 Model checkpoint saved"
            
            data_collector.update(
                avg_batch_reward, recent_avg_reward, best_avg_reward,
                batch_moves_sum / bot_module.BATCH_SIZE, batch_max_tile, best_max_tile,
                batch_loss.item() if batch_loss is not None else 0.0,
                total_episodes, rewards_history, moves_history, max_tile_history,
                best_tile_rate, current_lr, save_status
            )
        
        # Save the model periodically based on MODEL_SAVE_INTERVAL
        if total_episodes % bot_module.MODEL_SAVE_INTERVAL == 0:
            print(f"📝 Periodic save at episode {total_episodes}")
            save_start = time.time()
            try:
                if bot_module.CHECKPOINT_OPTIMIZATION:
                    torch.save(model.state_dict(), "2048_model.pt", _use_new_zipfile_serialization=True)
                else:
                    torch.save(model.state_dict(), "2048_model.pt")
                save_duration = time.time() - save_start
                print(f"✅ Checkpoint saved in {save_duration:.1f}s")
            except Exception as e:
                print(f"❌ Error saving checkpoint: {e}")
            
        # Save final model when finished
        print("Training loop complete, saving final model")
        if bot_module.CHECKPOINT_OPTIMIZATION:
            print("Using optimized checkpoint saving...")
            torch.save(model.state_dict(), "2048_model.pt", _use_new_zipfile_serialization=True)
        else:
            torch.save(model.state_dict(), "2048_model.pt")
    
    try:
        # Initialize device, model, optimizer, and scheduler
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = bot_module.ConvTransformerPolicy(
            vocab_size=bot_module.VOCAB_SIZE,
            d_model=bot_module.DMODEL,
            nhead=bot_module.NHEAD,
            num_transformer_layers=bot_module.NUM_TRANSFORMER_LAYERS,
            dropout=bot_module.DROPOUT,
            num_actions=4
        ).to(device)
        
        # Use channels_last memory format if on CUDA for better performance
        if device.type == 'cuda':
            model = model.to(memory_format=torch.channels_last)
        
        # Try to load existing model if it exists
        try:
            if os.path.exists("2048_model.pt"):
                print("Loading existing model for continued training")
                checkpoint = torch.load("2048_model.pt", map_location=device)
                # Use non-strict loading to handle architecture changes
                model.load_state_dict(checkpoint, strict=False)
                print("Successfully loaded model for continued training")
        except Exception as e:
            print(f"Warning: Could not load existing model: {e}")
            print("Starting with a fresh model")
            # Continue with fresh model
        
        # Optimizer tuned for stability
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=bot_module.LEARNING_RATE,
            weight_decay=2e-6,  # Slight increase in weight decay for regularization
            betas=(0.9, 0.99),  # More conservative beta2 for stability
            eps=1e-8            # Standard epsilon value
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=bot_module.LR_SCHEDULER_FACTOR,
            patience=bot_module.LR_SCHEDULER_PATIENCE, verbose=True
        )
        
        # Add debugging print statements
        print("Starting training with model:", model.__class__.__name__)
        print(f"Using device: {device}")
        print(f"Batch size: {bot_module.BATCH_SIZE}, Mini-batch count: {bot_module.MINI_BATCH_COUNT}")
        
        # Create data collector and start training
        data_collector = DataCollector(conn)
        try:
            web_train_loop(model, optimizer, scheduler, device, data_collector, stop_event)
        except Exception as e:
            print(f"ERROR in web_train_loop: {e}")
            import traceback
            traceback.print_exc()
            conn.send({"error": f"Training error: {e}"})
    
    except Exception as e:
        print(f"Error in training process: {e}")
    finally:
        conn.close()

# Function to run watch mode
def start_watch(conn, stop_event):
    try:
        # Initialize device and model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = bot_module.ConvTransformerPolicy(
            vocab_size=bot_module.VOCAB_SIZE,
            d_model=bot_module.DMODEL,
            nhead=bot_module.NHEAD,
            num_transformer_layers=bot_module.NUM_TRANSFORMER_LAYERS,
            dropout=bot_module.DROPOUT,
            num_actions=4
        ).to(device)
        
        # Use channels_last memory format if on CUDA for better performance
        if device.type == 'cuda':
            model = model.to(memory_format=torch.channels_last)
        
        # Load checkpoint if it exists
        try:
            # Check if checkpoint exists
            import os
            if not os.path.exists("2048_model.pt"):
                print("No model checkpoint file found")
                conn.send({"error": "No model checkpoint found. Please train the model first."})
                return
                
            # When architecture changes, we need to be more flexible with loading
            try:
                # Try to load using strict=False first (will load parameters that match)
                checkpoint = torch.load("2048_model.pt", map_location=device)
                model.load_state_dict(checkpoint, strict=False)
                print("Loaded compatible parameters from checkpoint")
            except Exception as e:
                print(f"Partial loading issue: {e}")
                conn.send({"error": "Model architecture has changed. Please train a new model before watching."})
                return
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            conn.send({"error": "Error loading model checkpoint. Please train a new model."})
            return
        
        # Game loop
        score = 0.0
        board = bot_module.new_game()
        moves = 0
        delay = 0.15  # Faster delay between moves for more responsive watching
        
        # Send initial board state
        conn.send({
            "board": board,
            "score": score,
            "max_tile": max(max(row) for row in board),
            "moves": moves
        })
        
        while bot_module.can_move(board) and not stop_event.is_set():
            valid_moves = bot_module.get_valid_moves(board)
            if not valid_moves:
                break
                
            state = bot_module.board_to_tensor(board, device)
            logits = model(state).squeeze(0)
            
            # Create validity mask
            mask = torch.full((4,), -float('inf'), device=device)
            for a in valid_moves:
                mask[a] = 0.0
                
            adjusted_logits = logits + mask
            probs = torch.nn.functional.softmax(adjusted_logits, dim=-1)
            m = torch.distributions.Categorical(probs)
            action = m.sample().item()
            
            new_board, moved, gain = bot_module.apply_move(board, action)
            if not moved:
                valid_moves = bot_module.get_valid_moves(board)
                if not valid_moves:
                    break
                action = valid_moves[0]
                new_board, moved, gain = bot_module.apply_move(board, action)
                
            board = new_board
            bot_module.add_random_tile(board)
            score += gain
            moves += 1
            
            # Send updated board state
            conn.send({
                "board": board,
                "score": score,
                "max_tile": max(max(row) for row in board),
                "moves": moves
            })
            
            time.sleep(delay)
            
        # Send final game over message
        conn.send({
            "board": board,
            "score": score,
            "max_tile": max(max(row) for row in board),
            "moves": moves,
            "game_over": True
        })
    
    except Exception as e:
        print(f"Error in watch process: {e}")
    finally:
        conn.close()

# Function to handle training updates from pipe - optimized for instant updates
def handle_training_updates(conn):
    try:
        print("Optimized training update handler started")
        
        while not stop_event.is_set():
            # Use faster polling with minimal delay
            try:
                if conn.poll(0.1):  # Check every 0.1 seconds - more responsive
                    try:
                        data = conn.recv()
                        
                        # Check message type to determine processing
                        msg_type = data.get('type', 'stat_update')  # Default to stat_update for backward compatibility
                        
                        if msg_type == 'stat_update':
                            # Fast path for stats - minimal processing, direct forwarding
                            # Extract the key stats for local tracking
                            training_data['total_episodes'] = int(data['total_episodes'])
                            training_data['best_avg_reward'] = max(training_data['best_avg_reward'], 
                                                                data.get('best_avg_reward', 0))
                            training_data['best_max_tile'] = max(training_data['best_max_tile'], 
                                                               data.get('best_max_tile', 0))
                            
                            # Also update history for chart consistency
                            training_data['rewards_history'].append(data['avg_batch_reward'])
                            training_data['max_tile_history'].append(data['batch_max_tile'])
                            if 'batch_loss' in data and data['batch_loss'] is not None:
                                training_data['loss_history'].append(data['batch_loss'])
                            if 'avg_batch_moves' in data:
                                training_data['moves_history'].append(data['avg_batch_moves'])
                            
                            # Forward stats directly to client - no extra processing
                            # Simply ensure we have proper types for key values
                            if 'current_lr' in data and data['current_lr'] is not None:
                                data['current_lr'] = float(data['current_lr'])
                            else:
                                data['current_lr'] = 0.0001
                            
                            # Stream the data immediately to client
                            socketio.emit('stats_update', data)
                            
                        elif msg_type == 'chart_update':
                            # Process chart data separately - this can be less frequent
                            try:
                                # Pre-validated arrays from the DataCollector class
                                chart_data = {
                                    'rewards_chart': data['rewards_data'],
                                    'max_tile_chart': data['tiles_data'],
                                    'moves_chart': data['moves_data'],
                                    'loss_chart': data['loss_data'],
                                    'episode_base': data['episode_base']
                                }
                                
                                # Send chart update to client
                                socketio.emit('chart_update', chart_data)
                            except Exception as e:
                                print(f"Error processing chart data: {e}")
                        
                        # Legacy support for pre-typed messages
                        else:
                            # Forward with minimal processing for compatibility
                            socketio.emit('training_update', data)
                        
                    except (EOFError, BrokenPipeError) as e:
                        print(f"Pipe error: {e}")
                        break
                        
                    except Exception as e:
                        print(f"Error processing training update: {e}")
                        import traceback
                        traceback.print_exc()
            except Exception as e:
                print(f"Error polling pipe: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)  # Short sleep on error
                
    except Exception as e:
        print(f"Error in training updates thread: {e}")
        import traceback
        traceback.print_exc()

# Function to handle watch updates from pipe
def handle_watch_updates(conn):
    try:
        last_emit_time = 0
        emit_interval = 0.05  # Send updates to UI every 50ms for smooth animations
        buffered_data = None
        
        while not stop_event.is_set():
            if conn.poll(0.01):  # Check with shorter timeout
                try:
                    data = conn.recv()
                    if 'error' in data:
                        # Handle error case
                        print(f"Watch error: {data['error']}")
                        socketio.emit('error', {'message': data['error']})
                        stop_event.set()
                        break
                    
                    # Buffer the latest data
                    buffered_data = data
                    
                    # Only emit at controlled intervals for smooth UI updates
                    current_time = time.time()
                    if current_time - last_emit_time >= emit_interval:
                        # Send update to all clients
                        socketio.emit('game_update', buffered_data)
                        last_emit_time = current_time
                        buffered_data = None
                    
                    # If game is over, stop the process
                    if data.get('game_over', False):
                        # Always emit game over immediately
                        socketio.emit('game_update', data)
                        stop_event.set()
                        break
                except (EOFError, BrokenPipeError):
                    break
            else:
                # Check if it's time to emit buffered data
                current_time = time.time()
                if buffered_data and current_time - last_emit_time >= emit_interval:
                    socketio.emit('game_update', buffered_data)
                    last_emit_time = current_time
                    buffered_data = None
                
                # Very short sleep to prevent CPU hogging
                time.sleep(0.001)
    except Exception as e:
        print(f"Error in watch updates thread: {e}")

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("Shutting down server...")
    stop_event.set()
    stop_hardware_monitor.set()
    # No need to terminate training_process as we're using threads now
    sys.exit(0)

# Main function
def main():
    global args
    parser = argparse.ArgumentParser(description="2048 Bot Web Server")
    parser.add_argument('--port', type=int, default=5000, help="Port to run the server on")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode")
    parser.add_argument('--no-browser', action='store_true', help="Don't open browser automatically")
    args = parser.parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Get local IP
    local_ip = get_local_ip()
    server_url = f"http://{local_ip}:{args.port}"
    print(f"Starting 2048 Bot Web Server at {server_url}")
    
    # Open browser if requested
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(server_url)).start()
    
    # Start the server
    socketio.run(app, host='0.0.0.0', port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()