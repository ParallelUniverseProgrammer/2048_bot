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
    'best_avg_reward': 0,
    'best_max_tile': 0
}

# Hardware info update interval (in seconds)
HARDWARE_UPDATE_INTERVAL = 2.0

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
    stop_event.set()
    
    if training_thread is not None:
        # Don't join, just let it terminate naturally
        # This prevents blocking the main thread if training is slow to stop
        training_thread = None
    
    training_process = None
    socketio.emit('process_stopped')

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
    # Simplified data collector for reliable updates
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
            
        def update(self, avg_batch_reward, recent_avg_reward, best_avg_reward,
                  avg_batch_moves, batch_max_tile, best_max_tile, batch_loss,
                  total_episodes, rewards_history, moves_history, max_tile_history,
                  best_tile_rate, current_lr):
            # Rate limit updates to avoid overwhelming the UI (max 2 updates per second)
            current_time = time.time()
            if current_time - self.last_update_time < 0.5 and self.total_episodes > 0:
                return
                
            self.last_update_time = current_time
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
            
            # Print what we're sending (for debugging)
            print(f"Sending update: episodes={self.total_episodes}, reward={self.batch_reward:.2f}, max_tile={self.batch_max_tile}")
            
            # Send update through pipe with explicit error handling
            try:
                self.conn.send({
                    'total_episodes': self.total_episodes,
                    'batch_loss': self.batch_loss,
                    'avg_batch_reward': self.batch_reward,
                    'recent_avg_reward': self.recent_avg_reward,
                    'best_avg_reward': self.best_avg_reward,
                    'avg_batch_moves': self.avg_batch_moves,
                    'batch_max_tile': self.batch_max_tile,
                    'best_max_tile': self.best_max_tile,
                    'best_tile_rate': self.best_tile_rate,
                    'current_lr': self.current_lr
                })
                print("Update sent successfully")
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
                log_probs, entropies, episode_reward, moves, max_tile = bot_module.simulate_episode(model, device)
                print(f"Episode complete: reward={episode_reward:.2f}, moves={moves}, max_tile={max_tile}")
                
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
                if total_episodes % 2 == 0:  # Update every other episode to reduce UI load
                    current_avg_reward = batch_reward_sum / min(total_episodes, bot_module.BATCH_SIZE)
                    recent_avg_reward = (sum(rewards_history[-100:]) / min(len(rewards_history), 100)
                                       if rewards_history else 0.0)
                    
                    print(f"Sending update to UI: episodes={total_episodes}, reward={current_avg_reward:.2f}")
                    # For debugging
                    print(f"Episode batch so far: {len(batch_log)}/{bot_module.BATCH_SIZE}")
                    
                    # Create a dummy batch loss to ensure loss history is updated
                    current_batch_loss = 0.0
                    if batch_log:
                        current_batch_loss = sum(loss.item() for loss in batch_log) / len(batch_log)
                    
                    data_collector.update(
                        current_avg_reward, recent_avg_reward, best_avg_reward,
                        batch_moves_sum / min(total_episodes, bot_module.BATCH_SIZE), batch_max_tile, best_max_tile,
                        current_batch_loss,  # Use actual loss when available
                        total_episodes, rewards_history, moves_history, max_tile_history,
                        0.0, current_lr
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
            
            # Update running baseline using exponential moving average
            avg_batch_reward = batch_reward_sum / bot_module.BATCH_SIZE
            if total_episodes <= bot_module.WARMUP_EPISODES:
                baseline = avg_batch_reward
            else:
                baseline = 0.95 * baseline + 0.05 * avg_batch_reward
            
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
            data_collector.update(
                avg_batch_reward, recent_avg_reward, best_avg_reward,
                batch_moves_sum / bot_module.BATCH_SIZE, batch_max_tile, best_max_tile,
                batch_loss.item() if batch_loss is not None else 0.0,
                total_episodes, rewards_history, moves_history, max_tile_history,
                best_tile_rate, current_lr
            )
        
        # Save the model periodically regardless of performance
        if total_episodes % 20 == 0:
            print(f"Periodic save at episode {total_episodes}")
            torch.save(model.state_dict(), "2048_model.pt", _use_new_zipfile_serialization=True)
            
        # Save final model when finished
        print("Training loop complete, saving final model")
        torch.save(model.state_dict(), "2048_model.pt", _use_new_zipfile_serialization=True)
    
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
        
        # Enhanced optimizer for more aggressive exploration
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=bot_module.LEARNING_RATE,
            weight_decay=1e-6,  # Reduced weight decay to allow more exploration
            betas=(0.9, 0.95),  # More aggressive beta2 for faster adaptation
            eps=1e-5            # Slightly higher epsilon for more stable updates with novel reward function
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

# Function to handle training updates from pipe - simplified for reliability
def handle_training_updates(conn):
    try:
        print("Training update handler started")
        
        while not stop_event.is_set():
            # Simple polling approach with clear error handling
            try:
                if conn.poll(0.5):  # Check every 0.5 seconds
                    try:
                        data = conn.recv()
                        print(f"Received training update: episodes={data.get('total_episodes', 0)}")
                        
                        # Update local training data
                        training_data['total_episodes'] = data['total_episodes']
                        training_data['rewards_history'].append(data['avg_batch_reward'])
                        training_data['max_tile_history'].append(data['batch_max_tile'])
                        if 'batch_loss' in data and data['batch_loss'] is not None:
                            training_data['loss_history'].append(data['batch_loss'])
                        training_data['best_avg_reward'] = max(training_data['best_avg_reward'], data['best_avg_reward'])
                        training_data['best_max_tile'] = max(training_data['best_max_tile'], data['best_max_tile'])
                        
                        # Create a formatted version for the client with chart data
                        client_data = data.copy()
                        
                        # Add history arrays for charts
                        client_data['rewards_chart'] = list(training_data['rewards_history'])
                        client_data['max_tile_chart'] = list(training_data['max_tile_history'])
                        client_data['loss_chart'] = list(training_data['loss_history'])
                        client_data['moves_chart'] = list(moves_history[-100:]) if 'moves_history' in locals() else []
                        
                        # For debugging:
                        print(f"Chart data sizes: rewards={len(client_data['rewards_chart'])}, tiles={len(client_data['max_tile_chart'])}, loss={len(client_data['loss_chart'])}")
                        
                        # Immediately send the update to the client
                        print("Emitting training update to clients")
                        socketio.emit('training_update', client_data)
                        
                    except (EOFError, BrokenPipeError) as e:
                        print(f"Pipe error: {e}")
                        break
                        
                    except Exception as e:
                        print(f"Error processing training update: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # No data received in timeout period
                    pass
                    
            except Exception as e:
                print(f"Error polling pipe: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)  # Sleep longer on error
                
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