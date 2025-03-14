#!/usr/bin/env python3
"""
Web interface for 2048 bot.
Uses Flask and SocketIO for real-time communication.
"""

import os
import sys
import json
import time
import threading
import socket
import webbrowser
from collections import deque
import torch
import psutil
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify
from flask_socketio import SocketIO

from ..utils import config, checkpoint
from ..game import board
from ..agent import model, training

# Try to import GPUtil for GPU monitoring (fails gracefully if not available)
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# Initialize Flask app and SocketIO
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'static'))
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
training_process = None
training_thread = None
hardware_monitor_thread = None
stop_event = threading.Event()
stop_hardware_monitor = threading.Event()
current_mode = None  # Will be 'train', 'watch', or None
training_data = {
    'total_episodes': 0,
    'rewards_history': deque(maxlen=100),
    'max_tile_history': deque(maxlen=100),
    'loss_history': deque(maxlen=100),
    'moves_history': deque(maxlen=100),
    'best_avg_reward': 0,
    'best_max_tile': 0
}

# Hardware info update interval (in seconds)
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

@app.route('/list_checkpoints')
def list_checkpoints():
    """Return a list of all available checkpoints"""
    try:
        return jsonify({
            "checkpoints": checkpoint.list_checkpoints()
        })
    except Exception as e:
        print(f"Error listing checkpoints: {e}")
        return jsonify({"error": str(e)})

@app.route('/checkpoint_info')
def checkpoint_info():
    try:
        # Get path parameter, default to current model
        checkpoint_path = request.args.get('path', "2048_model.pt")
        
        # Get retry parameter
        retry_count = int(request.args.get('retry', "0"))
        
        # Debug output to help with troubleshooting
        print(f"Checkpoint info requested for: {checkpoint_path} (retry: {retry_count})")
        
        # Get basic info that doesn't require loading the model
        basic_info = checkpoint.get_checkpoint_info(checkpoint_path)
        
        if not basic_info["exists"]:
            return jsonify(basic_info)
        
        # Try to load model metadata if possible
        try:
            # Create a thread to load the checkpoint with timeout protection
            import threading
            metadata = None
            load_error = None
            
            def load_with_timeout():
                nonlocal metadata, load_error
                try:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    dummy_model = model.create_model(device)
                    metadata = model.load_checkpoint(dummy_model, checkpoint_path, device)
                except Exception as e:
                    load_error = str(e)
                    print(f"Error in load_with_timeout: {e}")
            
            # Increase timeout based on retry count for more reliability
            base_timeout = 3.0  # Base timeout in seconds
            timeout_increment = 1.0  # Increment per retry
            max_timeout = 8.0  # Maximum timeout
            
            # Calculate timeout based on retry count, but cap at max_timeout
            timeout_duration = min(base_timeout + (retry_count * timeout_increment), max_timeout)
            print(f"Using timeout of {timeout_duration:.1f}s for retry {retry_count}")
            
            load_thread = threading.Thread(target=load_with_timeout)
            load_thread.daemon = True
            load_thread.start()
            load_thread.join(timeout=timeout_duration)
            
            if load_thread.is_alive():
                print(f"Checkpoint loading timed out after {timeout_duration:.1f}s")
                # If this is not the max retry, suggest a retry
                if retry_count < 3:  # Limit to 3 retries
                    basic_info["should_retry"] = True
                    basic_info["retry_count"] = retry_count + 1
                    basic_info["retry_message"] = f"Loading timed out, retrying ({retry_count + 1}/3)..."
                
                return jsonify(basic_info)
            
            if load_error:
                print(f"Error loading checkpoint data: {load_error}")
                # If we encountered an error and haven't hit max retries, suggest a retry
                if retry_count < 3:  # Limit to 3 retries
                    basic_info["should_retry"] = True
                    basic_info["retry_count"] = retry_count + 1
                    basic_info["retry_message"] = f"Error loading, retrying ({retry_count + 1}/3)..."
                
                return jsonify(basic_info)
            
            # If we have metadata, add it to the response
            if metadata:
                print("Successfully loaded checkpoint metadata")
                
                # Extract training duration if available
                if "training_duration" in metadata:
                    basic_info["training_time"] = checkpoint.format_duration(metadata["training_duration"])
                elif "training_duration_formatted" in metadata:
                    basic_info["training_time"] = metadata["training_duration_formatted"]
                
                # Extract other metadata
                if "total_episodes" in metadata:
                    basic_info["episodes"] = str(metadata["total_episodes"])
                    
                if "best_avg_reward" in metadata:
                    basic_info["best_reward"] = f"{metadata['best_avg_reward']:.2f}"
                    
                if "best_max_tile" in metadata:
                    basic_info["best_tile"] = str(metadata["best_max_tile"])
                    
        except Exception as e:
            print(f"Error extracting metadata from checkpoint: {e}")
            # Continue with basic info if metadata extraction fails
        
        return jsonify(basic_info)
    except Exception as e:
        print(f"Error getting checkpoint info: {e}")
        return jsonify({"exists": False, "message": f"Error: {str(e)}"})

@app.route('/download_checkpoint')
def download_checkpoint():
    checkpoint_path = request.args.get('path', "2048_model.pt")
    if os.path.exists(checkpoint_path):
        return send_file(checkpoint_path, 
                         as_attachment=True, 
                         download_name=os.path.basename(checkpoint_path))
    else:
        return f"Checkpoint {checkpoint_path} not found", 404

@app.route('/delete_checkpoint', methods=['POST'])
def delete_checkpoint():
    checkpoint_path = request.json.get('path', "2048_model.pt") if request.json else "2048_model.pt"
    
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            return {"success": True, "message": f"Checkpoint {os.path.basename(checkpoint_path)} deleted successfully"}
        except Exception as e:
            return {"success": False, "message": f"Error deleting checkpoint: {str(e)}"}
    else:
        return {"success": False, "message": f"Checkpoint {checkpoint_path} not found"}

# SocketIO event handlers
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    # Send server URL to client
    local_ip = get_local_ip()
    server_url = f"http://{local_ip}:{port}"  # port is global
    socketio.emit('server_url', {'url': server_url})
    
    # Start hardware monitoring if not already running
    start_hardware_monitoring()
    
    # Always inform the client about the current mode
    socketio.emit('mode_change', {'mode': current_mode}, room=request.sid)
    
    # If training is active, join the training room
    if training_thread is not None and not stop_event.is_set() and current_mode == 'train':
        print(f"Adding client {request.sid} to training viewers")
        # Send current training state to new client
        if training_data['total_episodes'] > 0:
            # Send the latest stats to the newly connected client
            socketio.emit('stats_update', {
                'total_episodes': training_data['total_episodes'],
                'avg_batch_reward': training_data['rewards_history'][-1] if training_data['rewards_history'] else 0,
                'recent_avg_reward': sum(training_data['rewards_history']) / len(training_data['rewards_history']) if training_data['rewards_history'] else 0,
                'best_avg_reward': training_data['best_avg_reward'],
                'batch_max_tile': training_data['max_tile_history'][-1] if training_data['max_tile_history'] else 0,
                'best_max_tile': training_data['best_max_tile'],
                'avg_batch_moves': training_data['moves_history'][-1] if training_data['moves_history'] else 0,
                'current_lr': 0.0001  # Default value
            }, room=request.sid)
            
            # Send chart data as well
            if training_data['rewards_history']:
                socketio.emit('chart_update', {
                    'rewards_chart': list(training_data['rewards_history']),
                    'max_tile_chart': list(training_data['max_tile_history']),
                    'loss_chart': list(training_data['loss_history']),
                    'moves_chart': list(training_data['moves_history']),
                    'episode_base': max(0, training_data['total_episodes'] - len(training_data['rewards_history']))
                }, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")

@socketio.on('start')
def handle_start(data):
    global training_process, training_thread, current_mode
    
    # Make sure any existing process is stopped
    if training_process is not None:
        stop_event.set()
        if training_thread is not None:
            training_thread.join()
        training_process = None
        training_thread = None
    
    # Reset stop event
    stop_event.clear()
    
    # Get hyperparameters if provided
    hyperparams = data.get('hyperparams', None)
    if hyperparams:
        # Apply hyperparameters
        config.apply_hyperparameters(hyperparams)
    
    # Start the requested mode
    mode = data.get('mode', 'train')
    current_mode = mode  # Update the global mode tracker
    
    # Notify all clients about the mode change
    socketio.emit('mode_change', {'mode': mode})
    
    if mode == 'train':
        # Start the training in a thread
        parent_conn, child_conn = torch.multiprocessing.Pipe()
        
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
        # Start the watch mode in a thread
        parent_conn, child_conn = torch.multiprocessing.Pipe()
        
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

# Helper function to run functions in a thread
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
    global training_process, training_thread, current_mode
    
    print("Stopping current process...")
    
    # Send immediate feedback that the stopping process has begun
    socketio.emit('stopping_process')
    
    # Set the stop event
    stop_event.set()
    
    # Remember what mode we're stopping
    stopped_mode = current_mode
    
    # Reset the mode
    current_mode = None
    
    if training_thread is not None:
        # Don't join, just let it terminate naturally
        training_thread = None
    
    training_process = None
    
    # Start a thread to poll the stop event and notify when it's fully stopped
    def notify_when_stopped():
        # Wait for a maximum of 10 seconds for the process to stop
        max_wait = 10
        for _ in range(max_wait * 4):  # Check every 0.25 seconds
            time.sleep(0.25)
        
        # Clear the stop event to signal the process is fully stopped
        stop_event.clear()
        # Send notification that the process is fully stopped
        socketio.emit('process_stopped')
        # Notify all clients about the mode change to None
        socketio.emit('mode_change', {'mode': None, 'previous_mode': stopped_mode})
    
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

# Training data collector
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
        self.current_lr = config.LEARNING_RATE
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
        # Update internal state
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
            if self.total_episodes % config.MODEL_SAVE_INTERVAL >= config.MODEL_SAVE_INTERVAL - 2:
                stat_data['approaching_checkpoint'] = True
                
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

# Function to run training mode
def start_training(conn, stop_event):
    try:
        # Initialize device, model, optimizer, and scheduler
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = model.create_model(device)
        
        # Try to load existing model if it exists
        try:
            if os.path.exists("2048_model.pt"):
                print("Loading existing model for continued training")
                model.load_checkpoint(agent, "2048_model.pt", device)
                print("Successfully loaded model for continued training")
        except Exception as e:
            print(f"Warning: Could not load existing model: {e}")
            print("Starting with a fresh model")
        
        # Optimizer tuned for stability
        optimizer = torch.optim.AdamW(
            agent.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=2e-6,  # Slight increase in weight decay for regularization
            betas=(0.9, 0.99),  # More conservative beta2 for stability
            eps=1e-8            # Standard epsilon value
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE, verbose=True
        )
        
        # Add debugging print statements
        print("Starting training with model:", agent.__class__.__name__)
        print(f"Using device: {device}")
        print(f"Batch size: {config.BATCH_SIZE}, Mini-batch count: {config.MINI_BATCH_COUNT}")
        
        # Create data collector for UI updates
        data_collector = DataCollector(conn)
        
        # Define training callbacks
        def on_episode_end(total_episodes, current_reward, avg_batch_reward, recent_avg_reward,
                          best_avg_reward, moves, max_tile, best_max_tile, loss, current_lr):
            # Send update after each episode
            data_collector.update(
                avg_batch_reward, recent_avg_reward, best_avg_reward,
                moves, max_tile, best_max_tile, loss, total_episodes,
                [], [], [], 0.0, current_lr
            )
            
        def on_batch_end(total_episodes, avg_batch_reward, recent_avg_reward, best_avg_reward,
                        avg_batch_moves, batch_max_tile, best_max_tile, loss, current_lr,
                        rewards_history, moves_history, max_tile_history, loss_history):
            # Calculate tile achievement rate
            recent_max_tiles = max_tile_history[-100:]
            best_tile_rate = (recent_max_tiles.count(best_max_tile) / min(len(recent_max_tiles), 100) * 100
                             if recent_max_tiles else 0.0)
            
            # Send comprehensive update after each batch
            data_collector.update(
                avg_batch_reward, recent_avg_reward, best_avg_reward,
                avg_batch_moves, batch_max_tile, best_max_tile, loss, total_episodes,
                rewards_history, moves_history, max_tile_history, best_tile_rate, current_lr
            )
            
        def on_model_save(agent, reason, total_episodes, best_avg_reward, best_max_tile, training_duration):
            # Create metadata about training
            metadata = {
                "timestamp": time.time(),
                "total_episodes": total_episodes,
                "best_avg_reward": float(best_avg_reward),
                "best_max_tile": int(best_max_tile),
                "training_duration": training_duration,
                "training_duration_formatted": checkpoint.format_duration(training_duration),
                "save_reason": reason
            }
            
            # Archive if it's a periodic save
            archive = (reason == "periodic")
            
            # Save the checkpoint
            checkpoint.save_model(agent, "2048_model.pt", metadata, archive)
                
            # Send status update
            status_message = ""
            if reason == "new_best":
                status_message = "🏆 New best performance - model saved"
            elif reason == "new_tile":
                status_message = f"🎮 New best tile {best_max_tile} - model saved"
            elif reason == "periodic":
                status_message = "🔄 Periodic model checkpoint saved"
            elif reason == "final":
                status_message = "✅ Final model saved"
                
            # Update the UI
            data_collector.update(
                0, best_avg_reward, best_avg_reward,
                0, 0, best_max_tile, 0, total_episodes,
                [], [], [], 0.0, config.LEARNING_RATE, status_message
            )
        
        # Create callbacks dictionary
        callbacks = {
            'on_episode_end': on_episode_end,
            'on_batch_end': on_batch_end,
            'on_model_save': on_model_save
        }
        
        # Run the training loop
        try:
            training.train_loop(agent, optimizer, scheduler, device, callbacks, stop_event)
        except Exception as e:
            print(f"ERROR in training loop: {e}")
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
        agent = model.create_model(device)
        
        # Load checkpoint if it exists
        try:
            # Check if checkpoint exists
            if not os.path.exists("2048_model.pt"):
                print("No model checkpoint file found")
                conn.send({"error": "No model checkpoint found. Please train the model first."})
                return
                
            # Try to load the model
            metadata = model.load_checkpoint(agent, "2048_model.pt", device)
            if metadata:
                print("Loaded model checkpoint with metadata")
            else:
                print("Loaded model checkpoint")
                
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            conn.send({"error": "Error loading model checkpoint. Please train a new model."})
            return
        
        # Game loop
        score = 0.0
        game_board = board.new_game()
        moves = 0
        delay = 0.15  # Faster delay between moves for more responsive watching
        
        # Send initial board state
        conn.send({
            "board": game_board,
            "score": score,
            "max_tile": max(max(row) for row in game_board),
            "moves": moves
        })
        
        while board.can_move(game_board) and not stop_event.is_set():
            valid_moves = board.get_valid_moves(game_board)
            if not valid_moves:
                break
                
            state = model.board_to_tensor(game_board, device)
            logits = agent(state).squeeze(0)
            
            # Create validity mask
            mask = torch.full((4,), -float('inf'), device=device)
            for a in valid_moves:
                mask[a] = 0.0
                
            adjusted_logits = logits + mask
            probs = torch.nn.functional.softmax(adjusted_logits, dim=-1)
            m = torch.distributions.Categorical(probs)
            action = m.sample().item()
            
            new_board, moved, gain = board.move(game_board, action)
            if not moved:
                valid_moves = board.get_valid_moves(game_board)
                if not valid_moves:
                    break
                action = valid_moves[0]
                new_board, moved, gain = board.move(game_board, action)
                
            game_board = new_board
            board.add_random_tile(game_board)
            score += gain
            moves += 1
            
            # Send updated board state
            conn.send({
                "board": game_board,
                "score": score,
                "max_tile": max(max(row) for row in game_board),
                "moves": moves
            })
            
            time.sleep(delay)
            
        # Send final game over message
        conn.send({
            "board": game_board,
            "score": score,
            "max_tile": max(max(row) for row in game_board),
            "moves": moves,
            "game_over": True
        })
    
    except Exception as e:
        print(f"Error in watch process: {e}")
    finally:
        conn.close()

# Function to handle training updates from pipe
def handle_training_updates(conn):
    try:
        print("Training update handler started")
        
        while not stop_event.is_set():
            # Use faster polling with minimal delay
            try:
                if conn.poll(0.1):  # Check every 0.1 seconds - more responsive
                    try:
                        data = conn.recv()
                        
                        # Check for error
                        if 'error' in data:
                            print(f"Training error: {data['error']}")
                            socketio.emit('error', {'message': data['error']})
                            stop_event.set()
                            break
                            
                        # Check message type to determine processing
                        msg_type = data.get('type', 'stat_update')  # Default to stat_update for compatibility
                        
                        if msg_type == 'stat_update':
                            # Extract the key stats for local tracking
                            if 'total_episodes' in data:
                                training_data['total_episodes'] = int(data['total_episodes'])
                            training_data['best_avg_reward'] = max(training_data['best_avg_reward'], 
                                                                data.get('best_avg_reward', 0))
                            training_data['best_max_tile'] = max(training_data['best_max_tile'], 
                                                               data.get('best_max_tile', 0))
                            
                            # Also update history for chart consistency
                            if 'avg_batch_reward' in data:
                                training_data['rewards_history'].append(data['avg_batch_reward'])
                            if 'batch_max_tile' in data:
                                training_data['max_tile_history'].append(data['batch_max_tile'])
                            if 'batch_loss' in data and data['batch_loss'] is not None:
                                training_data['loss_history'].append(data['batch_loss'])
                            if 'avg_batch_moves' in data:
                                training_data['moves_history'].append(data['avg_batch_moves'])
                            
                            # Forward stats directly to client
                            if 'current_lr' in data and data['current_lr'] is not None:
                                data['current_lr'] = float(data['current_lr'])
                            else:
                                data['current_lr'] = 0.0001
                            
                            # Stream the data immediately to ALL clients
                            socketio.emit('stats_update', data)
                            
                        elif msg_type == 'chart_update':
                            # Process chart data separately
                            try:
                                chart_data = {
                                    'rewards_chart': data['rewards_data'],
                                    'max_tile_chart': data['tiles_data'],
                                    'moves_chart': data['moves_data'],
                                    'loss_chart': data['loss_data'],
                                    'episode_base': data['episode_base']
                                }
                                
                                # Send chart update to ALL clients
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
                        # Broadcast error to all clients
                        socketio.emit('error', {'message': data['error']})
                        stop_event.set()
                        break
                    
                    # Buffer the latest data
                    buffered_data = data
                    
                    # Only emit at controlled intervals for smooth UI updates
                    current_time = time.time()
                    if current_time - last_emit_time >= emit_interval:
                        # Broadcast update to all clients
                        socketio.emit('game_update', buffered_data)
                        last_emit_time = current_time
                        buffered_data = None
                    
                    # If game is over, stop the process
                    if data.get('game_over', False):
                        # Always emit game over immediately to all clients
                        socketio.emit('game_update', data)
                        stop_event.set()
                        break
                except (EOFError, BrokenPipeError):
                    break
            else:
                # Check if it's time to emit buffered data
                current_time = time.time()
                if buffered_data and current_time - last_emit_time >= emit_interval:
                    # Broadcast to all clients
                    socketio.emit('game_update', buffered_data)
                    last_emit_time = current_time
                    buffered_data = None
                
                # Very short sleep to prevent CPU hogging
                time.sleep(0.001)
    except Exception as e:
        print(f"Error in watch updates thread: {e}")

# Global port variable
port = 5000

def run_server(port_number=5000, debug=False, open_browser=True):
    """
    Run the 2048 bot web interface.
    
    Args:
        port_number: Port to run the server on
        debug: Whether to run in debug mode
        open_browser: Whether to open the browser automatically
    """
    global port
    port = port_number
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Get local IP
    local_ip = get_local_ip()
    server_url = f"http://{local_ip}:{port}"
    print(f"Starting 2048 Bot Web Server at {server_url}")
    
    # Open browser if requested
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(server_url)).start()
    
    # Start the server
    socketio.run(app, host='0.0.0.0', port=port, debug=debug)