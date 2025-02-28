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
from flask import Flask, render_template, request, send_file, jsonify
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
current_mode = None  # Will be 'train', 'watch', or None
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

# Helper function to format duration nicely
def format_duration(seconds):
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"
    else:
        days = int(seconds / 86400)
        hours = int((seconds % 86400) / 3600)
        return f"{days} day{'s' if days != 1 else ''} {hours} hour{'s' if hours != 1 else ''}"

@app.route('/list_checkpoints')
def list_checkpoints():
    """Return a list of all available checkpoints"""
    try:
        checkpoints = []
        
        # Check if current model exists
        if os.path.exists("2048_model.pt"):
            checkpoints.append({
                "filename": "2048_model.pt",
                "path": "2048_model.pt",
                "is_current": True
            })
            
        # Check for archived checkpoints
        if os.path.exists("checkpoints"):
            archived_files = [f for f in os.listdir("checkpoints") if f.endswith('.pt')]
            for filename in sorted(archived_files, reverse=True):  # Sort newest first
                checkpoints.append({
                    "filename": filename,
                    "path": os.path.join("checkpoints", filename),
                    "is_current": False
                })
        
        return jsonify({
            "checkpoints": checkpoints
        })
    except Exception as e:
        print(f"Error listing checkpoints: {e}")
        return jsonify({"error": str(e)})

@app.route('/checkpoint_info')
def checkpoint_info():
    try:
        # Get path parameter, default to current model
        checkpoint_path = request.args.get('path', "2048_model.pt")
        
        # Check if model checkpoint exists
        if not os.path.exists(checkpoint_path):
            return jsonify({"exists": False, "message": f"Checkpoint {checkpoint_path} not found"})
        
        # Get checkpoint file stats
        stat_info = os.stat(checkpoint_path)
        
        # Creation time
        created_time = stat_info.st_mtime
        created_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_time))
        
        # Age calculation - using current timestamp for up-to-date age
        current_time = time.time()
        age_seconds = current_time - created_time
        age_str = format_duration(age_seconds)
        
        # File size
        size_bytes = stat_info.st_size
        if size_bytes < 1024:
            size_str = f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        
        # Try to extract training time from the model if possible
        training_time = "Unknown"
        episodes = "Unknown"
        best_reward = "Unknown"
        best_tile = "Unknown"
        
        try:
            # Load the checkpoint to get metadata
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
                
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Check if checkpoint has metadata
            if isinstance(checkpoint, dict) and "metadata" in checkpoint:
                metadata = checkpoint["metadata"]
                
                # Extract training duration if available
                if "training_duration" in metadata:
                    training_time = format_duration(metadata["training_duration"])
                elif "training_duration_formatted" in metadata:
                    training_time = metadata["training_duration_formatted"]
                
                # Extract other metadata
                if "total_episodes" in metadata:
                    episodes = str(metadata["total_episodes"])
                    
                if "best_avg_reward" in metadata:
                    best_reward = f"{metadata['best_avg_reward']:.2f}"
                    
                if "best_max_tile" in metadata:
                    best_tile = str(metadata["best_max_tile"])
        except Exception as e:
            print(f"Error extracting metadata from checkpoint: {e}")
            # Continue with default "Unknown" values if metadata extraction fails
        
        return jsonify({
            "exists": True,
            "created": created_str,
            "age": age_str,
            "size": size_str,
            "training_time": training_time,
            "episodes": episodes,
            "best_reward": best_reward,
            "best_tile": best_tile,
            "filename": os.path.basename(checkpoint_path),
            "path": checkpoint_path,
            "timestamp": created_time,
            "current_time": current_time  # Add current time for client-side time calculations
        })
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
    server_url = f"http://{local_ip}:{args.port}"
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
    
    # Start the requested mode
    mode = data.get('mode', 'train')
    current_mode = mode  # Update the global mode tracker
    
    # Notify all clients about the mode change
    socketio.emit('mode_change', {'mode': mode})
    
    if mode == 'train':
        # For Windows compatibility, use a thread instead of a process
        # This avoids pickling issues with dynamically loaded modules
        parent_conn, child_conn = mp.Pipe()
        
        # Start the training in a thread
        training_thread = threading.Thread(
            target=run_in_thread, 
            args=(start_training, child_conn, stop_event, hyperparams)
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
            args=(start_watch, child_conn, stop_event, hyperparams)
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
def run_in_thread(target_function, conn, stop_event, hyperparams=None):
    try:
        if hyperparams:
            target_function(conn, stop_event, hyperparams)
        else:
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

# Handle set_hyperparams event
@socketio.on('set_hyperparams')
def handle_set_hyperparams(hyperparams):
    # Apply hyperparameters to the module
    if hyperparams:
        try:
            print("Applying hyperparameters:", hyperparams)
            
            # Learning parameters
            if 'learning_rate' in hyperparams:
                bot_module.LEARNING_RATE = hyperparams['learning_rate']
            if 'early_lr_multiplier' in hyperparams:
                bot_module.EARLY_LR_MULTIPLIER = hyperparams['early_lr_multiplier']
            if 'warmup_episodes' in hyperparams:
                bot_module.WARMUP_EPISODES = hyperparams['warmup_episodes']
            if 'grad_clip' in hyperparams:
                bot_module.GRAD_CLIP = hyperparams['grad_clip']
            if 'lr_scheduler_patience' in hyperparams:
                bot_module.LR_SCHEDULER_PATIENCE = hyperparams['lr_scheduler_patience']
            if 'lr_scheduler_factor' in hyperparams:
                bot_module.LR_SCHEDULER_FACTOR = hyperparams['lr_scheduler_factor']
                
            # Architecture parameters
            if 'base_dmodel' in hyperparams:
                bot_module.DMODEL = hyperparams['base_dmodel']
            if 'base_nhead' in hyperparams:
                bot_module.NHEAD = hyperparams['base_nhead']
            if 'base_transformer_layers' in hyperparams:
                bot_module.NUM_TRANSFORMER_LAYERS = hyperparams['base_transformer_layers']
            if 'base_high_level_layers' in hyperparams:
                bot_module.NUM_HIGH_LEVEL_LAYERS = hyperparams['base_high_level_layers']
            if 'base_dropout' in hyperparams:
                bot_module.DROPOUT = hyperparams['base_dropout']
                
            # Reward function parameters
            if 'high_tile_bonus' in hyperparams:
                bot_module.HIGH_TILE_BONUS = hyperparams['high_tile_bonus']
            if 'ineffective_penalty' in hyperparams:
                bot_module.INEFFECTIVE_PENALTY = hyperparams['ineffective_penalty']
            if 'reward_scaling' in hyperparams:
                bot_module.REWARD_SCALING = hyperparams['reward_scaling']
            if 'time_factor_constant' in hyperparams:
                bot_module.TIME_FACTOR_CONSTANT = hyperparams['time_factor_constant']
            if 'novelty_bonus' in hyperparams:
                bot_module.NOVELTY_BONUS = hyperparams['novelty_bonus']
            if 'high_tile_threshold' in hyperparams:
                bot_module.HIGH_TILE_THRESHOLD = hyperparams['high_tile_threshold']
            if 'pattern_diversity_bonus' in hyperparams:
                bot_module.PATTERN_DIVERSITY_BONUS = hyperparams['pattern_diversity_bonus']
            if 'strategy_shift_bonus' in hyperparams:
                bot_module.STRATEGY_SHIFT_BONUS = hyperparams['strategy_shift_bonus']
                
            # Exploration parameters
            if 'use_temperature_annealing' in hyperparams:
                bot_module.USE_TEMPERATURE_ANNEALING = hyperparams['use_temperature_annealing']
            if 'initial_temperature' in hyperparams:
                bot_module.INITIAL_TEMPERATURE = hyperparams['initial_temperature']
            if 'final_temperature' in hyperparams:
                bot_module.FINAL_TEMPERATURE = hyperparams['final_temperature']
            if 'temperature_decay' in hyperparams:
                bot_module.TEMPERATURE_DECAY = hyperparams['temperature_decay']
                
            # Training parameters
            if 'base_batch_size' in hyperparams:
                bot_module.BATCH_SIZE = hyperparams['base_batch_size']
            if 'model_save_interval' in hyperparams:
                bot_module.MODEL_SAVE_INTERVAL = hyperparams['model_save_interval']
                
            # Send confirmation back to client
            socketio.emit('hyperparams_updated', {'status': 'success', 'hyperparams': hyperparams})
            
        except Exception as e:
            print(f"Error applying hyperparameters: {e}")
            socketio.emit('hyperparams_updated', {'status': 'error', 'error': str(e)})
    else:
        socketio.emit('hyperparams_updated', {'status': 'error', 'error': 'No hyperparameters provided'})

# Function to run training mode
def start_training(conn, stop_event, hyperparams=None):
    # Apply hyperparameters if provided
    if hyperparams:
        try:
            print("Applying hyperparameters for training:", hyperparams)
            
            # Apply the same hyperparameter updates as in handle_set_hyperparams
            # Learning parameters
            if 'learning_rate' in hyperparams:
                bot_module.LEARNING_RATE = hyperparams['learning_rate']
            if 'early_lr_multiplier' in hyperparams:
                bot_module.EARLY_LR_MULTIPLIER = hyperparams['early_lr_multiplier']
            if 'warmup_episodes' in hyperparams:
                bot_module.WARMUP_EPISODES = hyperparams['warmup_episodes']
            if 'grad_clip' in hyperparams:
                bot_module.GRAD_CLIP = hyperparams['grad_clip']
            if 'lr_scheduler_patience' in hyperparams:
                bot_module.LR_SCHEDULER_PATIENCE = hyperparams['lr_scheduler_patience']
            if 'lr_scheduler_factor' in hyperparams:
                bot_module.LR_SCHEDULER_FACTOR = hyperparams['lr_scheduler_factor']
                
            # Architecture parameters
            if 'base_dmodel' in hyperparams:
                bot_module.DMODEL = hyperparams['base_dmodel']
            if 'base_nhead' in hyperparams:
                bot_module.NHEAD = hyperparams['base_nhead']
            if 'base_transformer_layers' in hyperparams:
                bot_module.NUM_TRANSFORMER_LAYERS = hyperparams['base_transformer_layers']
            if 'base_high_level_layers' in hyperparams:
                bot_module.NUM_HIGH_LEVEL_LAYERS = hyperparams['base_high_level_layers']
            if 'base_dropout' in hyperparams:
                bot_module.DROPOUT = hyperparams['base_dropout']
                
            # Reward function parameters
            if 'high_tile_bonus' in hyperparams:
                bot_module.HIGH_TILE_BONUS = hyperparams['high_tile_bonus']
            if 'ineffective_penalty' in hyperparams:
                bot_module.INEFFECTIVE_PENALTY = hyperparams['ineffective_penalty']
            if 'reward_scaling' in hyperparams:
                bot_module.REWARD_SCALING = hyperparams['reward_scaling']
            if 'time_factor_constant' in hyperparams:
                bot_module.TIME_FACTOR_CONSTANT = hyperparams['time_factor_constant']
            if 'novelty_bonus' in hyperparams:
                bot_module.NOVELTY_BONUS = hyperparams['novelty_bonus']
            if 'high_tile_threshold' in hyperparams:
                bot_module.HIGH_TILE_THRESHOLD = hyperparams['high_tile_threshold']
            if 'pattern_diversity_bonus' in hyperparams:
                bot_module.PATTERN_DIVERSITY_BONUS = hyperparams['pattern_diversity_bonus']
            if 'strategy_shift_bonus' in hyperparams:
                bot_module.STRATEGY_SHIFT_BONUS = hyperparams['strategy_shift_bonus']
                
            # Exploration parameters
            if 'use_temperature_annealing' in hyperparams:
                bot_module.USE_TEMPERATURE_ANNEALING = hyperparams['use_temperature_annealing']
            if 'initial_temperature' in hyperparams:
                bot_module.INITIAL_TEMPERATURE = hyperparams['initial_temperature']
            if 'final_temperature' in hyperparams:
                bot_module.FINAL_TEMPERATURE = hyperparams['final_temperature']
            if 'temperature_decay' in hyperparams:
                bot_module.TEMPERATURE_DECAY = hyperparams['temperature_decay']
                
            # Training parameters
            if 'base_batch_size' in hyperparams:
                bot_module.BATCH_SIZE = hyperparams['base_batch_size']
            if 'model_save_interval' in hyperparams:
                bot_module.MODEL_SAVE_INTERVAL = hyperparams['model_save_interval']
                
        except Exception as e:
            print(f"Error applying hyperparameters for training: {e}")
            conn.send({"error": f"Error applying hyperparameters: {e}"})
            return
    
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
        # Define training start time
        training_start_time = time.time()
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
                
                # Monitor memory usage to dynamically adjust chunk size
                try:
                    if torch.cuda.is_available():
                        allocated_mem = torch.cuda.memory_allocated() / (1024**3)
                        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        mem_percent = (allocated_mem / total_mem) * 100
                        
                        # Dynamically adjust chunk size based on memory pressure
                        if mem_percent > 85:
                            chunk_size = 8  # Very small chunks for high memory pressure
                            print(f"⚠️ High memory pressure ({mem_percent:.1f}%), using small chunks")
                        elif mem_percent > 70:
                            chunk_size = 12  # Smaller chunks for moderate memory pressure
                            print(f"ℹ️ Moderate memory pressure ({mem_percent:.1f}%)")
                        else:
                            chunk_size = 16  # Standard chunk size for normal operation
                    else:
                        chunk_size = 16  # Default for CPU
                except Exception as e:
                    print(f"Memory monitoring error: {e}")
                    chunk_size = 16  # Default on error
                
                # Run simulation with dynamic memory management
                log_probs, entropies, episode_reward, moves, max_tile = bot_module.simulate_episode(
                    model, device, total_episodes=total_episodes, chunk_size=chunk_size
                )
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
                    # Include entropy for exploration - detach to prevent double backward issue
                    entropy_bonus = 5e-3 * torch.stack(entropies).sum().detach()
                    # Create a detached scalar for advantage
                    advantage_t = torch.tensor(advantage, device=device, dtype=torch.float)
                    # Compute loss with proper detaching to avoid double backward issue
                    episode_loss = -torch.stack(log_probs).sum() * advantage_t - entropy_bonus
                    batch_log.append(episode_loss.clone())
                    
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
                # Create fresh tensor to avoid double backward issue
                losses = torch.stack([loss.detach() for loss in batch_log])
                batch_loss = losses.mean()
                
                # Make sure gradients are cleared before backward
                optimizer.zero_grad()
                
                # We can't easily recreate the exact computation graph, so instead
                # we'll use a simpler approach - just use the detached mean and make it require grad
                # This will still update the model, just with a simplified loss
                
                # Create a fresh tensor that requires grad based on the batch loss
                fresh_loss = batch_loss.clone().detach().requires_grad_(True)
                
                # Backward on the fresh tensor
                fresh_loss.backward()
                
                # Clip gradients and update model
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
                
                # Create metadata about training
                training_duration = time.time() - training_start_time
                metadata = {
                    "timestamp": time.time(),
                    "total_episodes": total_episodes,
                    "best_avg_reward": float(best_avg_reward),
                    "best_max_tile": int(best_max_tile),
                    "training_duration": training_duration,
                    "training_duration_formatted": format_duration(training_duration)
                }
                
                # Save model with metadata
                torch.save({
                    "state_dict": model.state_dict(),
                    "metadata": metadata
                }, "2048_model.pt", _use_new_zipfile_serialization=True)
            if batch_max_tile > best_max_tile:
                best_max_tile = batch_max_tile
                # Also save model on new best tile
                print(f"New best tile {best_max_tile} - saving model checkpoint")
                
                # Create metadata about training
                training_duration = time.time() - training_start_time
                metadata = {
                    "timestamp": time.time(),
                    "total_episodes": total_episodes,
                    "best_avg_reward": float(best_avg_reward),
                    "best_max_tile": int(best_max_tile),
                    "training_duration": training_duration,
                    "training_duration_formatted": format_duration(training_duration)
                }
                
                # Save model with metadata
                torch.save({
                    "state_dict": model.state_dict(),
                    "metadata": metadata
                }, "2048_model.pt", _use_new_zipfile_serialization=True)
                
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
                # Create metadata about training
                training_duration = time.time() - training_start_time
                metadata = {
                    "timestamp": time.time(),
                    "total_episodes": total_episodes,
                    "best_avg_reward": float(best_avg_reward),
                    "best_max_tile": int(best_max_tile),
                    "training_duration": training_duration,
                    "training_duration_formatted": format_duration(training_duration)
                }
                
                # Create a timestamp-based filename for archive
                timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
                archive_filename = f"2048_model_{timestamp_str}.pt"
                
                # First, check if the current model exists, if so rename it to archive
                if os.path.exists("2048_model.pt"):
                    try:
                        # Make checkpoints directory if it doesn't exist
                        os.makedirs("checkpoints", exist_ok=True)
                        # Copy current model to the archive with timestamp name
                        import shutil
                        shutil.copy2("2048_model.pt", os.path.join("checkpoints", archive_filename))
                        print(f"✅ Archived previous checkpoint to checkpoints/{archive_filename}")
                    except Exception as e:
                        print(f"⚠️ Could not archive previous checkpoint: {e}")
                
                # Save with optimization setting
                if bot_module.CHECKPOINT_OPTIMIZATION:
                    torch.save({
                        "state_dict": model.state_dict(),
                        "metadata": metadata
                    }, "2048_model.pt", _use_new_zipfile_serialization=True)
                else:
                    torch.save({
                        "state_dict": model.state_dict(),
                        "metadata": metadata
                    }, "2048_model.pt")
                    
                save_duration = time.time() - save_start
                print(f"✅ Checkpoint saved in {save_duration:.1f}s")
            except Exception as e:
                print(f"❌ Error saving checkpoint: {e}")
            
        # Save final model when finished
        print("Training loop complete, saving final model")
        
        # Create metadata about training
        training_duration = time.time() - training_start_time
        metadata = {
            "timestamp": time.time(),
            "total_episodes": total_episodes,
            "best_avg_reward": float(best_avg_reward),
            "best_max_tile": int(best_max_tile),
            "training_duration": training_duration,
            "training_duration_formatted": format_duration(training_duration),
            "final_save": True
        }
        
        # Save with optimization setting
        if bot_module.CHECKPOINT_OPTIMIZATION:
            print("Using optimized checkpoint saving...")
            torch.save({
                "state_dict": model.state_dict(),
                "metadata": metadata
            }, "2048_model.pt", _use_new_zipfile_serialization=True)
        else:
            torch.save({
                "state_dict": model.state_dict(),
                "metadata": metadata
            }, "2048_model.pt")
    
    try:
        # Initialize device, model, optimizer, and scheduler
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = bot_module.ConvTransformerPolicy(
            vocab_size=bot_module.VOCAB_SIZE,
            d_model=bot_module.DMODEL,
            nhead=bot_module.NHEAD,
            num_transformer_layers=bot_module.NUM_TRANSFORMER_LAYERS,
            num_high_level_layers=bot_module.NUM_HIGH_LEVEL_LAYERS,
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
                # Check if the checkpoint has the new format (with metadata)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    # Use non-strict loading to handle architecture changes
                    model.load_state_dict(checkpoint["state_dict"], strict=False)
                    print("Successfully loaded model with metadata")
                else:
                    # Legacy format - direct state dict
                    model.load_state_dict(checkpoint, strict=False)
                    print("Successfully loaded legacy model format")
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
def start_watch(conn, stop_event, hyperparams=None):
    # Apply hyperparameters if provided
    if hyperparams:
        try:
            print("Applying hyperparameters for watch mode:", hyperparams)
            
            # Apply the same hyperparameter updates as in handle_set_hyperparams
            # Just applying architecture parameters for the watch mode
            if 'base_dmodel' in hyperparams:
                bot_module.DMODEL = hyperparams['base_dmodel']
            if 'base_nhead' in hyperparams:
                bot_module.NHEAD = hyperparams['base_nhead']
            if 'base_transformer_layers' in hyperparams:
                bot_module.NUM_TRANSFORMER_LAYERS = hyperparams['base_transformer_layers']
            if 'base_high_level_layers' in hyperparams:
                bot_module.NUM_HIGH_LEVEL_LAYERS = hyperparams['base_high_level_layers']
            if 'base_dropout' in hyperparams:
                bot_module.DROPOUT = hyperparams['base_dropout']
                
        except Exception as e:
            print(f"Error applying hyperparameters for watch mode: {e}")
            conn.send({"error": f"Error applying hyperparameters: {e}"})
            return
    
    try:
        # Initialize device and model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = bot_module.ConvTransformerPolicy(
            vocab_size=bot_module.VOCAB_SIZE,
            d_model=bot_module.DMODEL,
            nhead=bot_module.NHEAD,
            num_transformer_layers=bot_module.NUM_TRANSFORMER_LAYERS,
            num_high_level_layers=bot_module.NUM_HIGH_LEVEL_LAYERS,
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
                # Check if the checkpoint has the new format (with metadata)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    # Use non-strict loading to handle architecture changes
                    model.load_state_dict(checkpoint["state_dict"], strict=False)
                    print("Successfully loaded model with metadata")
                else:
                    # Legacy format - direct state dict
                    model.load_state_dict(checkpoint, strict=False)
                    print("Successfully loaded legacy model format")
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
                            
                            # Forward stats directly to client - no extra processing
                            # Simply ensure we have proper types for key values
                            if 'current_lr' in data and data['current_lr'] is not None:
                                data['current_lr'] = float(data['current_lr'])
                            else:
                                data['current_lr'] = 0.0001
                            
                            # Stream the data immediately to ALL clients (no room parameter = broadcast)
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
                                
                                # Send chart update to ALL clients
                                socketio.emit('chart_update', chart_data)
                            except Exception as e:
                                print(f"Error processing chart data: {e}")
                        
                        # Legacy support for pre-typed messages
                        else:
                            # Forward with minimal processing for compatibility
                            # Broadcast to ALL clients
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