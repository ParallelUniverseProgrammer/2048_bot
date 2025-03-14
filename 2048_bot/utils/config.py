#!/usr/bin/env python3
"""
Configuration and hyperparameter management for 2048 bot.
Centralizes all parameter definitions and provides functions for applying parameters.
"""

import os
import torch
import math

# ---------------- CONFIGURATION PARAMETERS ----------------
# Learning parameters
LEARNING_RATE = 3e-4           # Base learning rate
EARLY_LR_MULTIPLIER = 1.8      # Increased multiplier for faster early exploration
WARMUP_EPISODES = 25           # Shorter warmup for faster adaptation
GRAD_CLIP = 0.85               # Slightly increased for more dynamic gradient updates
LR_SCHEDULER_PATIENCE = 80     # Reduced patience for faster adaptation to plateaus
LR_SCHEDULER_FACTOR = 0.75     # More aggressive reduction to escape local optima
BASE_BATCH_SIZE = 20           # Base batch size (will be dynamically adjusted)
MINI_BATCH_COUNT = 5           # More mini-batches to improve gradient estimation
MODEL_SAVE_INTERVAL = 50       # Changed to save checkpoint every 50 episodes

# Exploration and strategy parameters
USE_TEMPERATURE_ANNEALING = True  # Enable temperature annealing for action selection
INITIAL_TEMPERATURE = 1.4         # Start with high temperature for diverse exploration
FINAL_TEMPERATURE = 0.8           # End with lower temperature for exploitation
TEMPERATURE_DECAY = 0.99995       # Slow decay rate for smooth transition

# Cyclical Learning Rate parameters
USE_CYCLICAL_LR = True         # Enable cyclical learning rate
CYCLICAL_LR_BASE = 2.5e-4      # Lower base rate for stability between cycles
CYCLICAL_LR_MAX = 8e-4         # Higher max rate for more aggressive exploration
CYCLICAL_LR_STEP_SIZE = 50     # Faster cycle time for quicker strategy shifts

# Base model architecture parameters
BASE_DMODEL = 192              # Base dimension for model (will be scaled)
BASE_NHEAD = 8                 # Base number of attention heads (will be scaled)
BASE_TRANSFORMER_LAYERS = 6    # Base transformer depth (will be scaled)
BASE_HIGH_LEVEL_LAYERS = 1     # Base high-level processing layers (will be scaled)
BASE_DROPOUT = 0.15            # Base dropout rate (may be adjusted based on model size)
VOCAB_SIZE = 16                # Vocabulary size for tile embeddings (unchanged)

# Reward function hyperparameters
HIGH_TILE_BONUS = 5.5          # Bonus for achieving high tiles
INEFFECTIVE_PENALTY = 0.15     # Penalty for ineffective moves
REWARD_SCALING = 0.12          # Scaling factor for rewards
TIME_FACTOR_CONSTANT = 50.0    # Time-based scaling factor
NOVELTY_BONUS = 4.0            # Bonus for novel board states
HIGH_TILE_THRESHOLD = 512      # Threshold for additional bonuses
PATTERN_DIVERSITY_BONUS = 2.0  # Bonus for trying different board patterns
STRATEGY_SHIFT_BONUS = 1.0     # Bonus for successfully changing strategies

# Optimization settings
CHECKPOINT_OPTIMIZATION = True # Enable checkpoint optimization
SKIP_BACKWARD_PASS = False     # For debugging: skip backward pass to isolate slowdowns

# Scaled model parameters (set by scale_model_size())
DMODEL = None
NHEAD = None
NUM_TRANSFORMER_LAYERS = None
NUM_HIGH_LEVEL_LAYERS = None
DROPOUT = None
BATCH_SIZE = None

def get_available_memory_gb():
    """Get available VRAM in GB, or system RAM if CUDA not available"""
    try:
        # Try to get CUDA memory if available
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            # Get total memory in bytes and convert to GB
            return torch.cuda.get_device_properties(device).total_memory / (1024**3)
        else:
            # Fall back to system memory if CUDA not available
            import psutil
            return psutil.virtual_memory().available / (1024**3)
    except Exception:
        # Default to conservative estimate if detection fails
        return 4.0  # Assume 4GB as fallback

def get_dynamic_batch_size():
    """Determine batch size based on available memory"""
    # Get available memory
    available_gb = get_available_memory_gb()
    
    # Scale batch size based on available memory
    if available_gb >= 16.0:
        # High memory system
        batch_size = BASE_BATCH_SIZE
    elif available_gb >= 8.0:
        # Mid-range system
        batch_size = max(10, BASE_BATCH_SIZE - 4)
    elif available_gb >= 4.0:
        # Lower memory system
        batch_size = max(8, BASE_BATCH_SIZE - 8)
    else:
        # Constrained memory system
        batch_size = max(4, BASE_BATCH_SIZE - 16)
    
    print(f"Dynamic batch size: {batch_size} (based on {available_gb:.1f}GB available memory)")
    return batch_size

def scale_model_size():
    """Scale model size based on available memory"""
    global DMODEL, NHEAD, NUM_TRANSFORMER_LAYERS, NUM_HIGH_LEVEL_LAYERS, DROPOUT, BATCH_SIZE
    
    available_gb = get_available_memory_gb()
    
    # Define scaling factors based on available memory
    scale_lookup = {
        (16.0, float('inf')): {"scale": 1.25, "depth_scale": 1.33, "hlayers": 2},  # 16+ GB
        (8.0, 16.0): {"scale": 1.15, "depth_scale": 1.17, "hlayers": 2},           # 8-16 GB
        (4.0, 8.0): {"scale": 1.05, "depth_scale": 1.0, "hlayers": 1},             # 4-8 GB
        (0.0, 4.0): {"scale": 1.0, "depth_scale": 0.83, "hlayers": 1}              # <4 GB
    }
    
    # Find the appropriate scaling factors
    for (min_mem, max_mem), factors in scale_lookup.items():
        if min_mem <= available_gb < max_mem:
            scale = factors["scale"]
            depth_scale = factors["depth_scale"]
            hlayers = factors["hlayers"]
            break
    
    # Calculate dimension and make divisible by 8 for efficiency
    dmodel = int(BASE_DMODEL * scale)
    dmodel = (dmodel // 8) * 8  # Make divisible by 8 for efficient computation
    
    # Calculate number of attention heads (must divide dmodel evenly)
    nhead = BASE_NHEAD
    while dmodel % nhead != 0:
        nhead -= 1
    
    # Set the global variables
    DMODEL = dmodel
    NHEAD = nhead
    NUM_TRANSFORMER_LAYERS = max(2, int(BASE_TRANSFORMER_LAYERS * depth_scale))
    NUM_HIGH_LEVEL_LAYERS = hlayers
    
    # Adjust dropout based on model size (larger models benefit from higher dropout)
    DROPOUT = BASE_DROPOUT * (1.0 + (scale - 1.0) * 2)
    
    # Set batch size
    BATCH_SIZE = get_dynamic_batch_size()
    
    print(f"Model scaled for {available_gb:.1f}GB memory:")
    print(f"  - DMODEL: {DMODEL} (base: {BASE_DMODEL})")
    print(f"  - NHEAD: {NHEAD} (base: {BASE_NHEAD})")
    print(f"  - TRANSFORMER_LAYERS: {NUM_TRANSFORMER_LAYERS} (base: {BASE_TRANSFORMER_LAYERS})")
    print(f"  - HIGH_LEVEL_LAYERS: {NUM_HIGH_LEVEL_LAYERS} (base: {BASE_HIGH_LEVEL_LAYERS})")
    print(f"  - DROPOUT: {DROPOUT:.3f} (base: {BASE_DROPOUT})")
    print(f"  - BATCH_SIZE: {BATCH_SIZE} (base: {BASE_BATCH_SIZE})")

def apply_hyperparameters(hyperparams):
    """
    Apply hyperparameters to global configuration.
    Takes a dictionary of parameter names and values.
    Returns a list of applied parameters.
    """
    if not hyperparams:
        return []
    
    global LEARNING_RATE, EARLY_LR_MULTIPLIER, WARMUP_EPISODES, GRAD_CLIP
    global LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR, DMODEL, NHEAD
    global NUM_TRANSFORMER_LAYERS, NUM_HIGH_LEVEL_LAYERS, DROPOUT
    global HIGH_TILE_BONUS, INEFFECTIVE_PENALTY, REWARD_SCALING
    global TIME_FACTOR_CONSTANT, NOVELTY_BONUS, HIGH_TILE_THRESHOLD
    global PATTERN_DIVERSITY_BONUS, STRATEGY_SHIFT_BONUS
    global USE_TEMPERATURE_ANNEALING, INITIAL_TEMPERATURE
    global FINAL_TEMPERATURE, TEMPERATURE_DECAY, BATCH_SIZE
    global MODEL_SAVE_INTERVAL
    
    applied = []
    
    # Learning parameters
    if 'learning_rate' in hyperparams:
        LEARNING_RATE = hyperparams['learning_rate']
        applied.append('learning_rate')
    if 'early_lr_multiplier' in hyperparams:
        EARLY_LR_MULTIPLIER = hyperparams['early_lr_multiplier']
        applied.append('early_lr_multiplier')
    if 'warmup_episodes' in hyperparams:
        WARMUP_EPISODES = hyperparams['warmup_episodes']
        applied.append('warmup_episodes')
    if 'grad_clip' in hyperparams:
        GRAD_CLIP = hyperparams['grad_clip']
        applied.append('grad_clip')
    if 'lr_scheduler_patience' in hyperparams:
        LR_SCHEDULER_PATIENCE = hyperparams['lr_scheduler_patience']
        applied.append('lr_scheduler_patience')
    if 'lr_scheduler_factor' in hyperparams:
        LR_SCHEDULER_FACTOR = hyperparams['lr_scheduler_factor']
        applied.append('lr_scheduler_factor')
        
    # Architecture parameters
    if 'base_dmodel' in hyperparams:
        DMODEL = hyperparams['base_dmodel']
        applied.append('base_dmodel')
    if 'base_nhead' in hyperparams:
        NHEAD = hyperparams['base_nhead']
        applied.append('base_nhead')
    if 'base_transformer_layers' in hyperparams:
        NUM_TRANSFORMER_LAYERS = hyperparams['base_transformer_layers']
        applied.append('base_transformer_layers')
    if 'base_high_level_layers' in hyperparams:
        NUM_HIGH_LEVEL_LAYERS = hyperparams['base_high_level_layers']
        applied.append('base_high_level_layers')
    if 'base_dropout' in hyperparams:
        DROPOUT = hyperparams['base_dropout']
        applied.append('base_dropout')
        
    # Reward function parameters
    if 'high_tile_bonus' in hyperparams:
        HIGH_TILE_BONUS = hyperparams['high_tile_bonus']
        applied.append('high_tile_bonus')
    if 'ineffective_penalty' in hyperparams:
        INEFFECTIVE_PENALTY = hyperparams['ineffective_penalty']
        applied.append('ineffective_penalty')
    if 'reward_scaling' in hyperparams:
        REWARD_SCALING = hyperparams['reward_scaling']
        applied.append('reward_scaling')
    if 'time_factor_constant' in hyperparams:
        TIME_FACTOR_CONSTANT = hyperparams['time_factor_constant']
        applied.append('time_factor_constant')
    if 'novelty_bonus' in hyperparams:
        NOVELTY_BONUS = hyperparams['novelty_bonus']
        applied.append('novelty_bonus')
    if 'high_tile_threshold' in hyperparams:
        HIGH_TILE_THRESHOLD = hyperparams['high_tile_threshold']
        applied.append('high_tile_threshold')
    if 'pattern_diversity_bonus' in hyperparams:
        PATTERN_DIVERSITY_BONUS = hyperparams['pattern_diversity_bonus']
        applied.append('pattern_diversity_bonus')
    if 'strategy_shift_bonus' in hyperparams:
        STRATEGY_SHIFT_BONUS = hyperparams['strategy_shift_bonus']
        applied.append('strategy_shift_bonus')
        
    # Exploration parameters
    if 'use_temperature_annealing' in hyperparams:
        USE_TEMPERATURE_ANNEALING = hyperparams['use_temperature_annealing']
        applied.append('use_temperature_annealing')
    if 'initial_temperature' in hyperparams:
        INITIAL_TEMPERATURE = hyperparams['initial_temperature']
        applied.append('initial_temperature')
    if 'final_temperature' in hyperparams:
        FINAL_TEMPERATURE = hyperparams['final_temperature']
        applied.append('final_temperature')
    if 'temperature_decay' in hyperparams:
        TEMPERATURE_DECAY = hyperparams['temperature_decay']
        applied.append('temperature_decay')
        
    # Training parameters
    if 'base_batch_size' in hyperparams:
        BATCH_SIZE = hyperparams['base_batch_size']
        applied.append('base_batch_size')
    if 'model_save_interval' in hyperparams:
        MODEL_SAVE_INTERVAL = hyperparams['model_save_interval']
        applied.append('model_save_interval')
    
    return applied

def optimize_torch_settings():
    """Apply optimal torch performance settings based on hardware"""
    # Use as many CPU threads as available
    torch.set_num_threads(os.cpu_count() or 8)
    
    # Enable performance optimizations for faster training
    if torch.cuda.is_available():
        # Enable TF32 on Ampere devices for faster training (at slightly reduced precision)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmark for optimized kernels
        torch.backends.cudnn.benchmark = True
        
        # Set higher GPU memory allocation priority 
        try:
            torch.cuda.set_per_process_memory_fraction(0.95)
        except AttributeError:
            # Fall back for older PyTorch versions
            pass
        
        # Clear CUDA cache
        torch.cuda.empty_cache()

# Initialize module when imported
scale_model_size()
optimize_torch_settings()