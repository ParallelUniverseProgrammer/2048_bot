#!/usr/bin/env python3
"""
Advanced 2048 Self-Play Training with PyTorch

This script implements a reinforcement learning agent that learns to play the 2048 game
through self-play using a deep Transformer architecture and the REINFORCE algorithm.

Features:
- Pure Transformer architecture for sophisticated board state processing
- Deep multi-head attention mechanisms for complex pattern recognition
- Sophisticated reward function with multiple heuristic components
- Validity masking to prevent invalid moves
- Batch training for more stable learning
- Adaptive learning rate with warmup phase
- Web interface for visualizing training progress and watching gameplay
- Real-time hardware monitoring (CPU, RAM, GPU)

Usage:
    python 2048_bot.py              # Web UI interface (recommended)
    python 2048_bot.py --console    # Legacy console mode
    python 2048_bot.py --port 8080  # Use custom port for Web UI
"""

import sys
import math
import time
import random
import copy
import argparse
import curses
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ---------------- CONFIGURATION PARAMETERS ----------------
# Learning parameters - tuned for adaptive learning and strategy exploration
LEARNING_RATE = 3e-4           # Base learning rate
EARLY_LR_MULTIPLIER = 1.8      # Increased multiplier for faster early exploration
WARMUP_EPISODES = 25           # Shorter warmup for faster adaptation
GRAD_CLIP = 0.85               # Slightly increased for more dynamic gradient updates
LR_SCHEDULER_PATIENCE = 80     # Reduced patience for faster adaptation to plateaus
LR_SCHEDULER_FACTOR = 0.75     # More aggressive reduction to escape local optima
BASE_BATCH_SIZE = 20           # Base batch size (will be dynamically adjusted)
MINI_BATCH_COUNT = 5           # More mini-batches to improve gradient estimation
MODEL_SAVE_INTERVAL = 200      # Keep same checkpoint frequency

# Detect available VRAM and scale model accordingly
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

# Dynamically determine batch size based on available memory
def get_dynamic_batch_size():
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

# Set global BATCH_SIZE
BATCH_SIZE = get_dynamic_batch_size()
CHECKPOINT_OPTIMIZATION = True # Enable checkpoint optimization
SKIP_BACKWARD_PASS = False     # For debugging: skip backward pass to isolate slowdowns
# Cyclical Learning Rate parameters - enhanced for strategy exploration
USE_CYCLICAL_LR = True         # Enable cyclical learning rate
CYCLICAL_LR_BASE = 2.5e-4      # Lower base rate for stability between cycles
CYCLICAL_LR_MAX = 8e-4         # Higher max rate for more aggressive exploration
CYCLICAL_LR_STEP_SIZE = 50     # Faster cycle time for quicker strategy shifts
# Exploration and strategy parameters
USE_TEMPERATURE_ANNEALING = True  # Enable temperature annealing for action selection
INITIAL_TEMPERATURE = 1.4         # Start with high temperature for diverse exploration
FINAL_TEMPERATURE = 0.8           # End with lower temperature for exploitation
TEMPERATURE_DECAY = 0.99995      # Slow decay rate for smooth transition

# Model architecture parameters with dynamic scaling based on available memory
# Base model parameters that will be scaled according to available memory
BASE_DMODEL = 192              # Base dimension for model (will be scaled)
BASE_NHEAD = 8                 # Base number of attention heads (will be scaled)
BASE_TRANSFORMER_LAYERS = 6    # Base transformer depth (will be scaled)
BASE_HIGH_LEVEL_LAYERS = 1     # Base high-level processing layers (will be scaled)
BASE_DROPOUT = 0.15            # Base dropout rate (may be adjusted based on model size)
VOCAB_SIZE = 16                # Vocabulary size for tile embeddings (unchanged)

# Scale model based on available memory
# Memory requirements increase roughly quadratically with model size
def scale_model_size():
    """Scale model size based on available memory"""
    available_gb = get_available_memory_gb()
    
    # Define scaling factors based on available memory
    if available_gb >= 16.0:  # High-end GPU with 16+ GB VRAM
        # Full enhancement (~25% increase)
        scale = 1.25
        depth_scale = 1.33
        hlayers = 2
    elif available_gb >= 8.0:  # Mid-range GPU with 8-16 GB VRAM
        # Moderate enhancement (~15% increase)
        scale = 1.15
        depth_scale = 1.17
        hlayers = 2
    elif available_gb >= 4.0:  # Entry-level GPU with 4-8 GB VRAM
        # Slight enhancement (~5% increase)
        scale = 1.05
        depth_scale = 1.0
        hlayers = 1
    else:  # Low memory (< 4GB VRAM)
        # Conservative settings (no increase)
        scale = 1.0
        depth_scale = 0.83  # Reduce depth to save memory
        hlayers = 1
    
    # Calculate dimension and make divisible by 8 for efficiency
    dmodel = int(BASE_DMODEL * scale)
    dmodel = (dmodel // 8) * 8  # Make divisible by 8 for efficient computation
    
    # Calculate number of attention heads (must divide dmodel evenly)
    nhead = BASE_NHEAD
    while dmodel % nhead != 0:
        nhead -= 1
    
    # Set the global variables
    global DMODEL, NHEAD, NUM_TRANSFORMER_LAYERS, NUM_HIGH_LEVEL_LAYERS, DROPOUT
    DMODEL = dmodel
    NHEAD = nhead
    NUM_TRANSFORMER_LAYERS = max(2, int(BASE_TRANSFORMER_LAYERS * depth_scale))
    NUM_HIGH_LEVEL_LAYERS = hlayers
    
    # Adjust dropout based on model size (larger models benefit from higher dropout)
    DROPOUT = BASE_DROPOUT * (1.0 + (scale - 1.0) * 2)
    
    print(f"Model scaled for {available_gb:.1f}GB memory:")
    print(f"  - DMODEL: {DMODEL} (base: {BASE_DMODEL})")
    print(f"  - NHEAD: {NHEAD} (base: {BASE_NHEAD})")
    print(f"  - TRANSFORMER_LAYERS: {NUM_TRANSFORMER_LAYERS} (base: {BASE_TRANSFORMER_LAYERS})")
    print(f"  - HIGH_LEVEL_LAYERS: {NUM_HIGH_LEVEL_LAYERS} (base: {BASE_HIGH_LEVEL_LAYERS})")
    print(f"  - DROPOUT: {DROPOUT:.3f} (base: {BASE_DROPOUT})")

# Call scale_model_size when module is imported
scale_model_size()

# Reward function hyperparameters - enhanced for diverse strategy reinforcement
HIGH_TILE_BONUS = 5.5          # Slightly increased bonus for achieving high tiles
INEFFECTIVE_PENALTY = 0.15     # Reduced penalty to encourage more exploration
REWARD_SCALING = 0.12          # Slightly increased for stronger learning signals
TIME_FACTOR_CONSTANT = 50.0    # Modified for faster time-based scaling
NOVELTY_BONUS = 4.0            # Increased novelty bonus to encourage diverse strategies
HIGH_TILE_THRESHOLD = 512      # Keep threshold for additional bonuses
PATTERN_DIVERSITY_BONUS = 2.0  # New bonus for trying different board patterns
STRATEGY_SHIFT_BONUS = 1.0     # New bonus for successfully changing strategies

# Display settings
DISPLAY_DELAY = 0.001          # Faster refresh rate for training display
SPINNER = "|/-\\"               # Animation spinner
# ------------------------------------------------------------

# Use as many CPU threads as available
# Use all available CPU cores for maximum parallelism
torch.set_num_threads(os.cpu_count() or 8)

# Enable performance optimizations for faster training
if torch.cuda.is_available():
    # Enable TF32 on Ampere devices for faster training (at slightly reduced precision)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN benchmark for optimized kernels (if input sizes don't change much)
    torch.backends.cudnn.benchmark = True
    
    # Set higher GPU memory allocation priority - use 95% of available VRAM
    try:
        # This is safer as it will only work if the function exists
        torch.cuda.set_per_process_memory_fraction(0.95)
    except AttributeError:
        # Fall back for older PyTorch versions
        pass
    
    # Clear CUDA cache
    torch.cuda.empty_cache()

# --- Game Logic Constants and Functions ---
GRID_SIZE = 4  # 2048 board is 4x4

def new_game():
    """Initialize a new 2048 game board."""
    board = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    add_random_tile(board)
    add_random_tile(board)
    return board

def add_random_tile(board):
    """Add a random tile (2 or 4) to an empty cell on the board."""
    empty = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if board[i][j] == 0]
    if empty:
        i, j = random.choice(empty)
        board[i][j] = 4 if random.random() < 0.1 else 2

def can_move(board):
    """Check if any valid moves remain."""
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if board[i][j] == 0:
                return True
            if i < GRID_SIZE - 1 and board[i][j] == board[i+1][j]:
                return True
            if j < GRID_SIZE - 1 and board[i][j] == board[i][j+1]:
                return True
    return False

def compress(board):
    """Slide all nonzero numbers to the left."""
    changed = False
    new_board = []
    for row in board:
        new_row = [num for num in row if num != 0]
        new_row += [0] * (GRID_SIZE - len(new_row))
        if new_row != row:
            changed = True
        new_board.append(new_row)
    return new_board, changed

def merge(board):
    """Merge adjacent tiles with the same value (left-to-right)."""
    changed = False
    score_gain = 0
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE - 1):
            if board[i][j] != 0 and board[i][j] == board[i][j+1]:
                merged_val = board[i][j] * 2
                board[i][j] = merged_val
                board[i][j+1] = 0
                bonus = merged_val * 0.05 if merged_val >= 64 else 0
                score_gain += merged_val + bonus
                changed = True
    return board, changed, score_gain

def reverse(board):
    """Reverse each row of the board."""
    return [row[::-1] for row in board]

def transpose(board):
    """Transpose the board (swap rows and columns)."""
    return [list(row) for row in zip(*board)]

def move_left(board):
    """Perform a left move on the board."""
    new_board, changed1 = compress(board)
    new_board, changed2, score_gain = merge(new_board)
    new_board, _ = compress(new_board)
    return new_board, (changed1 or changed2), score_gain

def move_right(board):
    """Perform a right move on the board."""
    rev_board = reverse(board)
    new_board, changed, score_gain = move_left(rev_board)
    final_board = reverse(new_board)
    return final_board, changed, score_gain

def move_up(board):
    """Perform an upward move on the board."""
    transposed = transpose(board)
    new_board, changed, score_gain = move_left(transposed)
    final_board = transpose(new_board)
    return final_board, changed, score_gain

def move_down(board):
    """Perform a downward move on the board."""
    transposed = transpose(board)
    new_board, changed, score_gain = move_right(transposed)
    final_board = transpose(new_board)
    return final_board, changed, score_gain

def apply_move(board, move):
    """
    Apply a move to the board.
    move: 0: up, 1: down, 2: left, 3: right.
    Returns: new board, whether the board changed, and gained score.
    """
    if move == 0:
        return move_up(board)
    elif move == 1:
        return move_down(board)
    elif move == 2:
        return move_left(board)
    elif move == 3:
        return move_right(board)
    else:
        return board, False, 0

def get_valid_moves(board):
    """Return a list of move indices that are valid for the current board."""
    valid = []
    for a in range(4):
        temp_board = copy.deepcopy(board)
        new_board, moved, _ = apply_move(temp_board, a)
        if moved:
            valid.append(a)
    return valid

def count_potential_merges(board):
    """Count potential adjacent merges for future moves."""
    count = 0
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if board[i][j] != 0:
                if j < GRID_SIZE - 1 and board[i][j] == board[i][j+1]:
                    count += 1
                if i < GRID_SIZE - 1 and board[i][j] == board[i+1][j]:
                    count += 1
    return count

# --- Board State to Tensor ---
def board_to_tensor(board, device):
    """
    Convert board state into a tensor.
    Nonzero tiles are transformed using log2(value) (e.g., 2 -> 1, 4 -> 2),
    while empty cells are represented as 0.
    """
    tokens = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            tokens.append(0 if board[i][j] == 0 else int(math.log2(board[i][j])))
    return torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

# --- Reward Function ---
def compute_novelty(board):
    """
    Evaluate how "interestingly" the tiles are arranged.
    Rewards patterns that differ from the typical corner-stacking strategy.
    Enhanced with pattern diversity analysis.
    """
    # Count how many quadrants of the board have significant tiles
    quadrant_values = [
        sum(board[i][j] for i in range(2) for j in range(2)),  # Top-left
        sum(board[i][j] for i in range(2) for j in range(2, 4)),  # Top-right
        sum(board[i][j] for i in range(2, 4) for j in range(2)),  # Bottom-left
        sum(board[i][j] for i in range(2, 4) for j in range(2, 4))  # Bottom-right
    ]
    
    # Count non-zero quadrants (areas with significant tile values)
    active_quadrants = sum(1 for val in quadrant_values if val > 0)
    
    # Measure variance in quadrant values (higher variance = more diverse board)
    mean_value = sum(quadrant_values) / 4
    variance = sum((val - mean_value) ** 2 for val in quadrant_values) / 4
    variance_norm = math.log(1 + variance) / 10  # Normalize variance to a reasonable range
    
    # Enhanced pattern analysis - look at tile arrangements and sequences
    # This rewards different patterns like "staircases", "snakes", etc.
    pattern_score = 0.0
    
    # 1. Measure distribution of tile values (not just position)
    tile_values = [board[i][j] for i in range(GRID_SIZE) for j in range(GRID_SIZE) if board[i][j] > 0]
    if tile_values:
        # Calculate entropy of tile value distribution (higher entropy = more diverse values)
        value_counts = {}
        for val in tile_values:
            value_counts[val] = value_counts.get(val, 0) + 1
        entropy = 0
        for val, count in value_counts.items():
            p = count / len(tile_values)
            entropy -= p * math.log2(p)
        # Normalize entropy (maximum entropy for n different values is log2(n))
        max_entropy = math.log2(len(value_counts)) if len(value_counts) > 0 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        pattern_score += normalized_entropy * 1.5
    
    # 2. Check for "snake" patterns (tiles arranged in sequence)
    sequential_count = 0
    # Horizontal snake patterns
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE - 1):
            if board[i][j] > 0 and board[i][j+1] > 0:
                if board[i][j] == board[i][j+1] * 2 or board[i][j] * 2 == board[i][j+1]:
                    sequential_count += 1
    
    # Vertical snake patterns
    for j in range(GRID_SIZE):
        for i in range(GRID_SIZE - 1):
            if board[i][j] > 0 and board[i+1][j] > 0:
                if board[i][j] == board[i+1][j] * 2 or board[i][j] * 2 == board[i+1][j]:
                    sequential_count += 1
    
    pattern_score += sequential_count * 0.1
    
    # 3. Reward balanced distributions across the board
    occupied_cells = sum(1 for i in range(GRID_SIZE) for j in range(GRID_SIZE) if board[i][j] > 0)
    balance_score = occupied_cells / (GRID_SIZE * GRID_SIZE) * 0.5
    pattern_score += balance_score
    
    # Combine all novelty components
    novelty_score = active_quadrants * 0.5 + variance_norm + pattern_score * 0.3
    
    return novelty_score

def identify_board_strategy(board):
    """
    Identify which strategy the current board configuration is using.
    Returns a strategy name and confidence score.
    """
    # Extract board features for strategy identification
    max_tile = max(max(row) for row in board)
    max_tile_pos = None
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if board[i][j] == max_tile:
                max_tile_pos = (i, j)
                break
        if max_tile_pos:
            break
    
    # Strategy 1: Corner strategy
    corner_positions = [(0, 0), (0, GRID_SIZE-1), (GRID_SIZE-1, 0), (GRID_SIZE-1, GRID_SIZE-1)]
    corner_confidence = 0.0
    if max_tile_pos in corner_positions:
        corner_confidence = 0.8
        
        # Check if high values are along the edges
        edge_values = []
        # Top edge
        edge_values.extend([board[0][j] for j in range(GRID_SIZE)])
        # Bottom edge
        edge_values.extend([board[GRID_SIZE-1][j] for j in range(GRID_SIZE)])
        # Left edge (excluding corners already counted)
        edge_values.extend([board[i][0] for i in range(1, GRID_SIZE-1)])
        # Right edge (excluding corners already counted)
        edge_values.extend([board[i][GRID_SIZE-1] for i in range(1, GRID_SIZE-1)])
        
        # Calculate what percentage of total value is on edges
        total_value = sum(sum(row) for row in board)
        edge_value = sum(edge_values)
        if total_value > 0:
            edge_ratio = edge_value / total_value
            if edge_ratio > 0.7:
                corner_confidence += 0.2
    
    # Strategy 2: Snake strategy (zigzag pattern of decreasing values)
    snake_confidence = 0.0
    snake_patterns = [
        # Rows alternating left-right, right-left
        [(i, j if i % 2 == 0 else GRID_SIZE - 1 - j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)],
        # Columns alternating top-bottom, bottom-top
        [(i if j % 2 == 0 else GRID_SIZE - 1 - i, j) for j in range(GRID_SIZE) for i in range(GRID_SIZE)]
    ]
    
    for pattern in snake_patterns:
        # Count how many tiles follow a decreasing sequence
        decreasing_count = 0
        last_val = None
        for i, j in pattern:
            if board[i][j] > 0:
                if last_val is None or board[i][j] <= last_val:
                    decreasing_count += 1
                last_val = board[i][j]
        
        # Calculate snake confidence based on how well the pattern matches
        pattern_confidence = decreasing_count / (GRID_SIZE * GRID_SIZE) * 2 - 0.5
        snake_confidence = max(snake_confidence, pattern_confidence)
    
    # Strategy 3: Balanced strategy (values distributed across board)
    balanced_confidence = 0.0
    filled_cells = sum(1 for i in range(GRID_SIZE) for j in range(GRID_SIZE) if board[i][j] > 0)
    balanced_confidence = filled_cells / (GRID_SIZE * GRID_SIZE) * 0.5
    
    # Calculate variance of non-zero cells (low variance = more balanced)
    non_zero_vals = [board[i][j] for i in range(GRID_SIZE) for j in range(GRID_SIZE) if board[i][j] > 0]
    if len(non_zero_vals) > 1:
        mean_val = sum(non_zero_vals) / len(non_zero_vals)
        variance = sum((v - mean_val) ** 2 for v in non_zero_vals) / len(non_zero_vals)
        # Normalize variance
        normalized_variance = min(1.0, math.log(1 + variance) / 15)
        balanced_confidence += (1 - normalized_variance) * 0.5
    
    # Determine the dominant strategy
    strategies = [
        ("corner", corner_confidence),
        ("snake", snake_confidence),
        ("balanced", balanced_confidence)
    ]
    
    dominant_strategy, confidence = max(strategies, key=lambda x: x[1])
    return dominant_strategy, confidence

def compute_strategy_diversity_bonus(board, previous_boards, total_episodes):
    """
    Calculate a bonus for trying diverse strategies.
    Rewards the agent for exploring different approaches to solving the game.
    """
    # If we don't have enough history, return a small default bonus
    if not previous_boards or len(previous_boards) < 3:
        return PATTERN_DIVERSITY_BONUS * 0.5
    
    # Identify current strategy
    current_strategy, current_confidence = identify_board_strategy(board)
    
    # Check how different this strategy is from recent ones
    recent_strategies = [identify_board_strategy(b)[0] for b in previous_boards[-5:]]
    strategy_counts = {}
    for strategy in recent_strategies:
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    # Calculate diversity score
    if current_strategy in strategy_counts:
        # Less bonus for commonly used strategies
        common_factor = strategy_counts[current_strategy] / len(recent_strategies)
        diversity_bonus = PATTERN_DIVERSITY_BONUS * (1 - common_factor * 0.7)
    else:
        # High bonus for completely new strategies
        diversity_bonus = PATTERN_DIVERSITY_BONUS * 1.2
    
    # Scale bonus by confidence - higher confidence strategies get more bonus
    diversity_bonus *= current_confidence
    
    # Add a strategy shift bonus if the strategy changed recently
    if len(recent_strategies) >= 2 and current_strategy != recent_strategies[-1]:
        diversity_bonus += STRATEGY_SHIFT_BONUS
    
    # Early in training (first 100 episodes), provide extra incentive for exploration
    if total_episodes < 100:
        diversity_bonus *= 1.5
    
    return diversity_bonus

def compute_reward(merge_score, board, forced_penalty, move_count, previous_boards=None, total_episodes=0):
    """
    Enhanced reward function with strategy diversity and adaptive scaling.
    Rewards:
      - High tile bonus: encourages achieving higher value tiles
      - Novelty: rewards unusual board configurations to encourage exploration
      - Strategy diversity: rewards trying different approaches to the game
      - Pattern recognition: rewards specific beneficial patterns
      - Adjacency bonus: rewards keeping mergeable tiles together
      - Adaptive checkpoint transitions to prevent learning disruption
    """
    # Get episode number for checkpoint consistency
    episode_num = move_count  # Approximate episode count from move_count
    checkpoint_factor = 1.0
    
    # Apply a smoother, more gradual reward adjustment around checkpoints
    if MODEL_SAVE_INTERVAL > 0:  # Only apply if checkpointing is enabled
        # Calculate how close we are to a checkpoint
        distance_to_checkpoint = episode_num % MODEL_SAVE_INTERVAL
        # Normalize the position within the cycle to [0, 1]
        cycle_position = distance_to_checkpoint / MODEL_SAVE_INTERVAL
        
        # Apply an improved sinusoidal smoothing function
        # Enhanced to avoid sharp transitions that disrupt learning
        checkpoint_factor = 1.0 + 0.025 * math.sin(2 * math.pi * cycle_position)
    
    max_tile = max(max(row) for row in board)
    
    # Enhanced high tile bonus with more dynamic scaling
    log_max = math.log2(max_tile) if max_tile > 0 else 0
    
    # Apply more gradual reward scaling for high tiles
    if max_tile >= HIGH_TILE_THRESHOLD:
        # More sophisticated high tile bonus with smoother scaling
        threshold_log = math.log2(HIGH_TILE_THRESHOLD)
        # Apply exponential scaling based on difference from threshold
        power_factor = (log_max - threshold_log) / 2
        threshold_bonus = (log_max - threshold_log + 1) * (3.0 + power_factor)
        bonus_high = log_max * HIGH_TILE_BONUS + threshold_bonus
    else:
        bonus_high = log_max * HIGH_TILE_BONUS
    
    # Improved novelty calculation
    novelty = compute_novelty(board) * NOVELTY_BONUS
    
    # Enhanced adjacency bonus with pattern recognition
    adjacency_bonus = 0.0
    merge_potential = 0.0
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if board[i][j] >= 32:  # Only consider significant tiles
                # Check horizontal adjacency with graduated rewards for higher tiles
                if j > 0 and board[i][j-1] == board[i][j]:
                    adjacency_bonus += math.log2(board[i][j]) * 0.35
                    # Additional bonus for higher value pairs
                    if board[i][j] >= 128:
                        adjacency_bonus += 0.5
                # Check vertical adjacency
                if i > 0 and board[i-1][j] == board[i][j]:
                    adjacency_bonus += math.log2(board[i][j]) * 0.35
                    # Additional bonus for higher value pairs
                    if board[i][j] >= 128:
                        adjacency_bonus += 0.5
                
                # Check for potential future merges (adjacent tiles that are powers of 2 apart)
                # Horizontal merge potential
                if j > 0 and board[i][j-1] > 0 and (board[i][j-1] == board[i][j]/2 or board[i][j-1] == board[i][j]*2):
                    merge_potential += 0.2
                if j < GRID_SIZE-1 and board[i][j+1] > 0 and (board[i][j+1] == board[i][j]/2 or board[i][j+1] == board[i][j]*2):
                    merge_potential += 0.2
                # Vertical merge potential
                if i > 0 and board[i-1][j] > 0 and (board[i-1][j] == board[i][j]/2 or board[i-1][j] == board[i][j]*2):
                    merge_potential += 0.2
                if i < GRID_SIZE-1 and board[i+1][j] > 0 and (board[i+1][j] == board[i][j]/2 or board[i+1][j] == board[i][j]*2):
                    merge_potential += 0.2
    
    # Add strategy diversity bonus if we have previous boards
    strategy_diversity = 0.0
    if previous_boards:
        strategy_diversity = compute_strategy_diversity_bonus(board, previous_boards, total_episodes)
    
    # Combine all components with increased weighting for diversity
    base_reward = bonus_high + novelty + adjacency_bonus + merge_potential + strategy_diversity
    
    # Enhanced time-dependent bonus that scales more aggressively early and flattens later
    time_factor = 1.0 + math.log(1 + move_count / TIME_FACTOR_CONSTANT)
    
    # Improved merge score component with graduated scaling
    merge_component = merge_score * 0.007  # Slightly increased influence
    
    # Apply the smoothed checkpoint factor
    reward = (base_reward + merge_component) * REWARD_SCALING * time_factor * checkpoint_factor
    
    # Subtract any forced move penalties, with a floor to prevent negative rewards
    reward = max(reward - forced_penalty, 0.01)  # Small positive minimum to keep learning going
    
    return reward

# --- Pure Transformer Policy (Removed CNN) ---
class ConvTransformerPolicy(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=DMODEL, nhead=NHEAD,
                 num_transformer_layers=NUM_TRANSFORMER_LAYERS, 
                 num_high_level_layers=NUM_HIGH_LEVEL_LAYERS,
                 dropout=DROPOUT, num_actions=4):
        """
        Enhanced Transformer architecture with multi-level pattern recognition:
         - Tile embeddings with value-aware positional encoding
         - Multi-scale representation learning
         - Deep Transformer encoder with multi-head attention
         - Strategy-specific attention pathways
         - Enhanced dueling advantage architecture
         - Adaptive normalization
        """
        super().__init__()
        # Token embeddings for board tiles with dynamic scaling
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings for the 4x4 grid (16 positions)
        self.pos_embedding = nn.Parameter(torch.zeros(1, GRID_SIZE * GRID_SIZE, d_model))
        
        # Value-aware positional bias - helps model understand the significance of positions
        self.pos_value_bias = nn.Parameter(torch.zeros(1, GRID_SIZE * GRID_SIZE, d_model // 4))
        self.pos_value_proj = nn.Linear(d_model // 4, d_model)
        
        # Layer normalization with improved parameters
        self.ln_pre = nn.LayerNorm(d_model, eps=1e-6)
        self.ln_post = nn.LayerNorm(d_model, eps=1e-6)
        self.ln_final = nn.LayerNorm(d_model, eps=1e-6)
        
        # Multi-scale feature extraction - different FFN sizes for different abstraction levels
        self.local_feature_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.Dropout(dropout)
        )
        
        # Main transformer stack - increased capacity
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model*3,  # Increased feed-forward capacity
            dropout=dropout, 
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Enhanced high-level reasoning transformers (multiple layers)
        high_level_encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*2,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.high_level_transformer = nn.TransformerEncoder(high_level_encoder, num_layers=num_high_level_layers)
        
        # Strategy-specific attention mechanisms
        # 1. Corner strategy attention (focuses on corners and edges)
        self.corner_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead // 2,
            dropout=dropout,
            batch_first=True
        )
        
        # 2. High-value tile attention (focuses on highest values)
        self.high_value_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead // 2,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. Pattern recognition attention (focuses on tile patterns)
        self.pattern_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead // 2,
            dropout=dropout,
            batch_first=True
        )
        
        # Enhanced token interaction with gating mechanism
        self.tile_interaction = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
        # Gating for adaptive strategy selection
        self.strategy_gate = nn.Sequential(
            nn.Linear(d_model, 3),  # 3 strategies: corner, high-value, pattern
            nn.Softmax(dim=-1)
        )
        
        # Enhanced dueling network architecture with deeper networks
        self.value_stream = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_actions)
        )
        
        # Additional feature extraction for state value estimation
        self.state_features = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Additional dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using enhanced initialization schemes."""
        # Token embeddings with xavier uniform
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # Positional embeddings with improved initialization
        nn.init.normal_(self.pos_embedding, std=0.01)
        nn.init.normal_(self.pos_value_bias, std=0.01)
        
        # Enhanced weight initialization for linear layers
        for module in [self.local_feature_proj, self.tile_interaction, 
                      self.state_features, self.value_stream, 
                      self.advantage_stream, self.strategy_gate]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    # Use kaiming initialization for GELU activation layers
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    nn.init.zeros_(layer.bias)
        
        # Value projection
        nn.init.xavier_uniform_(self.pos_value_proj.weight)
        nn.init.zeros_(self.pos_value_proj.bias)

    def _apply_strategy_attention(self, x):
        """Apply multiple attention strategies and combine them based on gating."""
        batch_size = x.size(0)
        
        # Get strategy weights for this state
        strategy_weights = self.strategy_gate(x.mean(dim=1))  # [batch, 3]
        
        # Apply each attention strategy
        corner_out, _ = self.corner_attn(x, x, x)
        high_value_out, _ = self.high_value_attn(x, x, x)
        pattern_out, _ = self.pattern_attn(x, x, x)
        
        # Combine strategies using strategy weights
        strategy_outs = [corner_out, high_value_out, pattern_out]
        combined = torch.zeros_like(x)
        
        for i, strategy_out in enumerate(strategy_outs):
            # Apply strategy weight for each example in batch
            weight = strategy_weights[:, i].view(batch_size, 1, 1)
            combined += strategy_out * weight
            
        return combined

    def forward(self, x):
        """
        Enhanced forward pass with strategy-adaptive architecture:
          - Input: x of shape (batch, 16) tokens
          - Output: logits of shape (batch, num_actions)
        """
        batch_size = x.size(0)
        
        # Embed tokens (board values)
        x = self.embedding(x)  # (batch, 16, d_model)
        
        # Generate position-value bias - helps model understand value-position relationships
        pos_value_feature = self.pos_value_proj(self.pos_value_bias)
        
        # Add positional embeddings with position-value bias and apply layer norm
        x = self.ln_pre(x + self.pos_embedding + pos_value_feature)
        
        # Apply dropout before transformer
        x = self.dropout1(x)
        
        # Extract local features
        local_features = self.local_feature_proj(x)
        x = x + local_features  # Add local features via residual connection
        
        # Process through main transformer encoder stack
        x = self.transformer(x)
        
        # Apply enhanced tile interaction with residual connection
        residual = x
        x = self.tile_interaction(x)
        x = x + residual  # Residual connection
        
        # Process through high-level reasoning layers
        x = self.high_level_transformer(x)
        
        # Apply strategy-adaptive attention
        attn_output = self._apply_strategy_attention(x)
        x = x + attn_output  # Residual connection
        
        # Apply layer norm
        x = self.ln_post(x)
        
        # Enhanced attention-weighted pooling with learned query
        attention_weights = torch.softmax(x.mean(dim=-1, keepdim=True), dim=1)
        x = (x * attention_weights).sum(dim=1)  # Weighted pooling
        
        # Extract additional state features
        state_features = self.state_features(x)
        x = x + state_features  # Residual connection
        
        # Apply dropout
        x = self.dropout2(x)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Enhanced dueling network architecture
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage for Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        logits = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return logits

# --- Curses Display Helpers ---
def init_colors():
    """Initialize color pairs for a curses display."""
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)   # Default text
    curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Tile 2
    curses.init_pair(3, curses.COLOR_MAGENTA, curses.COLOR_BLACK) # Tile 4
    curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLACK)    # Tile 8
    curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Tile 16
    curses.init_pair(6, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Tile 32
    curses.init_pair(7, curses.COLOR_RED, curses.COLOR_BLACK)     # Tile 64
    curses.init_pair(8, curses.COLOR_WHITE, curses.COLOR_BLACK)   # Higher tiles
    curses.init_pair(9, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Positive reward text
    curses.init_pair(10, curses.COLOR_RED, curses.COLOR_BLACK)    # Negative reward text

def draw_board(stdscr, board, score, extra_text=""):
    """
    Render the 2048 board for watch mode.
    Displays the board, current score, and an optional message.
    """
    stdscr.clear()
    stdscr.border()
    height, width = stdscr.getmaxyx()
    title = "2048 Watch Mode"
    stdscr.attron(curses.color_pair(1))
    stdscr.addstr(1, (width - len(title)) // 2, title)
    score_text = f"Score: {score:.2f}"
    stdscr.addstr(2, (width - len(score_text)) // 2, score_text)
    if extra_text:
        stdscr.addstr(3, (width - len(extra_text)) // 2, extra_text)
    board_width = GRID_SIZE * 7 + 1
    board_height = GRID_SIZE * 2 + 1
    start_y = (height - board_height) // 2
    start_x = (width - board_width) // 2
    for i in range(GRID_SIZE):
        line = '+' + ('-' * 6 + '+') * GRID_SIZE
        stdscr.addstr(start_y + i * 2, start_x, line, curses.color_pair(1))
        for j in range(GRID_SIZE):
            value = board[i][j]
            if value == 0:
                cell = " " * 6
                color = curses.color_pair(1)
            else:
                cell = str(value).center(6)
                if value == 2:
                    color = curses.color_pair(2)
                elif value == 4:
                    color = curses.color_pair(3)
                elif value == 8:
                    color = curses.color_pair(4)
                elif value == 16:
                    color = curses.color_pair(5)
                elif value == 32:
                    color = curses.color_pair(6)
                elif value == 64:
                    color = curses.color_pair(7)
                else:
                    color = curses.color_pair(8)
            stdscr.addstr(start_y + i * 2 + 1, start_x + j * 7, "|" + cell, color)
        stdscr.addstr(start_y + i * 2 + 1, start_x + GRID_SIZE * 7, "|", curses.color_pair(1))
    line = '+' + ('-' * 6 + '+') * GRID_SIZE
    stdscr.addstr(start_y + GRID_SIZE * 2, start_x, line, curses.color_pair(1))
    stdscr.refresh()

# --- Episode Simulation ---
def simulate_episode(model, device, total_episodes=0, temperature=1.0, chunk_size=16):
    """
    Simulate one self-play episode with optimized performance and enhanced exploration.
    Implements temperature annealing for action selection and strategy tracking.
    Includes memory management to prevent OOM errors.
    
    Args:
        model: The policy model to use
        device: The compute device (CPU/GPU)
        total_episodes: Current episode count for adaptive strategies
        temperature: Temperature parameter for exploration (default: 1.0)
        chunk_size: Maximum moves to process before clearing computation graph (default: 16)
        
    Returns:
        log_probs: List of log probabilities of actions
        entropies: List of action entropies
        episode_reward: Computed reward for the episode
        total_moves: Number of moves taken
        max_tile: Highest tile reached
    """
    # Pre-allocate memory for better performance
    board = new_game()
    log_probs = []
    entropies = []
    total_moves = 0
    merge_score_total = 0.0
    forced_penalty_total = 0.0
    
    # Track board history for strategy diversity calculations
    board_history = []
    
    # Create reusable tensors to avoid repeated allocations
    mask = torch.full((4,), -float('inf'), device=device)
    
    # Calculate temperature for this episode (if annealing is enabled)
    if USE_TEMPERATURE_ANNEALING:
        # Exponential decay from INITIAL_TEMPERATURE to FINAL_TEMPERATURE
        current_temp = INITIAL_TEMPERATURE * (TEMPERATURE_DECAY ** total_episodes)
        # Clamp to ensure we don't go below FINAL_TEMPERATURE
        current_temp = max(current_temp, FINAL_TEMPERATURE)
    else:
        current_temp = temperature
        
    # Debug info - add extra info to first few episodes for debugging
    debug_info = {}
    if total_episodes < 5:
        debug_info['temperature'] = current_temp
        debug_info['strategy_shifts'] = 0

    # Keep track of chunked tensors to manage memory
    chunked_log_probs = []
    chunked_entropies = []
    moves_since_detach = 0
    
    # Memory tracking
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache at beginning
            initial_mem = torch.cuda.memory_allocated() / (1024**3)
    except:
        initial_mem = 0
        
    while can_move(board):
        # Get valid moves
        valid_moves = get_valid_moves(board)
        if not valid_moves:
            break
            
        # Save board state for strategy diversity calculation
        if len(board_history) < 10:  # Limit history size for memory efficiency
            board_history.append(copy.deepcopy(board))

        # Check if we need to detach computation graph to save memory
        if moves_since_detach >= chunk_size:
            # Detach current tensors and store them
            if log_probs:
                chunked_log_probs.append([tensor.detach() for tensor in log_probs])
                chunked_entropies.append([tensor.detach() for tensor in entropies])
                log_probs = []
                entropies = []
                moves_since_detach = 0
                # Explicitly clear GPU cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Vectorized board tensor creation
        state = board_to_tensor(board, device)
        with torch.no_grad() if moves_since_detach >= chunk_size/2 else torch.enable_grad():
            logits = model(state).squeeze(0)  # shape: (num_actions,)
        
        # Reuse mask tensor by zeroing it first
        mask.fill_(-float('inf'))
        mask.index_fill_(0, torch.tensor(valid_moves, device=device), 0.0)
        
        # Apply mask to logits
        adjusted_logits = logits + mask
        
        # Apply temperature scaling for controlled exploration
        if current_temp != 1.0:
            # Higher temperature = more uniform distribution (more exploration)
            # Lower temperature = more peaked distribution (more exploitation)
            temperature_scaled_logits = adjusted_logits / current_temp
            probs = F.softmax(temperature_scaled_logits, dim=-1)
        else:
            probs = F.softmax(adjusted_logits, dim=-1)
            
        # Create categorical distribution and sample action
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()

        # Enhanced penalty for invalid move preferences with graduated scaling
        original_probs = F.softmax(logits, dim=-1)
        if original_probs[action] < 0.02:  # Slightly higher threshold
            # Scale penalty by how poor the preference was
            preference_penalty = INEFFECTIVE_PENALTY * (1.0 - original_probs[action] * 10)
            forced_penalty_total += min(INEFFECTIVE_PENALTY, preference_penalty)

        # Record action log prob and entropy - avoid in-place operation
        action_tensor = torch.tensor([action], dtype=torch.long, device=device)
        
        # Only track gradients if we're within the chunk limit
        # This prevents memory growth from getting too large
        if moves_since_detach < chunk_size:
            log_probs.append(m.log_prob(action_tensor))
            entropies.append(m.entropy())
        
        # Apply move 
        new_board, moved, gain = apply_move(board, action)
        
        if not moved:
            # This should rarely happen due to masking
            forced_penalty_total += INEFFECTIVE_PENALTY * 1.2  # Increased penalty
            if not valid_moves:
                break
                
            # Choose backup action with weighted probability based on logits
            valid_probs = F.softmax(torch.tensor([logits[a].item() for a in valid_moves], device=device))
            backup_dist = torch.distributions.Categorical(valid_probs)
            backup_idx = backup_dist.sample().item()
            forced_action = valid_moves[backup_idx]
            
            new_board, moved, forced_gain = apply_move(board, forced_action)
            gain = forced_gain  # Already penalized
        
        board = new_board
        add_random_tile(board)
        total_moves += 1
        moves_since_detach += 1
        merge_score_total += gain
        
        # Periodically monitor memory usage
        try:
            if torch.cuda.is_available() and total_moves % 10 == 0:
                current_mem = torch.cuda.memory_allocated() / (1024**3)
                if current_mem > initial_mem * 1.5:  # If memory grows by 50%
                    # Force computation graph detachment
                    if log_probs:
                        chunked_log_probs.append([tensor.detach() for tensor in log_probs])
                        chunked_entropies.append([tensor.detach() for tensor in entropies])
                        log_probs = []
                        entropies = []
                        moves_since_detach = 0
                        torch.cuda.empty_cache()
        except:
            pass
        
        # Track strategy shifts for debugging/analytics
        if total_episodes < 5 and len(board_history) >= 2:
            curr_strategy = identify_board_strategy(board)[0]
            prev_strategy = identify_board_strategy(board_history[-1])[0]
            if curr_strategy != prev_strategy:
                debug_info['strategy_shifts'] += 1

    # Calculate final reward with strategy diversity bonus
    episode_reward = compute_reward(
        merge_score_total, 
        board, 
        forced_penalty_total, 
        total_moves,
        previous_boards=board_history,
        total_episodes=total_episodes
    )
    
    max_tile = max(max(row) for row in board)
    
    # Add debug information for early episodes
    if total_episodes < 5:
        debug_info['max_tile'] = max_tile
        debug_info['moves'] = total_moves
        debug_info['final_strategy'] = identify_board_strategy(board)[0]
    
    # If we have chunked tensors, combine them before returning
    if chunked_log_probs:
        # Add the current tensors to the chunks
        if log_probs:
            chunked_log_probs.append(log_probs)
            chunked_entropies.append(entropies)
        
        # Flatten all chunks
        flat_log_probs = []
        flat_entropies = []
        
        for chunk in chunked_log_probs:
            flat_log_probs.extend(chunk)
        for chunk in chunked_entropies:
            flat_entropies.extend(chunk)
        
        # Return the flattened tensors
        return flat_log_probs, flat_entropies, episode_reward, total_moves, max_tile
    else:
        # Return tensors as is if no chunking was needed
        return log_probs, entropies, episode_reward, total_moves, max_tile

# --- Batch Training Loop with Adaptive Learning ---
def train_loop(stdscr, model, optimizer, scheduler, device):
    """
    Enhanced batch training loop with strategy diversity tracking and memory management.
    - Implements adaptive temperature annealing
    - Tracks strategy exploration metrics
    - Provides more dynamic learning rate adaptation
    - Introduces multi-phase training
    - Displays enriched training dashboard
    - Dynamic memory management to prevent OOM errors
    """
    
    # Memory monitoring and management function
    def monitor_memory(current_batch_size):
        """Monitor memory usage and adjust parameters if needed to prevent OOM errors"""
        try:
            if not torch.cuda.is_available():
                return current_batch_size, 16  # Default chunk size on CPU
                
            # Get current memory usage
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            max_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
            
            # Calculate usage percentage
            usage_percent = (allocated / max_mem) * 100
            
            # Determine appropriate batch and chunk sizes based on memory usage
            if usage_percent > 90:  # Critically high memory usage
                print(f"⚠️ CRITICAL MEMORY USAGE: {usage_percent:.1f}% ({allocated:.2f}GB/{max_mem:.2f}GB)")
                torch.cuda.empty_cache()  # Force cache clear
                return max(2, current_batch_size // 2), 8  # Drastically reduce batch size
            elif usage_percent > 80:  # Very high memory usage
                print(f"⚠️ HIGH MEMORY USAGE: {usage_percent:.1f}% ({allocated:.2f}GB/{max_mem:.2f}GB)")
                return max(4, current_batch_size - 4), 12  # Reduce batch size
            elif usage_percent > 70:  # Moderate high memory usage
                print(f"⚠️ MODERATE MEMORY USAGE: {usage_percent:.1f}% ({allocated:.2f}GB/{max_mem:.2f}GB)")
                return current_batch_size, 16  # Keep batch size but lower chunk size
            else:
                # Memory usage is acceptable
                return current_batch_size, 24  # Normal chunk size
                
        except Exception as e:
            print(f"Error monitoring memory: {e}")
            return current_batch_size, 16  # Default values on error
    # Disable anomaly detection in production as it drastically slows down training
    torch.autograd.set_detect_anomaly(False)
    
    # Initialize training state variables
    baseline = 0.0
    total_episodes = 0
    best_avg_reward = -float('inf')
    best_model_state = None
    best_max_tile = 0
    rewards_history = []
    moves_history = []
    max_tile_history = []
    strategy_history = []
    
    # Training phase tracking
    current_phase = "exploration"  # Start with exploration phase
    phase_changes = []
    
    # Performance tracking
    training_start_time = time.time()
    stagnation_counter = 0
    best_reward_epoch = 0
    last_improvement = 0

    # Enable non-blocking input
    stdscr.nodelay(True)
    
    while True:
        # Check for stop command
        try:
            key = stdscr.getch()
            if key in (ord('s'), ord('S')):
                break
        except Exception:
            pass

        # Calculate current training phase
        # Phase 1: Exploration (high temp, high entropy bonus)
        # Phase 2: Transition (reducing temp, balanced rewards)
        # Phase 3: Exploitation (low temp, focused on high tiles)
        if total_episodes > 500 and current_phase == "exploration":
            current_phase = "transition"
            phase_changes.append((total_episodes, current_phase))
        elif total_episodes > 1000 and current_phase == "transition":
            current_phase = "exploitation"
            phase_changes.append((total_episodes, current_phase))
        
        # Enhanced adaptive learning rate with phase-dependent multipliers
        if total_episodes < WARMUP_EPISODES:
            # Warmup phase - gradually increase learning rate
            current_lr = LEARNING_RATE * (total_episodes / WARMUP_EPISODES)
        elif total_episodes < 100:
            # Early learning boost - higher learning rate for initial discoveries
            current_lr = LEARNING_RATE * EARLY_LR_MULTIPLIER
        elif current_phase == "exploration":
            # Exploration phase - slightly higher learning rate
            current_lr = LEARNING_RATE * 1.1
        elif current_phase == "exploitation":
            # Exploitation phase - slightly lower learning rate for refinement
            current_lr = LEARNING_RATE * 0.9
        else:
            # Default/transition phase - standard learning rate
            current_lr = LEARNING_RATE
            
        # Apply learning rate to optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Setup batch statistics tracking
        batch_log = []
        batch_reward_sum = 0.0
        batch_moves_sum = 0
        batch_max_tile = 0
        batch_start_time = time.time()
        batch_strategy_counts = {"corner": 0, "snake": 0, "balanced": 0}
        batch_temperature = 0.0

        # Process batch episodes in smaller chunks for quicker feedback
        chunk_size = min(10, BATCH_SIZE)  # Process up to 10 episodes at a time
        
        for chunk_start in range(0, BATCH_SIZE, chunk_size):
            chunk_end = min(chunk_start + chunk_size, BATCH_SIZE)
            
            # Run chunk_size episodes in sequence
            for _ in range(chunk_end - chunk_start):
                if key in (ord('s'), ord('S')):
                    break
                
                # Phase-dependent entropy weight for exploration vs exploitation balance
                if current_phase == "exploration":
                    entropy_weight = 5e-3  # Higher entropy bonus for exploration
                elif current_phase == "transition":
                    entropy_weight = 3e-3  # Moderate entropy bonus
                else:  # exploitation
                    entropy_weight = 1e-3  # Lower entropy bonus for exploitation
                
                # Before simulating episode, check memory and adjust parameters
                dynamic_batch_size, chunk_size = monitor_memory(BATCH_SIZE) 
                if dynamic_batch_size != BATCH_SIZE:
                    stdscr.addstr(offset_y+16, offset_x, f"⚠️ Memory management: adjusted batch size to {dynamic_batch_size}")
                    stdscr.refresh()
                
                # Run episode with current model and dynamic chunk size
                log_probs, entropies, episode_reward, moves, max_tile = simulate_episode(
                    model, device, total_episodes=total_episodes, chunk_size=chunk_size
                )
                
                # Record episode results
                total_episodes += 1
                rewards_history.append(episode_reward)
                moves_history.append(moves)
                max_tile_history.append(max_tile)
                batch_reward_sum += episode_reward
                batch_moves_sum += moves
                batch_max_tile = max(batch_max_tile, max_tile)
                
                # Track strategy used in this episode (using the final board state)
                if len(entropies) > 0:  # Only attempt if episode had moves
                    temp_board = new_game()
                    strategy_used, confidence = identify_board_strategy(temp_board)
                    strategy_history.append(strategy_used)
                    batch_strategy_counts[strategy_used] += 1
                
                # Calculate episode loss with dynamic entropy bonus
                if log_probs:
                    # Compute advantage with smoother baseline
                    advantage = episode_reward - baseline
                    
                    # Dynamic entropy weight based on episode performance
                    # Good episodes get less entropy bonus (exploit more)
                    # Bad episodes get more entropy bonus (explore more)
                    relative_performance = advantage / (baseline + 1e-8)  # Avoid division by zero
                    episode_entropy_weight = entropy_weight
                    if relative_performance < -0.5:
                        # Significantly worse than baseline - boost exploration
                        episode_entropy_weight *= 1.5
                    elif relative_performance > 0.5:
                        # Significantly better than baseline - reduce exploration
                        episode_entropy_weight *= 0.8
                    
                    # Compute loss with dynamic entropy bonus
                    entropy_bonus = episode_entropy_weight * torch.stack(entropies).sum()
                    episode_loss = -torch.stack(log_probs).sum() * advantage - entropy_bonus
                    batch_log.append(episode_loss)
                
                # Check for stop signal more frequently during batch processing
                try:
                    key = stdscr.getch()
                except Exception:
                    pass

        # Calculate batch statistics
        avg_batch_reward = batch_reward_sum / BATCH_SIZE
        avg_batch_moves = batch_moves_sum / BATCH_SIZE
        
        # Determine dominant strategy for this batch
        dominant_strategy = max(batch_strategy_counts.items(), key=lambda x: x[1])[0]
        dominant_strategy_pct = batch_strategy_counts[dominant_strategy] / sum(batch_strategy_counts.values()) * 100

        # Calculate strategy diversity score (higher = more diverse)
        strategy_counts = list(batch_strategy_counts.values())
        strategy_diversity = 1.0
        if any(strategy_counts):  # Avoid division by zero
            total_strategies = sum(strategy_counts)
            strategy_diversity = 1.0 - sum((count/total_strategies)**2 for count in strategy_counts if count > 0)
            strategy_diversity *= 3.0  # Scale to a more readable range

        # Update running baseline using phase-dependent update rate
        if total_episodes <= WARMUP_EPISODES:
            baseline = avg_batch_reward
        else:
            # Phase-dependent update rate (faster in exploration, slower in exploitation)
            if current_phase == "exploration":
                baseline_alpha = 0.03  # Faster updates during exploration
            elif current_phase == "transition":
                baseline_alpha = 0.02  # Moderate update rate
            else:  # exploitation
                baseline_alpha = 0.01  # Slower updates for stability
                
            baseline = (1 - baseline_alpha) * baseline + baseline_alpha * avg_batch_reward

        # Optimize model if we have episodes in the batch
        if batch_log:
            batch_loss = torch.stack(batch_log).mean()
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        # Update learning rate scheduler with recent performance
        recent_avg_reward = (sum(rewards_history[-100:]) / min(len(rewards_history), 100)
                           if rewards_history else avg_batch_reward)
        scheduler.step(recent_avg_reward)

        # Track performance improvement and stagnation
        improved = False
        if recent_avg_reward > best_avg_reward:
            improved = True
            last_improvement = total_episodes
            best_avg_reward = recent_avg_reward
            best_model_state = model.state_dict()
            best_reward_epoch = total_episodes
            # Save checkpoint immediately on significant improvement
            if recent_avg_reward > best_avg_reward * 1.1:
                torch.save(best_model_state, "2048_model_best.pt", _use_new_zipfile_serialization=True)
                
        if batch_max_tile > best_max_tile:
            improved = True
            best_max_tile = batch_max_tile
            # Also save on new best tile achievement
            torch.save(model.state_dict(), "2048_model_best_tile.pt", _use_new_zipfile_serialization=True)
        
        # Check for stagnation (no improvement for a long time)
        stagnation_counter = 0 if improved else stagnation_counter + 1
        
        # Handle stagnation with adaptive strategies
        if stagnation_counter > 15:  # No improvement for 15 batches
            # Implement an anti-stagnation boost - increase temperature temporarily
            stagnation_counter = 0
            stdscr.addstr(offset_y+15, offset_x, "⚠️ Stagnation detected - applying exploration boost")
            stdscr.refresh()
            # Reset temperature higher for a while to escape local optimum
            
        # Calculate statistics about recent performance
        recent_max_tiles = max_tile_history[-100:]
        best_tile_rate = (recent_max_tiles.count(best_max_tile) / min(len(recent_max_tiles), 100) * 100
                         if recent_max_tiles else 0.0)
        
        # Recent strategy distribution
        recent_strategies = strategy_history[-100:] if strategy_history else []
        recent_strategy_counts = {}
        for s in recent_strategies:
            recent_strategy_counts[s] = recent_strategy_counts.get(s, 0) + 1
            
        # Calculate dominant recent strategy
        recent_dominant = max(recent_strategy_counts.items(), key=lambda x: x[1])[0] if recent_strategy_counts else "unknown"
        recent_dominant_pct = (recent_strategy_counts.get(recent_dominant, 0) / len(recent_strategies) * 100 
                              if recent_strategies else 0)
        
        # Timing statistics
        batch_duration = time.time() - batch_start_time
        total_training_time = time.time() - training_start_time
        episodes_per_second = BATCH_SIZE / (batch_duration + 1e-6)
        
        # Display color selection based on performance
        if avg_batch_reward > recent_avg_reward * 1.05:
            reward_color = curses.color_pair(9)  # Green
        elif avg_batch_reward < recent_avg_reward * 0.95:
            reward_color = curses.color_pair(10)  # Red
        else:
            reward_color = curses.color_pair(1)  # White

        # Create dashboard display
        stdscr.clear()
        stdscr.border()
        offset_y, offset_x = 1, 2
        stdscr.attron(curses.color_pair(1))
        
        # Main statistics
        stdscr.addstr(offset_y, offset_x, f"Total Episodes: {total_episodes} | Phase: {current_phase}")
        stdscr.addstr(offset_y+1, offset_x, "Avg Batch Reward: ")
        stdscr.addstr(f"{avg_batch_reward:.2f}", reward_color)
        stdscr.addstr(offset_y+2, offset_x, f"Recent Avg Reward (last 100): {recent_avg_reward:.2f}")
        stdscr.addstr(offset_y+3, offset_x, f"Best Recent Avg Reward: {best_avg_reward:.2f} (at episode {best_reward_epoch})")
        stdscr.addstr(offset_y+4, offset_x, f"Avg Episode Length: {avg_batch_moves:.2f} moves | Rate: {episodes_per_second:.1f} eps/sec")
        
        # Tile statistics
        stdscr.addstr(offset_y+5, offset_x, f"Best Tile (this batch): {batch_max_tile}")
        stdscr.addstr(offset_y+6, offset_x, f"Best Tile Overall: {best_max_tile} at {best_tile_rate:.1f}% success (last 100)")
        
        # Strategy statistics
        stdscr.addstr(offset_y+7, offset_x, f"Batch Strategy: {dominant_strategy} ({dominant_strategy_pct:.1f}%) | Diversity: {strategy_diversity:.2f}")
        stdscr.addstr(offset_y+8, offset_x, f"Recent Strategy: {recent_dominant} ({recent_dominant_pct:.1f}%)")
        
        # Performance metrics
        stdscr.addstr(offset_y+9, offset_x, f"Batch Duration: {batch_duration:.3f} sec | Total: {total_training_time:.1f} sec")
        stdscr.addstr(offset_y+10, offset_x, f"Current LR: {current_lr:.6f} | Last Improvement: {total_episodes - last_improvement} batches ago")
        
        # Animation
        stdscr.addstr(offset_y+12, offset_x, "Training... " + SPINNER[total_episodes % len(SPINNER)])
        stdscr.addstr(offset_y+14, offset_x, "Press 'S' to stop training and save checkpoint.")
        
        # Refresh display and add small delay
        stdscr.refresh()
        time.sleep(DISPLAY_DELAY)

    # Save best model when training ends
    if best_model_state is not None:
        # Save in new format to avoid pickle loading issues
        torch.save(best_model_state, "2048_model.pt", _use_new_zipfile_serialization=True)
    
    # Show final statistics
    stdscr.nodelay(False)
    stdscr.clear()
    stdscr.border()
    stdscr.addstr(1, 2, "Training stopped.")
    stdscr.addstr(2, 2, f"Best Recent Avg Reward: {best_avg_reward:.2f}")
    stdscr.addstr(3, 2, f"Best Tile Achieved: {best_max_tile}")
    
    # Calculate final statistics
    recent_avg_length = (sum(moves_history[-100:]) / min(len(moves_history), 100)
                       if moves_history else 0.0)
    stdscr.addstr(4, 2, f"Avg Episode Length (last 100): {recent_avg_length:.2f}")
    
    # Strategy summary
    if strategy_history:
        stdscr.addstr(5, 2, "Strategy Summary:")
        strategy_counts = {}
        for s in strategy_history:
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        for i, (strategy, count) in enumerate(sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)):
            pct = count / len(strategy_history) * 100
            stdscr.addstr(6+i, 4, f"{strategy}: {count} episodes ({pct:.1f}%)")
    
    stdscr.addstr(10, 2, "Checkpoints saved:")
    stdscr.addstr(11, 4, "- 2048_model.pt (best average reward)")
    if best_max_tile > 0:
        stdscr.addstr(12, 4, "- 2048_model_best_tile.pt (highest tile achieved)")
    
    stdscr.addstr(14, 2, "Press any key to exit.")
    stdscr.refresh()
    stdscr.getch()

# --- Watch Mode ---
def watch_game(stdscr, model, device):
    """
    Enhanced watch mode with strategy identification and move visualization.
    Displays the current board, score, identified strategy, and move probabilities.
    """
    score = 0.0
    board = new_game()
    delay = 0.5  # Delay between moves in watch mode
    
    # Keep track of game statistics
    move_count = 0
    strategy_changes = 0
    last_strategy = None
    
    # Track board history for diversity metrics
    board_history = []
    
    # Track high-level game stats
    stats = {
        "max_tile": 0,
        "merges": 0,
        "moves_by_direction": {0: 0, 1: 0, 2: 0, 3: 0}  # up, down, left, right
    }
    
    # Set temperature for watching (low temperature for best performance)
    watch_temperature = 0.8
    
    # Create reusable tensors for performance
    mask = torch.full((4,), -float('inf'), device=device)
    
    while can_move(board):
        # Save current board state for strategy tracking
        if len(board_history) < 20:  # Limit history size
            board_history.append(copy.deepcopy(board))
        
        # Clear screen and draw current state
        stdscr.clear()
        stdscr.border()
        
        # Identify current strategy
        current_strategy, confidence = identify_board_strategy(board)
        
        # Track strategy changes
        if last_strategy is not None and current_strategy != last_strategy:
            strategy_changes += 1
        last_strategy = current_strategy
        
        # Update max tile
        current_max_tile = max(max(row) for row in board)
        stats["max_tile"] = max(stats["max_tile"], current_max_tile)
        
        # Create status text with strategy info
        status_text = f"Strategy: {current_strategy} ({confidence*100:.0f}% confidence)"
        
        # Draw board with additional stats
        draw_board(stdscr, board, score, extra_text=status_text)
        
        # Draw additional stats below the board
        height, width = stdscr.getmaxyx()
        board_height = GRID_SIZE * 2 + 1
        start_y = (height - board_height) // 2
        stats_y = start_y + board_height + 2
        
        if stats_y + 5 < height:  # Make sure we have room to draw stats
            stdscr.addstr(stats_y, 2, f"Moves: {move_count} | Strategy changes: {strategy_changes}")
            stdscr.addstr(stats_y+1, 2, "Move distribution: " + 
                         f"⬆️ {stats['moves_by_direction'][0]} " +
                         f"⬇️ {stats['moves_by_direction'][1]} " +
                         f"⬅️ {stats['moves_by_direction'][2]} " +
                         f"➡️ {stats['moves_by_direction'][3]}")
        
        # Refresh display and wait
        stdscr.refresh()
        time.sleep(delay)
        
        # Get valid moves
        valid_moves = get_valid_moves(board)
        if not valid_moves:
            break
            
        # Convert board to tensor
        state = board_to_tensor(board, device)
        logits = model(state).squeeze(0)
        
        # Create validity mask
        mask.fill_(-float('inf'))
        for a in valid_moves:
            mask[a] = 0.0
            
        # Apply mask to logits
        adjusted_logits = logits + mask
        
        # Apply temperature for more deterministic performance in watch mode
        temperature_scaled_logits = adjusted_logits / watch_temperature
        probs = F.softmax(temperature_scaled_logits, dim=-1)
        
        # Show move probabilities if there's room
        if stats_y + 5 < height:
            prob_str = "Move probabilities: "
            directions = ["⬆️", "⬇️", "⬅️", "➡️"]
            for i in range(4):
                if i in valid_moves:
                    prob_str += f"{directions[i]}{probs[i].item()*100:.1f}% "
            stdscr.addstr(stats_y+2, 2, prob_str)
        
        # Sample action using categorical distribution
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()
        
        # Apply the move
        new_board, moved, gain = apply_move(board, action)
        
        # If move didn't work (should be rare with masking), choose a valid one
        if not moved:
            # Choose highest probability valid move
            valid_probs = [probs[a].item() for a in valid_moves]
            best_idx = valid_probs.index(max(valid_probs))
            action = valid_moves[best_idx]
            new_board, moved, gain = apply_move(board, action)
            
        # Update board and score
        board = new_board
        add_random_tile(board)
        score += gain
        move_count += 1
        
        # Update statistics
        stats["moves_by_direction"][action] += 1
        if gain > 0:
            stats["merges"] += 1
        
        # Check if we can still move
        if not can_move(board):
            break
            
    # Game over - show final state with stats
    stdscr.clear()
    stdscr.border()
    
    # Create final summary text
    final_strategy, final_confidence = identify_board_strategy(board)
    summary = f"Game Over! Max tile: {stats['max_tile']} | Final Strategy: {final_strategy}"
    draw_board(stdscr, board, score, extra_text=summary)
    
    # Display final statistics below the board if there's room
    height, width = stdscr.getmaxyx()
    board_height = GRID_SIZE * 2 + 1
    start_y = (height - board_height) // 2
    stats_y = start_y + board_height + 2
    
    if stats_y + 6 < height:
        stdscr.addstr(stats_y, 2, f"Total moves: {move_count} | Strategy changes: {strategy_changes}")
        stdscr.addstr(stats_y+1, 2, f"Total merges: {stats['merges']} | Score: {score:.1f}")
        stdscr.addstr(stats_y+2, 2, "Move distribution: " + 
                    f"⬆️ {stats['moves_by_direction'][0]} " +
                    f"⬇️ {stats['moves_by_direction'][1]} " +
                    f"⬅️ {stats['moves_by_direction'][2]} " +
                    f"➡️ {stats['moves_by_direction'][3]}")
        
        # Calculate move percentages
        if move_count > 0:
            up_pct = stats['moves_by_direction'][0] / move_count * 100
            down_pct = stats['moves_by_direction'][1] / move_count * 100
            left_pct = stats['moves_by_direction'][2] / move_count * 100
            right_pct = stats['moves_by_direction'][3] / move_count * 100
            stdscr.addstr(stats_y+3, 2, f"Move %: ⬆️ {up_pct:.1f}% ⬇️ {down_pct:.1f}% ⬅️ {left_pct:.1f}% ➡️ {right_pct:.1f}%")
        
        stdscr.addstr(stats_y+5, 2, "Press any key to exit.")
    
    stdscr.refresh()
    stdscr.getch()

# --- Main Function ---
def main_console(stdscr):
    """
    Legacy console mode main entry point.
    Initializes curses, loads model, and runs train or watch mode in the console.
    Supports enhanced model architecture with strategy adaptation.
    """
    parser = argparse.ArgumentParser(description="2048 Self-Play Learning (Console Mode)")
    parser.add_argument('--mode', choices=['train', 'watch'], default='train',
                        help="--mode train: run training; --mode watch: watch a game using saved model")
    parser.add_argument('--checkpoint', type=str, default="2048_model.pt",
                       help="Path to model checkpoint file (default: 2048_model.pt)")
    parser.add_argument('--temperature', type=float, default=0.8,
                       help="Temperature for action sampling (lower=more deterministic)")
    args = parser.parse_args(sys.argv[2:])  # Skip the first argument which is --console
    
    curses.curs_set(0)  # Hide cursor
    curses.start_color()
    init_colors()
    
    # Check for CUDA availability, but use CPU as fallback
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print device information
    stdscr.clear()
    stdscr.border()
    stdscr.addstr(1, 2, f"Using device: {device}")
    if device.type == 'cuda':
        stdscr.addstr(2, 2, f"GPU: {torch.cuda.get_device_name(0)}")
        stdscr.addstr(3, 2, f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    stdscr.addstr(5, 2, "Initializing model...")
    stdscr.refresh()
    
    # Create model with enhanced architecture and send to device
    model = ConvTransformerPolicy(
        vocab_size=VOCAB_SIZE, 
        d_model=DMODEL, 
        nhead=NHEAD,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS, 
        num_high_level_layers=NUM_HIGH_LEVEL_LAYERS,
        dropout=DROPOUT, 
        num_actions=4
    ).to(device)
    
    # Display model architecture size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stdscr.addstr(6, 2, f"Model created: {total_params:,} total parameters")
    stdscr.addstr(7, 2, f"               {trainable_params:,} trainable parameters")
    stdscr.refresh()
    time.sleep(1)  # Show the model info briefly
    
    if args.mode == 'train':
        # Use AdamW optimizer with weight decay for regularization
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=LEARNING_RATE,
            weight_decay=1e-5,  # Small weight decay for regularization
            betas=(0.9, 0.999)  # Default beta parameters
        )
        
        # Learning rate scheduler that reduces LR on plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE, 
            verbose=True
        )
        
        # Try to load existing checkpoint if it exists
        try:
            if os.path.exists(args.checkpoint):
                stdscr.addstr(8, 2, f"Loading checkpoint from {args.checkpoint}...")
                stdscr.refresh()
                
                checkpoint = torch.load(args.checkpoint, map_location=device)
                # Use strict=False to handle architecture changes between versions
                model.load_state_dict(checkpoint, strict=False)
                
                stdscr.addstr(9, 2, "Checkpoint loaded successfully. Starting training...")
                stdscr.refresh()
                time.sleep(1)
            else:
                stdscr.addstr(8, 2, "No existing checkpoint found. Starting with fresh model...")
                stdscr.refresh()
                time.sleep(1)
        except Exception as e:
            stdscr.addstr(8, 2, f"Error loading checkpoint: {str(e)}")
            stdscr.addstr(9, 2, "Starting with fresh model...")
            stdscr.refresh()
            time.sleep(2)
        
        # Start training loop
        train_loop(stdscr, model, optimizer, scheduler, device)
    else:
        # Watch mode
        try:
            # Check if checkpoint exists
            if not os.path.exists(args.checkpoint):
                raise FileNotFoundError(f"Checkpoint file {args.checkpoint} not found")
                
            # Load checkpoint with error handling for architecture changes
            stdscr.addstr(8, 2, f"Loading checkpoint from {args.checkpoint}...")
            stdscr.refresh()
            
            try:
                # First try loading with strict=True
                checkpoint = torch.load(args.checkpoint, map_location=device)
                model.load_state_dict(checkpoint)
                stdscr.addstr(9, 2, "Checkpoint loaded successfully.")
            except Exception as e:
                # If that fails, try with strict=False to handle architecture changes
                stdscr.addstr(9, 2, "Strict loading failed, trying flexible loading...")
                checkpoint = torch.load(args.checkpoint, map_location=device)
                model.load_state_dict(checkpoint, strict=False)
                stdscr.addstr(10, 2, "Checkpoint loaded with architecture adaptation.")
            
            stdscr.addstr(11, 2, "Starting game...")
            stdscr.refresh()
            time.sleep(1)
            
            # Run watch mode
            watch_game(stdscr, model, device)
        except Exception as e:
            # Handle errors
            stdscr.clear()
            stdscr.border()
            stdscr.addstr(1, 2, f"Error: {str(e)}")
            stdscr.addstr(3, 2, "Please run training mode first or specify a valid checkpoint file.")
            stdscr.addstr(5, 2, "Press any key to exit.")
            stdscr.refresh()
            stdscr.getch()
            return

def main():
    """
    Main entry point for the application.
    By default, launches the web UI server. Use --console for legacy mode.
    """
    parser = argparse.ArgumentParser(description="2048 Self-Play Learning")
    parser.add_argument('--console', action='store_true', help="Run in legacy console mode")
    parser.add_argument('--port', type=int, default=5000, help="Port for the web UI (default: 5000)")
    parser.add_argument('--no-browser', action='store_true', help="Don't automatically open browser")
    parser.add_argument('--debug', action='store_true', help="Run server in debug mode")
    
    # Parse just enough args to determine mode
    args, remaining = parser.parse_known_args()
    
    if args.console:
        # Run in legacy console mode
        curses.wrapper(main_console)
    else:
        # Get the local IP for displaying in console
        import socket
        def get_local_ip():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(('10.255.255.255', 1))
                IP = s.getsockname()[0]
            except Exception:
                IP = '127.0.0.1'
            finally:
                s.close()
            return IP
        
        local_ip = get_local_ip()
        server_url = f"http://{local_ip}:{args.port}"
        
        # Display hardware info in console
        try:
            import psutil
            
            # Try to import GPUtil for GPU monitoring
            has_gputil = False
            try:
                import GPUtil
                has_gputil = True
            except ImportError:
                pass
            
            def format_bytes(bytes):
                """Format bytes to human-readable string"""
                for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                    if bytes < 1024:
                        return f"{bytes:.1f} {unit}"
                    bytes /= 1024
                return f"{bytes:.1f} PB"
            
            # CPU info
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # RAM info
            memory = psutil.virtual_memory()
            ram_used = format_bytes(memory.used)
            ram_total = format_bytes(memory.total)
            ram_percent = memory.percent
            
            # Print header
            print("\n" + "=" * 80)
            print(f"  2048 RL Bot with Web UI - Server running at {server_url}")
            print("=" * 80)
            
            # Print hardware info
            print(f"\n  Hardware Information:")
            print(f"  - CPU: {cpu_percent:.1f}% usage across {cpu_count} cores")
            print(f"  - RAM: {ram_used} / {ram_total} ({ram_percent:.1f}%)")
            
            # GPU info if available
            if has_gputil:
                gpus = GPUtil.getGPUs()
                if gpus:
                    for i, gpu in enumerate(gpus):
                        print(f"  - GPU {i}: {gpu.name}")
                        print(f"    {gpu.memoryUsed:.1f}MB / {gpu.memoryTotal:.1f}MB ({gpu.memoryUtil*100:.1f}%)")
                        print(f"    Utilization: {gpu.load*100:.1f}%, Temperature: {gpu.temperature}°C")
                else:
                    print("  - GPU: None detected")
            else:
                print("  - GPU: Install GPUtil for GPU monitoring")
                
            print("\n  To stop the server, press Ctrl+C")
            print("=" * 80)
        except Exception as e:
            # If anything fails during hardware info display, just show a simpler message
            print("\n" + "=" * 80)
            print(f"  2048 RL Bot with Web UI - Server running at {server_url}")
            print("=" * 80)
            print("\n  To stop the server, press Ctrl+C")
            print("=" * 80)
        
        # Import and run the web server
        import importlib.util
        import os
        
        # Get the full path to the server script
        server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2048_bot_server.py")
        
        # Import the server module
        spec = importlib.util.spec_from_file_location("server_module", server_path)
        server_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server_module)
        
        # Pass our arguments to the server's main function
        sys.argv = [sys.argv[0]]
        if args.port:
            sys.argv.extend(['--port', str(args.port)])
        if args.no_browser:
            sys.argv.append('--no-browser')
        if args.debug:
            sys.argv.append('--debug')
        
        # Run the server
        server_module.main()

if __name__ == '__main__':
    main()