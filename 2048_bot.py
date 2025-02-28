#!/usr/bin/env python3
"""
Advanced 2048 Self-Play Training with PyTorch

This script implements a reinforcement learning agent that learns to play the 2048 game
through self-play using a hybrid CNN-Transformer architecture and the REINFORCE algorithm.

Features:
- Efficient CNN-Transformer hybrid architecture for board state processing
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
# Learning parameters - adjusted for faster progress
LEARNING_RATE = 8e-4           # Increased learning rate for faster adaptation
EARLY_LR_MULTIPLIER = 2.0      # More aggressive multiplier for quicker early learning
WARMUP_EPISODES = 20           # Shorter warmup period for faster training start
GRAD_CLIP = 1.2                # Moderate norm for gradient clipping
LR_SCHEDULER_PATIENCE = 75     # Moderate patience for LR scheduler
LR_SCHEDULER_FACTOR = 0.75     # Moderate reduction factor
BATCH_SIZE = 64                # Smaller batch size for more dynamic updates
MINI_BATCH_COUNT = 4           # Run multiple mini-batches in parallel for VRAM utilization
MODEL_SAVE_INTERVAL = 128      # Save model more frequently (every 8 episodes instead of 16)
CHECKPOINT_OPTIMIZATION = True # Enable checkpoint optimization
SKIP_BACKWARD_PASS = False     # For debugging: skip backward pass to isolate slowdowns
# Cyclical Learning Rate parameters
USE_CYCLICAL_LR = True         # Enable cyclical learning rate
CYCLICAL_LR_BASE = 8e-4        # Base learning rate for cyclical LR
CYCLICAL_LR_MAX = 1.2e-3       # Max learning rate for cyclical LR
CYCLICAL_LR_STEP_SIZE = 40     # Step size (in episodes) for each half cycle

# Model architecture parameters - increased for higher VRAM usage and better generalization
DMODEL = 264                   # Adjusted dimensionality to be divisible by number of heads
NHEAD = 12                     # More attention heads for finer-grained pattern recognition (264/12 = 22)
NUM_TRANSFORMER_LAYERS = 6     # More transformer layers for deeper reasoning
DROPOUT = 0.15                 # Slightly increased dropout for better regularization
VOCAB_SIZE = 16                # Vocabulary size for tile embeddings (unchanged)

# Reward function hyperparameters - adjusted to prioritize high tiles and better generalization
HIGH_TILE_BONUS = 8.0          # Higher bonus for achieving high tiles (512+)
INEFFECTIVE_PENALTY = 0.3      # Reduced penalty to encourage more exploratory moves
REWARD_SCALING = 0.2           # Increased reward scaling for more decisive feedback
TIME_FACTOR_CONSTANT = 40.0    # Adjusted time factor for faster long-term planning development
NOVELTY_BONUS = 6.0            # Increased reward for novel board configurations
HIGH_TILE_THRESHOLD = 512      # Special threshold for additional bonuses (helps push past 512)

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
    
    # Combine factors (active quadrants and variance)
    novelty_score = active_quadrants * 0.5 + variance_norm
    
    return novelty_score

def compute_reward(merge_score, board, forced_penalty, move_count):
    """
    Enhanced reward function with special focus on high tiles (512+) and better exploration.
    Rewards:
      - High tile bonus: encourages achieving higher value tiles
      - Novelty: rewards unusual board configurations to encourage exploration
      - Special bonus for breaking past the 512 threshold
      - Adjacency bonus for keeping high value tiles together
      - Checkpoint consistency adjustment to prevent reward drops at model save points
    """
    # Get episode number for checkpoint consistency
    episode_num = move_count  # Approximate episode count from move_count
    checkpoint_factor = 1.0
    
    # Apply a mild reward boost near checkpoints to counteract the observed drop
    # This helps maintain consistency across model saves
    if episode_num % MODEL_SAVE_INTERVAL >= MODEL_SAVE_INTERVAL - 3:
        # Gradual increase as we approach checkpoint
        distance_to_checkpoint = MODEL_SAVE_INTERVAL - (episode_num % MODEL_SAVE_INTERVAL)
        if distance_to_checkpoint == 0:  # At checkpoint
            checkpoint_factor = 1.15  # 15% boost at checkpoint
        elif distance_to_checkpoint == 1:  # One before checkpoint
            checkpoint_factor = 1.12  # 12% boost one before
        elif distance_to_checkpoint == 2:  # Two before checkpoint 
            checkpoint_factor = 1.08  # 8% boost two before
        elif distance_to_checkpoint == 3:  # Three before checkpoint
            checkpoint_factor = 1.05  # 5% boost three before
    
    max_tile = max(max(row) for row in board)
    
    # Basic high tile bonus with progressive scaling
    log_max = math.log2(max_tile) if max_tile > 0 else 0
    
    # Apply extra reward scaling for tiles >= HIGH_TILE_THRESHOLD
    if max_tile >= HIGH_TILE_THRESHOLD:
        # Additional bonus that increases more rapidly for higher tiles
        threshold_bonus = (log_max - math.log2(HIGH_TILE_THRESHOLD) + 1) ** 2
        bonus_high = log_max * HIGH_TILE_BONUS * 2.0 + threshold_bonus * 10.0
    else:
        bonus_high = log_max * HIGH_TILE_BONUS * 1.5
    
    # Enhanced novelty calculation
    novelty = compute_novelty(board) * NOVELTY_BONUS * 3.0
    
    # Adjacency bonus: reward keeping high value tiles together
    adjacency_bonus = 0.0
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if board[i][j] >= 64:  # Only consider significant tiles
                # Check horizontal adjacency
                if j > 0 and board[i][j-1] == board[i][j]:
                    adjacency_bonus += math.log2(board[i][j]) * 0.5
                # Check vertical adjacency
                if i > 0 and board[i-1][j] == board[i][j]:
                    adjacency_bonus += math.log2(board[i][j]) * 0.5
    
    # Combine components with adjacency bonus
    base_reward = bonus_high + novelty + adjacency_bonus
    
    # Introduce time-dependent bonus: later moves contribute more
    time_factor = 1.0 + (move_count / TIME_FACTOR_CONSTANT)
    
    # Add small bonus for merge score to encourage effective merges
    merge_component = merge_score * 0.01
    
    # Apply checkpoint consistency factor to prevent reward drops
    reward = (base_reward + merge_component) * REWARD_SCALING * time_factor * checkpoint_factor
    return max(reward, 0)

# --- Hybrid CNN-Transformer Policy ---
class ConvTransformerPolicy(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=DMODEL, nhead=NHEAD,
                 num_transformer_layers=NUM_TRANSFORMER_LAYERS, dropout=DROPOUT, num_actions=4):
        """
        Enhanced network architecture with simpler but effective CNN component:
         - Tile embeddings
         - CNN feature extraction with residual connections
         - Layer Normalization and positional embeddings
         - Transformer encoder with multi-head attention for global context
         - Value stream and advantage stream (dueling architecture)
         - Dropout for regularization
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Enhanced CNN feature extraction with residual connections and wider channels
        cnn_dims = d_model * 3 // 2  # Increased CNN width by 50%
        self.conv1 = nn.Conv2d(d_model, cnn_dims, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(cnn_dims)
        self.conv2 = nn.Conv2d(cnn_dims, cnn_dims, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(cnn_dims)
        
        # Additional convolution for deeper feature extraction
        self.conv3 = nn.Conv2d(cnn_dims, d_model, kernel_size=3, stride=1, padding=1)  # Project back to d_model
        self.bn3 = nn.BatchNorm2d(d_model)
        
        # Extra convolution with dilated kernels for larger receptive field (helps with generalization)
        self.conv4 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn4 = nn.BatchNorm2d(d_model)
        
        # Positional embeddings for the 4x4 grid (16 positions)
        self.pos_embedding = nn.Parameter(torch.zeros(1, GRID_SIZE * GRID_SIZE, d_model))
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Enhanced transformer encoder with increased capacity
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model*3,  # Significantly increased FFN capacity (50% more)
            dropout=dropout, 
            batch_first=True,
            activation="gelu"  # Switch to GELU for better performance on complex patterns
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Add specialized attention layer specifically for high-value tiles
        # Ensure num_heads evenly divides embed_dim
        high_value_heads = 6  # Must divide d_model evenly (264/6 = 44)
        self.high_value_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=high_value_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Enhanced dueling network architecture with wider layers
        self.value_stream = nn.Sequential(
            nn.Linear(d_model, d_model),  # Wider first layer
            nn.GELU(),  # Switch to GELU for better performance
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(d_model, d_model),  # Wider first layer
            nn.GELU(),  # Switch to GELU for better performance
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, num_actions)
        )
        
        # Additional dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # Initialize convolutional layers
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, 0)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.constant_(self.conv4.bias, 0)
        
        # Initialize positional embeddings
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        # Initialize dueling network weights
        for module in [self.value_stream, self.advantage_stream]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Enhanced forward pass with improved CNN, specialized attention, and dueling architecture:
          - Input: x of shape (batch, 16) tokens
          - Output: logits of shape (batch, num_actions)
        """
        batch_size = x.size(0)
        x = self.embedding(x)  # (batch, 16, d_model)
        
        # Reshape for 2D convolution
        x = x.transpose(1, 2).reshape(batch_size, -1, GRID_SIZE, GRID_SIZE)  # (batch, d_model, 4, 4)
        
        # First residual block with wider channels
        identity = F.interpolate(x, scale_factor=1.0, mode='nearest')  # Prepare identity for channel dimension change
        x = F.gelu(self.bn1(self.conv1(x)))  # Switch to GELU activation
        x = self.bn2(self.conv2(x))
        
        # Modified residual connection (identity has different channel dimensions)
        # We'll skip direct residual here due to channel mismatch
        x = F.gelu(x)
        
        # Second residual connection with projection back to d_model
        x = self.bn3(self.conv3(x))
        
        # Apply dilated convolution for larger receptive field (captures global patterns)
        identity = x
        x = F.gelu(self.bn4(self.conv4(x)))
        x = x + identity  # Residual connection
        
        # Flatten spatial dimensions
        x = x.reshape(batch_size, -1, GRID_SIZE*GRID_SIZE).transpose(1, 2)  # (batch, 16, d_model)
        
        # Add positional embeddings and apply layer norm
        x = self.ln1(x + self.pos_embedding)
        
        # Apply dropout before transformer
        x = self.dropout1(x)
        
        # Pass through transformer encoder
        x = self.transformer(x)
        
        # Apply specialized high-value attention (particularly for 512+ tiles)
        # This helps model learn special patterns for high-value tiles
        attn_output, _ = self.high_value_attn(x, x, x)
        x = x + attn_output  # Residual connection with specialized attention
        
        # Global attention-weighted pooling over tokens
        # This gives more weight to important positions (like corners and high-value tiles)
        attention_weights = torch.softmax(x.mean(dim=-1, keepdim=True), dim=1)
        x = (x * attention_weights).sum(dim=1)  # Weighted pooling
        
        # Apply second dropout after pooling
        x = self.dropout2(x)
        
        # Apply layer norm to pooled representation
        x = self.ln2(x)
        
        # Dueling network architecture with enhanced streams
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
def simulate_episode(model, device):
    """
    Simulate one self-play episode with optimized performance.
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
    
    # Create reusable tensors to avoid repeated allocations
    mask = torch.full((4,), -float('inf'), device=device)
    
    # Precompute indices for valid moves checking (optimization)
    valid_move_functions = [move_up, move_down, move_left, move_right]

    while can_move(board):
        # Get valid moves more efficiently using precomputed functions
        valid_moves = get_valid_moves(board)
        if not valid_moves:
            break

        # Vectorized board tensor creation
        state = board_to_tensor(board, device)
        logits = model(state).squeeze(0)  # shape: (num_actions,)
        
        # Reuse mask tensor by zeroing it first
        mask.fill_(-float('inf'))
        mask.index_fill_(0, torch.tensor(valid_moves, device=device), 0.0)
        
        # Apply mask and get action
        adjusted_logits = logits + mask
        probs = F.softmax(adjusted_logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()

        # Check if the network preferred invalid moves (but only if needed)
        original_probs = F.softmax(logits, dim=-1)
        if original_probs[action] < 0.01:
            forced_penalty_total += INEFFECTIVE_PENALTY

        # Record action log prob and entropy - avoid in-place operation
        action_tensor = torch.tensor([action], dtype=torch.long, device=device)
        log_probs.append(m.log_prob(action_tensor))
        entropies.append(m.entropy())

        # Apply move directly without copying the board if possible
        new_board, moved, gain = apply_move(board, action)
        
        if not moved:
            # This should rarely happen due to masking
            forced_penalty_total += INEFFECTIVE_PENALTY
            if not valid_moves:
                break
            forced_action = random.choice(valid_moves)
            new_board, moved, forced_gain = apply_move(board, forced_action)
            gain = forced_gain  # Already penalized
        
        board = new_board
        add_random_tile(board)
        total_moves += 1
        merge_score_total += gain

    # Calculate final reward
    episode_reward = compute_reward(merge_score_total, board, forced_penalty_total, total_moves)
    max_tile = max(max(row) for row in board)
    return log_probs, entropies, episode_reward, total_moves, max_tile

# --- Batch Training Loop with Adaptive Learning ---
def train_loop(stdscr, model, optimizer, scheduler, device):
    """
    Batch training loop.
    Simulates multiple episodes per update.
    Implements a warmup phase and an early learning rate multiplier.
    Displays an animated training dashboard.
    """
    # Disable anomaly detection in production as it drastically slows down training
    torch.autograd.set_detect_anomaly(False)
    
    baseline = 0.0
    total_episodes = 0
    best_avg_reward = -float('inf')
    best_model_state = None
    best_max_tile = 0
    rewards_history = []
    moves_history = []
    max_tile_history = []
    training_start_time = time.time()

    stdscr.nodelay(True)  # Non-blocking input
    
    while True:
        try:
            key = stdscr.getch()
            if key in (ord('s'), ord('S')):
                break
        except Exception:
            pass

        # Adaptive learning rate: warmup, then early multiplier
        if total_episodes < WARMUP_EPISODES:
            current_lr = LEARNING_RATE * (total_episodes / WARMUP_EPISODES)
        elif total_episodes < 100:
            current_lr = LEARNING_RATE * EARLY_LR_MULTIPLIER
        else:
            current_lr = LEARNING_RATE
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        batch_log = []
        batch_reward_sum = 0.0
        batch_moves_sum = 0
        batch_max_tile = 0
        batch_start_time = time.time()

        # Process batch episodes in smaller chunks for quicker feedback
        chunk_size = min(8, BATCH_SIZE)  # Process 8 episodes at a time for faster updates
        
        for chunk_start in range(0, BATCH_SIZE, chunk_size):
            chunk_end = min(chunk_start + chunk_size, BATCH_SIZE)
            chunk_episodes = []
            
            # Run chunk_size episodes in sequence (could be parallelized further if needed)
            for _ in range(chunk_end - chunk_start):
                if key in (ord('s'), ord('S')):
                    break
                    
                log_probs, entropies, episode_reward, moves, max_tile = simulate_episode(model, device)
                total_episodes += 1
                rewards_history.append(episode_reward)
                moves_history.append(moves)
                max_tile_history.append(max_tile)
                batch_reward_sum += episode_reward
                batch_moves_sum += moves
                batch_max_tile = max(batch_max_tile, max_tile)
                
                if log_probs:
                    # Compute advantage - smoother updates with baseline
                    advantage = episode_reward - baseline
                    # Include entropy bonus for better exploration
                    entropy_bonus = 2e-3 * torch.stack(entropies).sum()  # Increased entropy weight
                    episode_loss = -torch.stack(log_probs).sum() * advantage - entropy_bonus
                    batch_log.append(episode_loss)
                
                # Check for stop signal more frequently during batch processing
                try:
                    key = stdscr.getch()
                except Exception:
                    pass

        avg_batch_reward = batch_reward_sum / BATCH_SIZE
        avg_batch_moves = batch_moves_sum / BATCH_SIZE

        # Update running baseline using exponential moving average
        if total_episodes <= WARMUP_EPISODES:
            baseline = avg_batch_reward
        else:
            baseline = 0.98 * baseline + 0.02 * avg_batch_reward

        if batch_log:
            batch_loss = torch.stack(batch_log).mean()
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        # Update learning rate scheduler based on recent average reward
        recent_avg_reward = (sum(rewards_history[-100:]) / len(rewards_history[-100:])
                           if rewards_history else avg_batch_reward)
        scheduler.step(recent_avg_reward)

        # Track best performance
        if recent_avg_reward > best_avg_reward:
            best_avg_reward = recent_avg_reward
            best_model_state = model.state_dict()
        if batch_max_tile > best_max_tile:
            best_max_tile = batch_max_tile

        recent_max_tiles = max_tile_history[-100:]
        best_tile_rate = (recent_max_tiles.count(best_max_tile) / len(recent_max_tiles) * 100
                         if recent_max_tiles else 0.0)

        batch_duration = time.time() - batch_start_time
        total_training_time = time.time() - training_start_time

        if avg_batch_reward > recent_avg_reward * 1.05:
            reward_color = curses.color_pair(9)
        elif avg_batch_reward < recent_avg_reward * 0.95:
            reward_color = curses.color_pair(10)
        else:
            reward_color = curses.color_pair(1)

        stdscr.clear()
        stdscr.border()
        offset_y, offset_x = 1, 2
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr(offset_y, offset_x, f"Total Episodes: {total_episodes}")
        stdscr.addstr(offset_y+1, offset_x, "Avg Batch Reward: ")
        stdscr.addstr(f"{avg_batch_reward:.2f}", reward_color)
        stdscr.addstr(offset_y+2, offset_x, f"Recent Avg Reward (last 100): {recent_avg_reward:.2f}")
        stdscr.addstr(offset_y+3, offset_x, f"Best Recent Avg Reward: {best_avg_reward:.2f}")
        stdscr.addstr(offset_y+4, offset_x, f"Avg Episode Length: {avg_batch_moves:.2f}")
        stdscr.addstr(offset_y+5, offset_x, f"Best Tile (this batch): {batch_max_tile}")
        stdscr.addstr(offset_y+6, offset_x, f"Best Tile so far: {best_max_tile} at {best_tile_rate:.1f}% (last 100)")
        stdscr.addstr(offset_y+7, offset_x, f"Batch Duration: {batch_duration:.3f} sec")
        stdscr.addstr(offset_y+8, offset_x, f"Total Training Time: {total_training_time:.1f} sec")
        stdscr.addstr(offset_y+9, offset_x, f"Current LR: {current_lr:.6f}")
        stdscr.addstr(offset_y+11, offset_x, "Training... " + SPINNER[total_episodes % len(SPINNER)])
        stdscr.addstr(offset_y+13, offset_x, "Press 'S' to stop training and save checkpoint.")
        stdscr.refresh()
        time.sleep(DISPLAY_DELAY)

    if best_model_state is not None:
        # Save in new format to avoid pickle loading issues
        torch.save(best_model_state, "2048_model.pt", _use_new_zipfile_serialization=True)
    stdscr.nodelay(False)
    stdscr.clear()
    stdscr.border()
    stdscr.addstr(1, 2, "Training stopped.")
    stdscr.addstr(2, 2, f"Best Recent Avg Reward: {best_avg_reward:.2f}")
    stdscr.addstr(3, 2, f"Best Tile Achieved: {best_max_tile}")
    recent_avg_length = (sum(moves_history[-100:]) / len(moves_history[-100:])
                       if moves_history else 0.0)
    stdscr.addstr(4, 2, f"Avg Episode Length (last 100): {recent_avg_length:.2f}")
    stdscr.addstr(6, 2, "Checkpoint saved as 2048_model.pt")
    stdscr.addstr(8, 2, "Press any key to exit.")
    stdscr.refresh()
    stdscr.getch()

# --- Watch Mode ---
def watch_game(stdscr, model, device):
    """
    Watch mode: Load the best checkpoint and display a live game.
    """
    score = 0.0
    board = new_game()
    delay = 0.5  # Delay between moves in watch mode
    
    while can_move(board):
        stdscr.clear()
        stdscr.border()
        draw_board(stdscr, board, score)
        time.sleep(delay)
        
        valid_moves = get_valid_moves(board)
        if not valid_moves:
            break
            
        state = board_to_tensor(board, device)
        logits = model(state).squeeze(0)
        
        # Create validity mask
        mask = torch.full((4,), -float('inf'), device=device)
        for a in valid_moves:
            mask[a] = 0.0
            
        adjusted_logits = logits + mask
        probs = F.softmax(adjusted_logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()
        
        new_board, moved, gain = apply_move(board, action)
        if not moved:
            valid_moves = get_valid_moves(board)
            if not valid_moves:
                break
            action = random.choice(valid_moves)
            new_board, moved, gain = apply_move(board, action)
            
        board = new_board
        add_random_tile(board)
        score += gain
        
        if not can_move(board):
            break
            
    stdscr.clear()
    stdscr.border()
    draw_board(stdscr, board, score, extra_text="Game Over! Press any key to exit.")
    stdscr.getch()

# --- Main Function ---
def main_console(stdscr):
    """
    Legacy console mode main entry point.
    Initializes curses, loads model, and runs train or watch mode in the console.
    """
    parser = argparse.ArgumentParser(description="2048 Self-Play Learning (Console Mode)")
    parser.add_argument('--mode', choices=['train', 'watch'], default='train',
                        help="--mode train: run training; --mode watch: watch a game using saved model")
    args = parser.parse_args(sys.argv[2:])  # Skip the first argument which is --console
    
    curses.curs_set(0)  # Hide cursor
    curses.start_color()
    init_colors()
    
    # Check for CUDA availability, but use CPU as fallback
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model and send to device
    model = ConvTransformerPolicy(vocab_size=VOCAB_SIZE, d_model=DMODEL, nhead=NHEAD,
                                num_transformer_layers=NUM_TRANSFORMER_LAYERS, 
                                dropout=DROPOUT, num_actions=4).to(device)
    
    if args.mode == 'train':
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        # Learning rate scheduler that reduces LR on plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE, verbose=True
        )
        train_loop(stdscr, model, optimizer, scheduler, device)
    else:
        try:
            # Load checkpoint for watch mode
            checkpoint = torch.load("2048_model.pt", map_location=device)
            model.load_state_dict(checkpoint)
            watch_game(stdscr, model, device)
        except Exception as e:
            # Handle missing checkpoint
            stdscr.clear()
            stdscr.border()
            stdscr.addstr(1, 2, "No checkpoint found. Run training mode first.")
            stdscr.addstr(3, 2, str(e))
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