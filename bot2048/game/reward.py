#!/usr/bin/env python3
"""
Reward computation for 2048 bot.
Implements sophisticated reward functions with multiple components.
"""

import math
from . import board

# Import configuration
from ..utils import config

def compute_novelty(game_board):
    """
    Evaluate how "interestingly" the tiles are arranged.
    Rewards patterns that differ from the typical corner-stacking strategy.
    Enhanced with pattern diversity analysis.
    """
    # Count how many quadrants of the board have significant tiles
    quadrant_values = [
        sum(game_board[i][j] for i in range(2) for j in range(2)),  # Top-left
        sum(game_board[i][j] for i in range(2) for j in range(2, 4)),  # Top-right
        sum(game_board[i][j] for i in range(2, 4) for j in range(2)),  # Bottom-left
        sum(game_board[i][j] for i in range(2, 4) for j in range(2, 4))  # Bottom-right
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
    tile_values = [game_board[i][j] for i in range(board.GRID_SIZE) for j in range(board.GRID_SIZE) 
                 if game_board[i][j] > 0]
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
    for i in range(board.GRID_SIZE):
        for j in range(board.GRID_SIZE - 1):
            if game_board[i][j] > 0 and game_board[i][j+1] > 0:
                if game_board[i][j] == game_board[i][j+1] * 2 or game_board[i][j] * 2 == game_board[i][j+1]:
                    sequential_count += 1
    
    # Vertical snake patterns
    for j in range(board.GRID_SIZE):
        for i in range(board.GRID_SIZE - 1):
            if game_board[i][j] > 0 and game_board[i+1][j] > 0:
                if game_board[i][j] == game_board[i+1][j] * 2 or game_board[i][j] * 2 == game_board[i+1][j]:
                    sequential_count += 1
    
    pattern_score += sequential_count * 0.1
    
    # 3. Reward balanced distributions across the board
    occupied_cells = sum(1 for i in range(board.GRID_SIZE) for j in range(board.GRID_SIZE) 
                        if game_board[i][j] > 0)
    balance_score = occupied_cells / (board.GRID_SIZE * board.GRID_SIZE) * 0.5
    pattern_score += balance_score
    
    # Combine all novelty components
    novelty_score = active_quadrants * 0.5 + variance_norm + pattern_score * 0.3
    
    return novelty_score

def identify_board_strategy(game_board):
    """
    Identify which strategy the current board configuration is using.
    Returns a strategy name and confidence score.
    """
    # Extract board features for strategy identification
    max_tile = max(max(row) for row in game_board)
    max_tile_pos = None
    for i in range(board.GRID_SIZE):
        for j in range(board.GRID_SIZE):
            if game_board[i][j] == max_tile:
                max_tile_pos = (i, j)
                break
        if max_tile_pos:
            break
    
    # Strategy 1: Corner strategy
    corner_positions = [(0, 0), (0, board.GRID_SIZE-1), 
                       (board.GRID_SIZE-1, 0), (board.GRID_SIZE-1, board.GRID_SIZE-1)]
    corner_confidence = 0.0
    if max_tile_pos in corner_positions:
        corner_confidence = 0.8
        
        # Check if high values are along the edges
        edge_values = []
        # Top edge
        edge_values.extend([game_board[0][j] for j in range(board.GRID_SIZE)])
        # Bottom edge
        edge_values.extend([game_board[board.GRID_SIZE-1][j] for j in range(board.GRID_SIZE)])
        # Left edge (excluding corners already counted)
        edge_values.extend([game_board[i][0] for i in range(1, board.GRID_SIZE-1)])
        # Right edge (excluding corners already counted)
        edge_values.extend([game_board[i][board.GRID_SIZE-1] for i in range(1, board.GRID_SIZE-1)])
        
        # Calculate what percentage of total value is on edges
        total_value = sum(sum(row) for row in game_board)
        edge_value = sum(edge_values)
        if total_value > 0:
            edge_ratio = edge_value / total_value
            if edge_ratio > 0.7:
                corner_confidence += 0.2
    
    # Strategy 2: Snake strategy (zigzag pattern of decreasing values)
    snake_confidence = 0.0
    snake_patterns = [
        # Rows alternating left-right, right-left
        [(i, j if i % 2 == 0 else board.GRID_SIZE - 1 - j) 
         for i in range(board.GRID_SIZE) 
         for j in range(board.GRID_SIZE)],
        # Columns alternating top-bottom, bottom-top
        [(i if j % 2 == 0 else board.GRID_SIZE - 1 - i, j) 
         for j in range(board.GRID_SIZE) 
         for i in range(board.GRID_SIZE)]
    ]
    
    for pattern in snake_patterns:
        # Count how many tiles follow a decreasing sequence
        decreasing_count = 0
        last_val = None
        for i, j in pattern:
            if game_board[i][j] > 0:
                if last_val is None or game_board[i][j] <= last_val:
                    decreasing_count += 1
                last_val = game_board[i][j]
        
        # Calculate snake confidence based on how well the pattern matches
        pattern_confidence = decreasing_count / (board.GRID_SIZE * board.GRID_SIZE) * 2 - 0.5
        snake_confidence = max(snake_confidence, pattern_confidence)
    
    # Strategy 3: Balanced strategy (values distributed across board)
    balanced_confidence = 0.0
    filled_cells = sum(1 for i in range(board.GRID_SIZE) 
                      for j in range(board.GRID_SIZE) 
                      if game_board[i][j] > 0)
    balanced_confidence = filled_cells / (board.GRID_SIZE * board.GRID_SIZE) * 0.5
    
    # Calculate variance of non-zero cells (low variance = more balanced)
    non_zero_vals = [game_board[i][j] for i in range(board.GRID_SIZE) 
                   for j in range(board.GRID_SIZE) 
                   if game_board[i][j] > 0]
    if len(non_zero_vals) > 1:
        mean_val = sum(non_zero_vals) / len(non_zero_vals)
        variance = sum((v - mean_val) ** 2 for v in non_zero_vals) / len(non_zero_vals)
        # Normalize variance
        normalized_variance = min(1.0, math.log(1 + variance) / 15)
        balanced_confidence += (1 - normalized_variance) * 0.5
    
    # Return the dominant strategy
    if corner_confidence >= max(snake_confidence, balanced_confidence):
        return "corner", corner_confidence
    elif snake_confidence >= balanced_confidence:
        return "snake", snake_confidence
    else:
        return "balanced", balanced_confidence

def compute_strategy_diversity_bonus(game_board, previous_boards):
    """
    Compute a bonus for having diverse strategies over time.
    Rewards the agent for successfully using different strategies.
    """
    if not previous_boards:
        return 0.0
    
    # Identify current strategy
    current_strategy, current_confidence = identify_board_strategy(game_board)
    
    # Check if we have enough boards to compute a meaningful diversity
    if len(previous_boards) < 3:
        return 0.0
    
    # Look at the last few boards to see what strategies were used
    recent_boards = list(previous_boards)[-3:]
    recent_strategies = [identify_board_strategy(b)[0] for b in recent_boards]
    
    # If current strategy is different from the majority of recent strategies
    # and we're confident about it, give a strategy shift bonus
    if current_strategy not in recent_strategies and current_confidence > 0.6:
        return config.STRATEGY_SHIFT_BONUS
    
    # If we've been consistently using diverse strategies, give a smaller bonus
    unique_strategies = len(set(recent_strategies))
    if unique_strategies >= 2 and current_confidence > 0.5:
        return config.STRATEGY_SHIFT_BONUS * 0.5
    
    return 0.0

def compute_reward(merge_score, game_board, forced_penalty, move_count, previous_boards=None, total_episodes=0):
    """
    Compute reward based on multiple components.
    
    Args:
        merge_score: Score from merging tiles
        game_board: Current board state
        forced_penalty: Penalty for forced moves or ineffective actions
        move_count: Number of moves made in the current game
        previous_boards: List of previous board states (for novelty computation)
        total_episodes: Total training episodes completed
    
    Returns:
        Total reward value
    """
    # Base reward components
    max_tile = max(max(row) for row in game_board)
    empty_count = sum(1 for i in range(board.GRID_SIZE) 
                     for j in range(board.GRID_SIZE) 
                     if game_board[i][j] == 0)
    
    # 1. Merge score component (primary reward signal)
    merge_reward = merge_score * config.REWARD_SCALING
    
    # 2. High tile bonus (reward achieving higher tiles)
    high_tile_bonus = 0
    if max_tile >= config.HIGH_TILE_THRESHOLD:
        high_tile_bonus = config.HIGH_TILE_BONUS * math.log2(max_tile / config.HIGH_TILE_THRESHOLD)
    
    # 3. Empty cell bonus (maintain open spaces)
    emptiness_factor = empty_count / (board.GRID_SIZE * board.GRID_SIZE)
    empty_cell_bonus = emptiness_factor * 2.0
    
    # 4. Potential merges bonus (reward potential future merges)
    merge_potential = board.count_potential_merges(game_board)
    merge_potential_bonus = merge_potential * 0.2
    
    # 5. Novelty bonus (reward interesting board configurations)
    novelty_bonus = 0
    if config.NOVELTY_BONUS > 0:
        novelty_score = compute_novelty(game_board) 
        novelty_bonus = novelty_score * config.NOVELTY_BONUS
    
    # 6. Strategy diversity bonus (reward trying different approaches)
    strategy_bonus = 0
    if previous_boards and config.PATTERN_DIVERSITY_BONUS > 0:
        strategy_bonus = compute_strategy_diversity_bonus(game_board, previous_boards) * config.PATTERN_DIVERSITY_BONUS
    
    # 7. Time-based exploration factor (encourage more exploration early in training)
    time_factor = 1.0
    if total_episodes > 0:
        phase = min(1.0, total_episodes / config.TIME_FACTOR_CONSTANT)
        time_factor = 1.0 + (1.0 - phase) * 0.5  # More exploration early (1.5x), tapering to 1.0
    
    # Combine all reward components
    total_reward = (
        merge_reward +
        high_tile_bonus + 
        empty_cell_bonus + 
        merge_potential_bonus +
        novelty_bonus * time_factor +
        strategy_bonus * time_factor
    )
    
    # Apply forced move penalty
    total_reward = max(0.01, total_reward - forced_penalty * config.INEFFECTIVE_PENALTY)
    
    return total_reward