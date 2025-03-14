#!/usr/bin/env python3
"""
Game board logic for 2048 bot.
Defines all board operations and game mechanics.
"""

import random
import math
import copy

# Game dimensions
GRID_SIZE = 4

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

def move_standard(board):
    """Base movement (left)"""
    board, changed1 = compress(board)
    board, changed2, score = merge(board)
    board, _ = compress(board)
    return board, (changed1 or changed2), score

def move_reverse_standard(board):
    """Reverse movement (right)"""
    board = reverse(board)
    board, changed, score = move_standard(board)
    return reverse(board), changed, score

def move(board, direction):
    """
    Apply a move to the board in the specified direction.
    direction: 0=up, 1=down, 2=left, 3=right
    Returns: new board, whether the board changed, and gained score.
    """
    if direction == 0:  # up
        board = transpose(board)
        result, changed, score = move_standard(board)
        return transpose(result), changed, score
    elif direction == 1:  # down
        board = transpose(board)
        result, changed, score = move_reverse_standard(board)
        return transpose(result), changed, score
    elif direction == 2:  # left
        return move_standard(board)
    elif direction == 3:  # right
        return move_reverse_standard(board)
    else:
        return board, False, 0

def get_valid_moves(board):
    """Return a list of move indices that are valid for the current board."""
    valid = []
    for a in range(4):
        temp_board = copy.deepcopy(board)
        new_board, moved, _ = move(temp_board, a)
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

def board_to_tensor_values(board):
    """
    Convert board state into tensor values.
    Nonzero tiles are transformed using log2(value) (e.g., 2 -> 1, 4 -> 2),
    while empty cells are represented as 0.
    """
    tokens = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            tokens.append(0 if board[i][j] == 0 else int(math.log2(board[i][j])))
    return tokens

def analyze_board(board):
    """
    Analyze the current board state for various metrics.
    Returns a dictionary with key statistics.
    """
    max_tile = max(max(row) for row in board)
    empty_cells = sum(1 for i in range(GRID_SIZE) for j in range(GRID_SIZE) if board[i][j] == 0)
    total_value = sum(sum(row) for row in board)
    
    # Calculate monotonicity (penalties for non-monotonic rows/cols)
    monotonicity_penalty = 0
    
    # Check rows
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE - 1):
            if board[i][j] > 0 and board[i][j+1] > 0 and board[i][j] < board[i][j+1]:
                monotonicity_penalty += 1
    
    # Check columns
    for j in range(GRID_SIZE):
        for i in range(GRID_SIZE - 1):
            if board[i][j] > 0 and board[i+1][j] > 0 and board[i][j] < board[i+1][j]:
                monotonicity_penalty += 1
    
    return {
        "max_tile": max_tile,
        "empty_cells": empty_cells,
        "total_value": total_value,
        "monotonicity_penalty": monotonicity_penalty,
        "potential_merges": count_potential_merges(board)
    }