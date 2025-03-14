#!/usr/bin/env python3
"""
Console interface for 2048 bot.
Provides a terminal-based UI for watching and training the bot.
"""

import time
import os
import curses
import torch

from ..utils import config
from ..game import board
from ..agent import model, training

DISPLAY_DELAY = 0.001  # Faster refresh rate for training display
SPINNER = "|/-\\"      # Animation spinner

def display_board(stdscr, game_board, score, moves, max_tile):
    """Display the game board in the console."""
    # Clear screen
    stdscr.clear()
    
    # Print game info
    stdscr.addstr(0, 0, f"Score: {score:.2f}")
    stdscr.addstr(1, 0, f"Moves: {moves}")
    stdscr.addstr(2, 0, f"Max Tile: {max_tile}")
    stdscr.addstr(3, 0, "")
    
    # Calculate column width based on max tile
    col_width = 6
    if max_tile >= 1000:
        col_width = 7
    
    # Print board
    for i, row in enumerate(game_board):
        for j, cell in enumerate(row):
            if cell == 0:
                stdscr.addstr(i + 4, j * col_width, "." + " " * (col_width - 1))
            else:
                stdscr.addstr(i + 4, j * col_width, str(cell) + " " * (col_width - len(str(cell))))
    
    # Print instructions
    stdscr.addstr(9, 0, "Press 'q' to quit")
    
    # Refresh screen
    stdscr.refresh()

def run_training_display(stdscr):
    """Run the training process with console display."""
    # Initialize curses
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()
    
    # Print initial message
    stdscr.addstr(0, 0, "Initializing 2048 bot training...")
    stdscr.refresh()
    
    # Initialize device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = model.create_model(device)
    
    # Try to load existing model
    try:
        if os.path.exists("2048_model.pt"):
            stdscr.addstr(1, 0, "Loading existing model...")
            stdscr.refresh()
            model.load_checkpoint(agent, "2048_model.pt", device)
            stdscr.addstr(2, 0, "Model loaded successfully")
            stdscr.refresh()
    except Exception as e:
        stdscr.addstr(2, 0, f"Error loading model: {str(e)}")
        stdscr.refresh()
        time.sleep(2)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=2e-6,
        betas=(0.9, 0.99),
        eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE, verbose=True
    )
    
    # Training state
    training_start = time.time()
    episodes = 0
    best_avg_reward = -float('inf')
    best_max_tile = 0
    
    # Setup UI areas
    header_lines = 4
    stats_start = header_lines + 5
    
    def on_episode_end(total_episodes, current_reward, avg_batch_reward, recent_avg_reward,
                      best_avg_reward, moves, max_tile, best_max_tile, loss, current_lr):
        nonlocal episodes
        episodes = total_episodes
        
        # Display progress spinner
        spinner_char = SPINNER[episodes % len(SPINNER)]
        stdscr.addstr(header_lines, 0, f"{spinner_char} Episode: {episodes}, Reward: {current_reward:.2f}, Max Tile: {max_tile}")
        
        # Check for user input
        key = stdscr.getch()
        if key == ord('q'):
            return False  # Signal to stop
        
        return True  # Continue
    
    def on_batch_end(total_episodes, avg_batch_reward, recent_avg_reward, best_avg_reward,
                    avg_batch_moves, batch_max_tile, best_max_tile, loss, current_lr,
                    rewards_history, moves_history, max_tile_history, loss_history):
        # Update stats display
        stdscr.addstr(stats_start, 0, f"Total Episodes: {total_episodes}")
        stdscr.addstr(stats_start + 1, 0, f"Avg Batch Reward: {avg_batch_reward:.2f}")
        stdscr.addstr(stats_start + 2, 0, f"Recent Avg Reward: {recent_avg_reward:.2f}")
        stdscr.addstr(stats_start + 3, 0, f"Best Avg Reward: {best_avg_reward:.2f}")
        stdscr.addstr(stats_start + 4, 0, f"Avg Batch Moves: {avg_batch_moves:.2f}")
        stdscr.addstr(stats_start + 5, 0, f"Batch Max Tile: {batch_max_tile}")
        stdscr.addstr(stats_start + 6, 0, f"Best Max Tile: {best_max_tile}")
        stdscr.addstr(stats_start + 7, 0, f"Loss: {loss:.4f}")
        stdscr.addstr(stats_start + 8, 0, f"Learning Rate: {current_lr:.6f}")
        
        training_time = time.time() - training_start
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        stdscr.addstr(stats_start + 9, 0, f"Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        stdscr.addstr(stats_start + 11, 0, "Press 'q' to quit")
        stdscr.refresh()
        
        # Check for user input
        key = stdscr.getch()
        if key == ord('q'):
            return False  # Signal to stop
        
        return True  # Continue
    
    def on_model_save(agent, reason, total_episodes, best_avg_reward, best_max_tile, training_duration):
        # Display save status
        if reason == "new_best":
            stdscr.addstr(stats_start + 10, 0, "Saved new best model!       ")
        elif reason == "new_tile":
            stdscr.addstr(stats_start + 10, 0, f"Saved model with new best tile: {best_max_tile}")
        elif reason == "periodic":
            stdscr.addstr(stats_start + 10, 0, f"Saved periodic checkpoint        ")
        elif reason == "final":
            stdscr.addstr(stats_start + 10, 0, "Saved final model                ")
        
        stdscr.refresh()
        
        # Create metadata
        metadata = {
            "timestamp": time.time(),
            "total_episodes": total_episodes,
            "best_avg_reward": float(best_avg_reward),
            "best_max_tile": int(best_max_tile),
            "training_duration": training_duration,
            "save_reason": reason
        }
        
        # Save model with metadata
        model.save_checkpoint(agent, "2048_model.pt", metadata)
    
    # Setup callbacks
    callbacks = {
        'on_episode_end': on_episode_end,
        'on_batch_end': on_batch_end,
        'on_model_save': on_model_save
    }
    
    # Run training
    try:
        # Clear screen and show initial UI
        stdscr.clear()
        stdscr.addstr(0, 0, "2048 Bot Training")
        stdscr.addstr(1, 0, f"Device: {device}")
        stdscr.addstr(2, 0, f"Batch Size: {config.BATCH_SIZE}")
        stdscr.addstr(3, 0, "Press 'q' to quit")
        stdscr.refresh()
        
        # Run the training loop
        training.train_loop(agent, optimizer, scheduler, device, callbacks)
    except KeyboardInterrupt:
        stdscr.addstr(stats_start + 10, 0, "Training interrupted by user")
        stdscr.refresh()
        time.sleep(2)
    except Exception as e:
        stdscr.addstr(stats_start + 10, 0, f"Error: {str(e)}")
        stdscr.refresh()
        time.sleep(2)

def run_watch_mode(stdscr):
    """Run watch mode with console display."""
    # Initialize curses
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()
    
    # Print initial message
    stdscr.addstr(0, 0, "Loading 2048 bot model...")
    stdscr.refresh()
    
    # Initialize device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = model.create_model(device)
    
    # Try to load existing model
    try:
        if not os.path.exists("2048_model.pt"):
            stdscr.addstr(1, 0, "No model found. Please train a model first.")
            stdscr.addstr(2, 0, "Press any key to exit")
            stdscr.refresh()
            stdscr.getch()
            return
        
        stdscr.addstr(1, 0, "Loading model...")
        stdscr.refresh()
        model.load_checkpoint(agent, "2048_model.pt", device)
        stdscr.addstr(2, 0, "Model loaded successfully")
        stdscr.refresh()
        time.sleep(1)
    except Exception as e:
        stdscr.addstr(2, 0, f"Error loading model: {str(e)}")
        stdscr.addstr(3, 0, "Press any key to exit")
        stdscr.refresh()
        stdscr.getch()
        return
    
    # Game loop
    score = 0.0
    game_board = board.new_game()
    moves = 0
    delay = 0.2  # Slightly slower for better visibility
    
    # Display initial board
    display_board(stdscr, game_board, score, moves, max(max(row) for row in game_board))
    
    # Main game loop
    while board.can_move(game_board):
        # Check for keyboard input
        key = stdscr.getch()
        if key == ord('q'):
            break
        
        # Get valid moves
        valid_moves = board.get_valid_moves(game_board)
        if not valid_moves:
            break
        
        # Get model prediction
        state = model.board_to_tensor(game_board, device)
        with torch.no_grad():
            logits = agent(state).squeeze(0)
        
        # Create validity mask
        mask = torch.full((4,), -float('inf'), device=device)
        for a in valid_moves:
            mask[a] = 0.0
        
        # Sample action
        adjusted_logits = logits + mask
        probs = torch.nn.functional.softmax(adjusted_logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()
        
        # Apply action
        new_board, moved, gain = board.move(game_board, action)
        if not moved:
            # If the move was invalid (shouldn't happen with masking), try a valid move
            if valid_moves:
                action = valid_moves[0]
                new_board, moved, gain = board.move(game_board, action)
        
        # Update board
        game_board = new_board
        board.add_random_tile(game_board)
        
        # Update stats
        score += gain
        moves += 1
        
        # Update display
        display_board(stdscr, game_board, score, moves, max(max(row) for row in game_board))
        
        # Sleep for visualization
        time.sleep(delay)
    
    # Game over
    stdscr.addstr(9, 0, "Game Over! Press any key to exit.")
    stdscr.refresh()
    stdscr.getch()

def run_console_ui(mode="watch"):
    """Run the console UI."""
    if mode == "train":
        curses.wrapper(run_training_display)
    else:
        curses.wrapper(run_watch_mode)