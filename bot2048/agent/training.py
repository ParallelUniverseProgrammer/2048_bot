#!/usr/bin/env python3
"""
Training logic for 2048 bot agent.
Implements episode simulation and core training loop.
"""

import time
import torch
import torch.nn.functional as F
from collections import deque

from ..utils import config
from ..game import board, reward
from . import model

def simulate_episode(agent, device, total_episodes=0, chunk_size=16):
    """
    Simulate one complete game episode.
    
    Args:
        agent: Agent model
        device: Torch device
        total_episodes: Current training progress for exploration annealing
        chunk_size: Process moves in chunks for memory efficiency
    
    Returns:
        log_probs: Log probabilities of chosen actions
        entropies: Entropy values for chosen actions (for exploration bonus)
        episode_reward: Total episode reward
        moves: Number of moves made
        max_tile: Highest tile achieved
    """
    # Initialize game
    game_board = board.new_game()
    score = 0.0
    log_probs = []
    entropies = []
    moves = 0
    previous_boards = deque(maxlen=20)  # Keep track of recent board states
    forced_move = False
    
    # Temperature for exploration (anneal over time if enabled)
    temperature = config.INITIAL_TEMPERATURE
    if config.USE_TEMPERATURE_ANNEALING and total_episodes > 0:
        temperature = max(
            config.FINAL_TEMPERATURE,
            config.INITIAL_TEMPERATURE * (config.TEMPERATURE_DECAY ** total_episodes)
        )
    
    # Continue while moves are possible
    while board.can_move(game_board):
        valid_moves = board.get_valid_moves(game_board)
        
        # If no valid moves are left, end episode
        if not valid_moves:
            break
            
        # Add current board to history
        previous_boards.append([row[:] for row in game_board])
        
        # Process the board state and get action probabilities
        state = model.board_to_tensor(game_board, device)
        
        # Get model predictions - retain computation graph for training
        logits = agent(state).squeeze(0)
            
        # Create validity mask
        mask = torch.full((4,), -float('inf'), device=device)
        for a in valid_moves:
            mask[a] = 0.0
            
        # Apply mask and temperature for exploration
        adjusted_logits = logits + mask
        if temperature != 1.0:
            adjusted_logits = adjusted_logits / temperature
            
        probs = F.softmax(adjusted_logits, dim=-1)
        
        # Sample action and compute log probability
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        entropy = m.entropy()
        
        log_probs.append(log_prob)
        entropies.append(entropy)
        
        # Apply the chosen action
        action = action.item()
        new_board, moved, merge_score = board.move(game_board, action)
        
        # Handle invalid moves (should be rare due to masking)
        if not moved:
            valid_moves = board.get_valid_moves(game_board)
            if not valid_moves:
                break
            action = valid_moves[0]
            new_board, moved, merge_score = board.move(game_board, action)
            forced_move = True
        
        # Update board state
        game_board = new_board
        board.add_random_tile(game_board)
        
        # Apply penalty for forced moves
        forced_penalty = 1.0 if forced_move else 0.0
        forced_move = False
        
        # Compute reward
        move_reward = reward.compute_reward(
            merge_score, game_board, forced_penalty, moves,
            previous_boards, total_episodes
        )
        
        score += move_reward
        moves += 1
        
    # Return episode results
    max_tile = max(max(row) for row in game_board)
    
    return log_probs, entropies, score, moves, max_tile

def train_loop(agent, optimizer, scheduler, device, callbacks=None, stop_event=None):
    """
    Main training loop.
    
    Args:
        agent: Agent model
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        device: Torch device
        callbacks: Optional dict of callback functions
        stop_event: Optional threading event to signal stopping
    
    Returns:
        Final training stats
    """
    # Disable anomaly detection for speed
    torch.autograd.set_detect_anomaly(False)
    
    # Initialize training state
    training_start_time = time.time()
    baseline = 0.0
    total_episodes = 0
    best_avg_reward = -float('inf')
    best_model_state = None
    best_max_tile = 0
    rewards_history = []
    moves_history = []
    max_tile_history = []
    loss_history = []
    
    # Main training loop
    while stop_event is None or not stop_event.is_set():
        # Set learning rate based on training progress
        if total_episodes < config.WARMUP_EPISODES:
            current_lr = config.LEARNING_RATE * (total_episodes / max(config.WARMUP_EPISODES, 1))
        elif total_episodes < 100:
            current_lr = config.LEARNING_RATE * config.EARLY_LR_MULTIPLIER
        else:
            current_lr = config.LEARNING_RATE
            
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        # Collect batch of episodes
        batch_log_probs = []
        batch_reward_sum = 0.0
        batch_moves_sum = 0
        batch_max_tile = 0
        
        for _ in range(config.BATCH_SIZE):
            if stop_event is not None and stop_event.is_set():
                break
            
            # Check if this episode will be followed by a model save
            checkpoint_saving = (total_episodes + 1) % config.MODEL_SAVE_INTERVAL == 0
            
            # Run episode
            log_probs, entropies, episode_reward, moves, max_tile = simulate_episode(
                agent, device, total_episodes
            )
            
            # Update stats
            total_episodes += 1
            rewards_history.append(episode_reward)
            moves_history.append(moves)
            max_tile_history.append(max_tile)
            batch_reward_sum += episode_reward
            batch_moves_sum += moves
            batch_max_tile = max(batch_max_tile, max_tile)
            
            # Calculate loss for this episode
            if log_probs:
                # Use reward with baseline subtracted
                advantage = episode_reward - baseline
                # Include entropy for exploration
                entropy_bonus = 5e-3 * torch.stack(entropies).sum().detach()
                # Create a detached scalar for advantage
                advantage_t = torch.tensor(advantage, device=device, dtype=torch.float)
                # Compute loss with proper detaching
                episode_loss = -torch.stack(log_probs).sum() * advantage_t - entropy_bonus
                batch_log_probs.append(episode_loss)
                
            # Callback after each episode
            if callbacks and 'on_episode_end' in callbacks:
                current_avg_reward = batch_reward_sum / min(total_episodes, config.BATCH_SIZE)
                recent_avg_reward = (sum(rewards_history[-100:]) / min(len(rewards_history), 100)
                                  if rewards_history else 0.0)
                
                # Prepare current batch loss
                current_batch_loss = 0.0
                if batch_log_probs:
                    current_batch_loss = sum(loss.item() for loss in batch_log_probs) / len(batch_log_probs)
                
                # Call callback
                callbacks['on_episode_end'](
                    total_episodes=total_episodes,
                    current_reward=episode_reward,
                    avg_batch_reward=current_avg_reward,
                    recent_avg_reward=recent_avg_reward,
                    best_avg_reward=best_avg_reward,
                    moves=moves,
                    max_tile=max_tile,
                    best_max_tile=best_max_tile,
                    loss=current_batch_loss,
                    current_lr=current_lr
                )
        
        # If stopping, exit the loop
        if stop_event is not None and stop_event.is_set():
            break
        
        # Process the batch
        if batch_log_probs:
            # Compute the mean loss for the batch
            batch_loss = torch.stack(batch_log_probs).mean()
            loss_history.append(batch_loss.item())
            
            # Perform gradient update
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), config.GRAD_CLIP)
            optimizer.step()
        
        # Update running baseline
        avg_batch_reward = batch_reward_sum / config.BATCH_SIZE
        if total_episodes <= config.WARMUP_EPISODES:
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
            best_model_state = agent.state_dict()
            
            # Save model at new best performance
            if callbacks and 'on_model_save' in callbacks:
                callbacks['on_model_save'](
                    agent=agent,
                    reason="new_best",
                    total_episodes=total_episodes,
                    best_avg_reward=best_avg_reward,
                    best_max_tile=best_max_tile,
                    training_duration=time.time() - training_start_time
                )
        
        if batch_max_tile > best_max_tile:
            best_max_tile = batch_max_tile
            
            # Save model on new best tile
            if callbacks and 'on_model_save' in callbacks:
                callbacks['on_model_save'](
                    agent=agent,
                    reason="new_tile",
                    total_episodes=total_episodes,
                    best_avg_reward=best_avg_reward,
                    best_max_tile=best_max_tile,
                    training_duration=time.time() - training_start_time
                )
        
        # Periodic save based on interval
        if total_episodes % config.MODEL_SAVE_INTERVAL == 0:
            if callbacks and 'on_model_save' in callbacks:
                callbacks['on_model_save'](
                    agent=agent,
                    reason="periodic",
                    total_episodes=total_episodes,
                    best_avg_reward=best_avg_reward,
                    best_max_tile=best_max_tile,
                    training_duration=time.time() - training_start_time
                )
                
        # Batch end callback
        if callbacks and 'on_batch_end' in callbacks:
            loss_value = batch_loss.item() if 'batch_loss' in locals() else 0.0
            callbacks['on_batch_end'](
                total_episodes=total_episodes,
                avg_batch_reward=avg_batch_reward,
                recent_avg_reward=recent_avg_reward,
                best_avg_reward=best_avg_reward,
                avg_batch_moves=batch_moves_sum / config.BATCH_SIZE,
                batch_max_tile=batch_max_tile,
                best_max_tile=best_max_tile,
                loss=loss_value,
                current_lr=current_lr,
                rewards_history=rewards_history,
                moves_history=moves_history,
                max_tile_history=max_tile_history,
                loss_history=loss_history
            )
    
    # Final save at end of training
    if callbacks and 'on_model_save' in callbacks:
        callbacks['on_model_save'](
            agent=agent,
            reason="final",
            total_episodes=total_episodes,
            best_avg_reward=best_avg_reward,
            best_max_tile=best_max_tile,
            training_duration=time.time() - training_start_time
        )
    
    return {
        "total_episodes": total_episodes,
        "best_avg_reward": best_avg_reward,
        "best_max_tile": best_max_tile,
        "training_duration": time.time() - training_start_time,
        "rewards_history": rewards_history,
        "moves_history": moves_history,
        "max_tile_history": max_tile_history,
        "loss_history": loss_history
    }