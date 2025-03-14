#!/usr/bin/env python3
"""
Neural network architecture for the 2048 bot agent.
Implements a Transformer-based policy model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import config
from ..game import board

class ConvTransformerPolicy(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_transformer_layers, num_high_level_layers, dropout, num_actions):
        super().__init__()
        
        # Hyperparameters
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_actions = num_actions
        
        # Embedding layer to convert tile values to vectors
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding for the 4x4 grid
        self.register_buffer("positions", torch.arange(0, 16).long())
        self.pos_embedding = nn.Embedding(16, d_model)
        
        # Transformer encoder for processing the board state
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Apply normalization before attention (better performance)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_transformer_layers)
        
        # High-level processing layers (complex pattern recognition)
        high_level_layers = []
        for i in range(num_high_level_layers):
            high_level_layers.extend([
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),  # GELU activation for better gradient flow
                nn.Dropout(dropout)
            ])
        self.high_level = nn.Sequential(*high_level_layers)
        
        # Policy head (outputs action logits)
        self.policy_head = nn.Linear(d_model, num_actions)
        
        # Initialize weights for better training dynamics
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for better training behavior"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of board state tokens [batch_size, 16]
        
        Returns:
            Action logits [batch_size, num_actions]
        """
        batch_size = x.size(0)
        
        # Convert board state tokens to embeddings
        emb = self.embedding(x)
        
        # Add positional embeddings
        pos_emb = self.pos_embedding(self.positions)
        x = emb + pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Process through transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling over the token dimension
        x = x.mean(dim=1)
        
        # High-level processing for complex pattern recognition
        x = self.high_level(x)
        
        # Policy head
        logits = self.policy_head(x)
        
        return logits

def board_to_tensor(game_board, device):
    """
    Convert board state into a tensor.
    Nonzero tiles are transformed using log2(value) (e.g., 2 -> 1, 4 -> 2),
    while empty cells are represented as 0.
    """
    tokens = board.board_to_tensor_values(game_board)
    return torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

def create_model(device=None):
    """Create a new model instance with current config settings"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = ConvTransformerPolicy(
        vocab_size=config.VOCAB_SIZE,
        d_model=config.DMODEL,
        nhead=config.NHEAD,
        num_transformer_layers=config.NUM_TRANSFORMER_LAYERS,
        num_high_level_layers=config.NUM_HIGH_LEVEL_LAYERS,
        dropout=config.DROPOUT,
        num_actions=4
    ).to(device)
    
    # Use channels_last memory format if on CUDA for better performance
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        
    return model

def load_checkpoint(model, path, device=None):
    """
    Load model checkpoint with more robust error handling.
    
    Args:
        model: The model to load weights into
        path: Path to checkpoint file
        device: Device to load to (defaults to model's device)
        
    Returns:
        Dictionary with metadata if available, or None if not present
    """
    if device is None:
        device = next(model.parameters()).device

    try:
        # Use weights_only=True to address PyTorch security warning
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        
        # Check if the checkpoint has the new format (with metadata)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            # Use non-strict loading to handle architecture changes
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            print(f"Successfully loaded model with metadata from {path}")
            
            # Return metadata if available
            if "metadata" in checkpoint:
                return checkpoint["metadata"]
            
        else:
            # Legacy format - direct state dict
            model.load_state_dict(checkpoint, strict=False)
            print(f"Successfully loaded legacy model format from {path}")
            
        return None
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def save_checkpoint(model, path, metadata=None):
    """
    Save model checkpoint with metadata.
    
    Args:
        model: Model to save
        path: Path to save to
        metadata: Optional dictionary of metadata to include
    """
    try:
        data = {
            "state_dict": model.state_dict()
        }
        
        if metadata:
            data["metadata"] = metadata
            
        torch.save(data, path, _use_new_zipfile_serialization=True)
        print(f"Model checkpoint saved to {path}")
        return True
    
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return False