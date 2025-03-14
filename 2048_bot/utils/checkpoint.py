#!/usr/bin/env python3
"""
Checkpoint management for 2048 bot.
Handles loading, saving, and organizing model checkpoints.
"""

import os
import time
import shutil
import torch

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

def save_model(model, path, metadata=None, archive=True):
    """
    Save model checkpoint with metadata.
    
    Args:
        model: Model to save
        path: Path to save to
        metadata: Optional dictionary of metadata to include
        archive: Whether to archive the previous checkpoint
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # If we're archiving and the current model exists, create an archive copy
        if archive and os.path.exists(path):
            try:
                # Make checkpoints directory if it doesn't exist
                os.makedirs("checkpoints", exist_ok=True)
                
                # Create a timestamp-based filename for archive
                timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
                archive_filename = f"2048_model_{timestamp_str}.pt"
                
                # Copy current model to the archive with timestamp name
                shutil.copy2(path, os.path.join("checkpoints", archive_filename))
                print(f"✅ Archived previous checkpoint to checkpoints/{archive_filename}")
            except Exception as e:
                print(f"⚠️ Could not archive previous checkpoint: {e}")
        
        # Prepare data to save
        data = {
            "state_dict": model.state_dict()
        }
        
        if metadata:
            data["metadata"] = metadata
        
        # Save the model
        torch.save(data, path, _use_new_zipfile_serialization=True)
        print(f"✅ Model checkpoint saved to {path}")
        return True
    
    except Exception as e:
        print(f"❌ Error saving checkpoint: {e}")
        return False

def load_model_checkpoint(model, path, device=None):
    """
    Load a model checkpoint with robust error handling.
    
    Args:
        model: Model to load weights into
        path: Path to checkpoint file
        device: Device to load checkpoint to (if None, uses model's device)
        
    Returns:
        Dictionary with metadata if available, None otherwise
    """
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}")
        return None
    
    if device is None:
        device = next(model.parameters()).device
    
    try:
        print(f"Loading checkpoint from {path}")
        
        # Use weights_only=True to address PyTorch security warning
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        
        # Check if the checkpoint has the new format (with metadata)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            # Use non-strict loading to handle architecture changes
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            print("Successfully loaded model with metadata")
            
            # Return metadata if available
            if "metadata" in checkpoint:
                return checkpoint["metadata"]
        else:
            # Legacy format - direct state dict
            model.load_state_dict(checkpoint, strict=False)
            print("Successfully loaded legacy model format")
        
        return None
    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def list_checkpoints(include_current=True):
    """
    List all available checkpoints.
    
    Args:
        include_current: Whether to include the current model
        
    Returns:
        List of dictionaries with checkpoint info
    """
    checkpoints = []
    
    # Check if current model exists
    if include_current and os.path.exists("2048_model.pt"):
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
    
    return checkpoints

def get_checkpoint_info(path):
    """
    Get basic information about a checkpoint without loading the model.
    
    Args:
        path: Path to the checkpoint
        
    Returns:
        Dictionary with checkpoint info
    """
    if not os.path.exists(path):
        return {"exists": False, "message": f"Checkpoint {path} not found"}
    
    # Get checkpoint file stats
    stat_info = os.stat(path)
    
    # Creation time
    created_time = stat_info.st_mtime
    created_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_time))
    
    # Age calculation
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
    
    return {
        "exists": True,
        "created": created_str,
        "age": age_str,
        "size": size_str,
        "filename": os.path.basename(path),
        "path": path
    }