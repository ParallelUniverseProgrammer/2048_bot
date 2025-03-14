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
    python main.py              # Web UI interface (recommended)
    python main.py --console    # Legacy console mode
    python main.py --port 8080  # Use custom port for Web UI
"""

import sys
import argparse
sys.path.append('.')  # Add current directory to path

from bot2048.ui.web import run_server
from bot2048.ui.console import run_console_ui
from bot2048.utils import config

def main():
    parser = argparse.ArgumentParser(description="2048 Bot - Reinforcement Learning Agent")
    parser.add_argument('--mode', choices=['train', 'watch'], default='watch',
                      help="Mode to run in when using console mode")
    parser.add_argument('--console', action='store_true',
                      help="Run in console mode instead of web interface")
    parser.add_argument('--port', type=int, default=5000,
                      help="Port to run the web server on")
    parser.add_argument('--debug', action='store_true',
                      help="Run in debug mode (web interface only)")
    parser.add_argument('--no-browser', action='store_true',
                      help="Don't open browser automatically (web interface only)")
    args = parser.parse_args()
    
    # Initialize configuration
    config.scale_model_size()
    config.optimize_torch_settings()
    
    if args.console:
        # Run in console mode
        run_console_ui(args.mode)
    else:
        # Run web interface
        run_server(args.port, args.debug, not args.no_browser)

if __name__ == "__main__":
    main()