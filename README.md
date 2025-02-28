# 2048 Self-Play Bot

A reinforcement learning agent that learns to play the 2048 game through self-play using a hybrid CNN-Transformer architecture and the REINFORCE algorithm.

## Features

- **Efficient Neural Architecture**: Hybrid CNN-Transformer model that processes board states using both local pattern recognition and global attention
- **Sophisticated Reward Function**: Multi-component reward system that considers merge scores, empty cells, high tiles, edge placement, and potential future merges
- **Smart Move Selection**: Validity masking system that prevents the model from selecting invalid moves
- **Batch Training**: More stable learning through batch updates
- **Adaptive Learning Rate**: Warmup phase followed by scheduled learning rate adjustments
- **Web Interface**: Modern dashboard with real-time training graphs and game visualization
- **Hardware Monitoring**: Real-time CPU, RAM, and GPU usage tracking
- **Legacy Console Mode**: Original console-based visualization is still available

## Requirements

- Python 3.6+
- PyTorch
- Flask and Flask-SocketIO (for web interface)
- psutil (for hardware monitoring)
- GPUtil (optional, for GPU monitoring)
- curses (included with Python on Unix/Mac, Windows users may need `windows-curses` for legacy mode)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/2048_bot.git
cd 2048_bot

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Interface (Recommended)

Run the bot with the modern web interface:

```bash
python 2048_bot.py
```

This will:
1. Start a web server
2. Display console hardware information
3. Automatically open your web browser to the dashboard
4. Show real-time training/gameplay visualization

Options:
- `--port 8080`: Use a custom port (default: 5000)
- `--no-browser`: Don't automatically open a browser
- `--debug`: Run in debug mode

### Web Dashboard Features

The web dashboard provides:
- **Hardware Monitoring**: Real-time CPU, RAM, and GPU usage information
- **Training Visualization**: Live charts for rewards, loss, and maximum tile achieved
- **Game Visualization**: Watch the agent play with a colorful game board
- **Interactive Controls**: Start/stop training or watching gameplay with a click

### Legacy Console Mode

For those who prefer the original console interface:

```bash
python 2048_bot.py --console --mode train  # Training mode
python 2048_bot.py --console --mode watch  # Watch mode
```

In console training mode, press 'S' at any time to stop training and save the best model checkpoint.

## How It Works

1. **Board Representation**: The 4x4 game board is converted to a tensor where each cell value is represented using log2 (e.g., 2→1, 4→2, etc.)

2. **Neural Network**: A hybrid architecture processes the board:
   - Embedding layer → CNN with batch normalization → Transformer encoder → Action logits

3. **Decision Making**: For each move, the network outputs action probabilities for up/down/left/right, masked to allow only valid moves

4. **Reward Function**: Combines multiple components to guide learning:
   - Merge score (when tiles combine)
   - Empty cell bonus
   - High tile bonus
   - Edge placement strategy
   - Potential merge detection
   - Time-dependent factor that rewards later, more impactful moves

5. **Training**: REINFORCE algorithm with advantage estimation and entropy regularization for exploration

## Web Architecture

The web interface uses:
- **Flask**: Backend web server
- **Socket.IO**: Real-time communication between server and browser
- **Chart.js**: Beautiful, responsive charts
- **Modern CSS**: Responsive design that works on desktop and mobile

## Customization

The script includes numerous configuration parameters at the top that can be adjusted:

```python
# Learning parameters
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
# ... and many more
```

Experiment with these values to find the optimal configuration for your hardware and preferences.

## License

MIT