# 2048 Bot Project Guidelines

## Commands
- Run training: `python 2048_bot.py --mode train`
- Watch trained agent: `python 2048_bot.py --mode watch`
- Run web interface: `python 2048_bot.py` (default)
- Run web interface with custom port: `python 2048_bot.py --port 8080`
- Run in legacy console mode: `python 2048_bot.py --console`
- Web server only: `python 2048_bot_server.py`
- Monitor and train with hardware stats: `python 2048_bot.py --mode train --monitor`
- Install dependencies: `pip install -r requirements.txt`

## Dependencies
Core dependencies: torch>=1.9.0, flask>=2.0.0, flask-socketio>=5.1.0, numpy>=1.19.0
Monitoring tools: gputil>=1.4.0, psutil>=5.8.0, matplotlib>=3.4.0, pandas>=1.3.0

## Code Style Guidelines
- **Imports**: Group standard library, third-party, local imports with blank lines between groups
- **Formatting**: 4-space indentation; 100 char line length; blank line between logical sections
- **Naming**: snake_case for variables/functions, PascalCase for classes, UPPERCASE for constants
- **Documentation**: Docstrings for all functions, classes, and modules
- **Type Hints**: Not used in the codebase (stick to this convention)
- **Error Handling**: Use try/except blocks for expected exceptions
- **Constants**: Define constants at the top of files in UPPERCASE
- **Threading**: Use daemon threads with stop events for graceful shutdown
- **Architecture**: Hybrid CNN-Transformer model using REINFORCE algorithm with batch training