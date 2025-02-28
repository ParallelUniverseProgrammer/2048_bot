# 2048 Bot Project Guidelines

## Commands
- Run training: `python 2048_bot.py --mode train`
- Watch trained agent: `python 2048_bot.py --mode watch`
- Run web interface: `python 2048_bot.py` (default)
- Run web interface with custom port: `python 2048_bot.py --port 8080`
- Run in legacy console mode: `python 2048_bot.py --console`
- Install dependencies: `pip install -r requirements.txt`
- Web server only: `python 2048_bot_server.py`

## Code Style Guidelines
- **Imports**: Group standard library, third-party, and local imports separated by blank lines
- **Formatting**: Use 4-space indentation; maximum line length of 100 characters
- **Naming**: snake_case for variables/functions, PascalCase for classes, UPPERCASE for constants
- **Documentation**: Docstrings for all functions, classes, and modules
- **Type Hints**: Not used in the codebase (stick to this convention)
- **Error Handling**: Use try/except blocks for expected exceptions
- **Constants**: Define constants at the top of files in UPPERCASE
- **Threading**: Use daemon threads and proper stop events for graceful shutdown
- **Optimizations**: Use torch-specific optimizations (e.g., channels_last memory format for CUDA)
- **Architecture**: Hybrid CNN-Transformer model for policy network
- **Algorithm**: REINFORCE with batch training and advantage estimation