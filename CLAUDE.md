# 2048 Bot Project Guidelines

## Commands
- Run training: `python 2048_bot.py --mode train`
- Watch trained agent: `python 2048_bot.py --mode watch`
- Run web interface: `python 2048_bot.py` (default)
- Run web interface with custom port: `python 2048_bot.py --port 8080`
- Run in legacy console mode: `python 2048_bot.py --console`
- Web server only: `python 2048_bot_server.py`
- Debug specific component: `python 2048_bot.py --debug [component_name]`
- Install dependencies: `pip install -r requirements.txt`

## Development Commands
- Run unit tests: `python -m unittest discover`
- Run specific test: `python -m unittest tests/test_file.py::TestClass::test_method`
- Check code quality: `flake8 --max-line-length=100 *.py`
- Export model: `python 2048_bot.py --mode export --output model.pt`

## Project Architecture
- Pure Transformer model for board state analysis
- REINFORCE algorithm with batch training
- Reward function with multiple components (high tiles, novelty, adjacency bonus)
- Web interface using Flask and SocketIO
- Monitoring tools for hardware performance tracking

## Dependencies
- Core: torch>=1.9.0, flask>=2.0.0, flask-socketio>=5.1.0, numpy>=1.19.0
- Monitoring: gputil>=1.4.0, psutil>=5.8.0, matplotlib>=3.4.0, pandas>=1.3.0

## Code Style Guidelines
- **Imports**: Group standard library, third-party, local imports with blank lines between groups
- **Formatting**: 4-space indentation; 100 character line length
- **Naming**: snake_case for variables/functions, PascalCase for classes, UPPERCASE for constants
- **Error Handling**: Use try/except blocks for expected exceptions; log errors appropriately
- **Constants**: Define constants at the top of files in UPPERCASE
- **Threading**: Use daemon threads with stop events for graceful shutdown