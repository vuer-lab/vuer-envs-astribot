# Vuer Envs Astribot

Autonomous teleoperation bridge for Astribot robot with Vision Pro integration.

## Installation

### Prerequisites
- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager
- Astribot SDK

### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Dependencies
```bash
# Install project dependencies
uv sync

# Install with dev dependencies
uv sync --dev
```

## Quick Start

```python
from vuer_envs_astribot import PolicyClient

# Initialize client (connects to robot automatically)
client = PolicyClient(
    robot_ip='192.168.200.111',
    trajectory_mode='both',
    freq=250.0
)

# Wait for robot to be ready
while not client.is_ready():
    time.sleep(0.1)

# Execute action
client.execute_action({
    'left_position': [0.3, 0.2, 0.5],
    'left_orientation': [0, 0, 0, 1],
    'left_gripper': 50.0,
    'right_position': [0.3, -0.2, 0.5],
    'right_orientation': [0, 0, 0, 1],
    'right_gripper': 50.0
})

# Cleanup
client.stop()
```

## Available Scripts

### Documentation
```bash
# Build and preview documentation (builds clean then serves)
uv run preview

# Build documentation only
uv run docs
```

### Testing
```bash
# Run tests
uv run test

# Run tests with coverage report (generates HTML report)
uv run test-cov
```

### Code Quality
```bash
# Format code with ruff
uv run format

# Lint code with ruff and mypy
uv run lint
```

### Command Line Tools
```bash
# Run Vision Pro teleoperation
uv run teleoperation --ip 192.168.200.111

# Run autonomous teleoperation
uv run autonomous-teleoperation --robot_ip 192.168.200.111

# Run robot executor node
uv run robot-executor --port 50051
```

## Components

### PolicyClient
Simple policy-agnostic bridge for executing actions from autonomous policy.

### Teleoperation
Real-time teleoperation using Vision Pro data.

### Robot Executor
Network-based robot executor for distributed control.

## Development

```bash
# Install dev dependencies
uv sync --dev

# Format and lint
uv run format
uv run lint

# Run tests
uv run test
```

## License

MIT License. Copyright (c) 2024-2025, FortyFive Labs Incorporated.
