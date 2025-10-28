"""
Vuer Envs Astribot
==================

Autonomous teleoperation bridge for Astribot robot with Vision Pro integration.

This package provides:
- PolicyClient: Simple interface for autonomous policy execution on robot
- Teleoperation: Vision Pro to robot teleoperation
- Robot Executor: Network to robot execution node
"""

from vuer_envs_astribot.autonomous_teleoperation import PolicyClient

__version__ = "0.1.0"
__all__ = ["PolicyClient"]
