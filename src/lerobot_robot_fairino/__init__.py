"""Fairino FR5 robot plugin for LeRobot.

Importing this package registers the ``fairino_fr5`` robot type with LeRobot (via the
``@RobotConfig.register_subclass`` decorator on :class:`FairinoFR5Config`). LeRobot's
plugin discovery imports this package automatically because its distribution name starts
with ``lerobot_robot_``.
"""

from .config_fairino_fr5 import FairinoFR5Config
from .fairino_fr5 import FairinoFR5

__all__ = ["FairinoFR5", "FairinoFR5Config"]
__version__ = "0.1.0"
