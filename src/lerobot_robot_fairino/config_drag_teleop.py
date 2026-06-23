"""Configuration for the Fairino FR5 drag-teach (kinesthetic) teleoperator."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("fairino_drag_teleop")
@dataclass
class FairinoDragTeleopConfig(TeleoperatorConfig):
    """Drag-teach teleop for a single FR5.

    It shares the FairinoFR5 robot's SDK connection (the FR controller allows only one
    state client), so ``ip`` must match the robot's ``ip``. During recording the operator
    hand-guides the arm in free-drive; ``get_action()`` returns the live joint angles,
    which lerobot-record stores as the demonstrated action.
    """

    ip: str = "192.168.58.2"  # must match the FairinoFR5 robot's ip (shared connection)
