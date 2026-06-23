"""Fairino FR5 drag-teach teleoperator for kinesthetic demo collection.

Pairs with a ``FairinoFR5`` robot started in ``drag_teach`` mode. The robot enters
free-drive and suppresses motion; this teleop reads the same arm's live joint angles
(over the robot's shared connection) and returns them as the action, so ``lerobot-record``
captures the hand-guided trajectory. The action keys match the robot's ``action_features``
(``joint1.pos`` … ``joint6.pos``), so recorded datasets train and roll out unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction

from .config_drag_teleop import FairinoDragTeleopConfig
from .fairino_fr5 import JOINTS, _ROBOT_BY_IP

logger = logging.getLogger(__name__)


class FairinoDragTeleop(Teleoperator):
    config_class = FairinoDragTeleopConfig
    name = "fairino_drag_teleop"

    def __init__(self, config: FairinoDragTeleopConfig):
        super().__init__(config)
        self.config = config
        self._robot = None

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{j}.pos": float for j in JOINTS}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._robot is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        robot = _ROBOT_BY_IP.get(self.config.ip)
        if robot is None:
            raise ConnectionError(
                f"No connected FairinoFR5 robot found at ip {self.config.ip}. The drag-teach "
                f"teleop shares the robot's connection — run, e.g.:\n"
                f"  lerobot-record --robot.type=fairino_fr5 --robot.drag_teach=true "
                f"--robot.ip={self.config.ip} --teleop.type=fairino_drag_teleop "
                f"--teleop.ip={self.config.ip} ..."
            )
        if not robot.config.drag_teach:
            logger.warning(
                "Robot at %s is connected but not in drag_teach mode; the arm won't be "
                "free to move. Pass --robot.drag_teach=true.",
                self.config.ip,
            )
        self._robot = robot

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> RobotAction:
        if self._robot is None:
            raise ConnectionError(f"{self} is not connected.")
        joints = self._robot._read_joints_deg()
        return {f"{j}.pos": float(v) for j, v in zip(JOINTS, joints)}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        # The robot owns the shared connection and exits drag mode on its own disconnect.
        self._robot = None
