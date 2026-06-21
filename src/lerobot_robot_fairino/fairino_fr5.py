"""Fairino FR5 robot driver for LeRobot.

Implements the LeRobot :class:`~lerobot.robots.robot.Robot` interface on top of the
``fairino`` Python SDK (``Robot.RPC``), so the standard LeRobot tools (record, train,
rollout/VLA) work against the FR5 with no changes to LeRobot itself.

Pattern follows the SDK-over-IP robots (reachy2, unitree_g1): no Feetech/Dynamixel
MotorsBus, no driver-side normalization (raw joint degrees flow through), calibration is
a no-op (the arm is factory-calibrated).
"""

from __future__ import annotations

import logging
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation

from ._sdk import _ok, _to_gripper_0_100, load_rpc
from .config_fairino_fr5 import FairinoFR5Config
from .safety import check_safety, clip_joint_step, enforce_joint_limits

logger = logging.getLogger(__name__)

JOINTS = [f"joint{i}" for i in range(1, 7)]
TCP = ["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz"]
NO_EXT_AXES = [0.0, 0.0, 0.0, 0.0]  # FR controllers expect 4 external-axis slots


class FairinoFR5(Robot):
    config_class = FairinoFR5Config
    name = "fairino_fr5"

    def __init__(self, config: FairinoFR5Config):
        super().__init__(config)
        self.config = config
        self.robot = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self._servo_started = False
        self._last_gripper_cmd: int | None = None

    # ----------------------------------------------------------------- features
    @property
    def _joint_ft(self) -> dict[str, type]:
        return {f"{j}.pos": float for j in JOINTS}

    @property
    def _gripper_ft(self) -> dict[str, type]:
        return {"gripper.pos": float} if self.config.use_gripper else {}

    @property
    def _tcp_ft(self) -> dict[str, type]:
        return {f"{t}.pos": float for t in TCP} if self.config.include_tcp_pose else {}

    @property
    def _vel_ft(self) -> dict[str, type]:
        if not self.config.include_joint_velocity:
            return {}
        return {f"joint{i}_vel.pos": float for i in range(1, 7)}

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, Any]:
        return {
            **self._joint_ft,
            **self._gripper_ft,
            **self._tcp_ft,
            **self._vel_ft,
            **self._cameras_ft,
        }

    @property
    def action_features(self) -> dict[str, type]:
        return {**self._joint_ft, **self._gripper_ft}

    # ----------------------------------------------------------------- status
    @property
    def is_connected(self) -> bool:
        return self.robot is not None and all(
            cam.is_connected for cam in self.cameras.values()
        )

    @property
    def is_calibrated(self) -> bool:
        return True  # FR5 is factory-calibrated; no MotorsBus calibration needed

    def calibrate(self) -> None:
        pass

    # ----------------------------------------------------------------- lifecycle
    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            logger.info("%s already connected", self)
            return
        rpc_cls = self._make_rpc_cls()
        self.robot = rpc_cls(self.config.ip)  # opens XML-RPC (+ UDP state thread)
        for cam in self.cameras.values():
            cam.connect()
        self.configure()
        logger.info("FairinoFR5 connected at %s", self.config.ip)

    def _make_rpc_cls(self):
        if self.config.mock:
            from ._sdk import MockRPC

            return MockRPC
        return load_rpc()

    def configure(self) -> None:
        r = self.robot
        _ok(r.RobotEnable(1), "RobotEnable")
        _ok(r.Mode(0), "Mode(automatic)")  # 0 = automatic mode, required for ServoJ
        if self.config.use_gripper:
            _ok(
                r.SetGripperConfig(self.config.gripper_company, self.config.gripper_device),
                "SetGripperConfig",
            )
            _ok(r.ActGripper(self.config.gripper_index, 1), "ActGripper")
        _ok(r.ServoMoveStart(self.config.servo_cmd_type), "ServoMoveStart")
        self._servo_started = True

    def disconnect(self) -> None:
        if self.robot is not None:
            try:
                if self._servo_started:
                    try:
                        self.robot.ServoMoveEnd(self.config.servo_cmd_type)
                    except Exception:  # pragma: no cover - best-effort teardown
                        logger.exception("ServoMoveEnd failed during disconnect")
                if self.config.disable_on_disconnect:
                    try:
                        self.robot.RobotEnable(0)
                    except Exception:  # pragma: no cover
                        logger.exception("RobotEnable(0) failed during disconnect")
            finally:
                self.robot = None
                self._servo_started = False
                self._last_gripper_cmd = None
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info("FairinoFR5 disconnected")

    # ----------------------------------------------------------------- I/O
    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        joints = self._read_joints_deg()
        obs: RobotObservation = {f"{j}.pos": float(v) for j, v in zip(JOINTS, joints)}

        if self.config.use_gripper:
            obs["gripper.pos"] = float(self.robot.robot_state_pkg.gripper_position)

        if self.config.include_tcp_pose:
            tcp = self._read_tcp()
            obs.update({f"{t}.pos": float(v) for t, v in zip(TCP, tcp)})

        if self.config.include_joint_velocity:
            _err, speeds = self.robot.GetActualJointSpeedsDegree(1)
            obs.update({f"joint{i + 1}_vel.pos": float(speeds[i]) for i in range(6)})

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        check_safety(self.robot, mock=self.config.mock)

        present = self._read_joints_deg()
        goal = [float(action[f"{j}.pos"]) for j in JOINTS]
        goal = enforce_joint_limits(goal, self.config.joint_limits_deg)
        goal = clip_joint_step(present, goal, self.config.max_joint_step_deg)

        cmd_t = 1.0 / self.config.fps  # interpolate over exactly one control tick
        _ok(
            self.robot.ServoJ(
                goal,
                NO_EXT_AXES,
                acc=self.config.joint_acc_limit,
                vel=self.config.joint_vel_limit_deg_s,
                cmdT=cmd_t,
                filterT=self.config.servo_filter_t,
                gain=0.0,
                id=0,
                cmdType=self.config.servo_cmd_type,
            ),
            "ServoJ",
        )
        sent: RobotAction = {f"{j}.pos": goal[i] for i, j in enumerate(JOINTS)}

        if self.config.use_gripper and "gripper.pos" in action:
            target = _to_gripper_0_100(action["gripper.pos"])
            if (
                self._last_gripper_cmd is None
                or abs(target - self._last_gripper_cmd) > self.config.gripper_command_epsilon
            ):
                _ok(
                    self.robot.MoveGripper(
                        self.config.gripper_index,
                        target,
                        self.config.gripper_speed,
                        self.config.gripper_force,
                        self.config.gripper_max_time_ms,
                        0,  # block = 0 (non-blocking, so the loop is not stalled)
                        self.config.gripper_type,
                        0,  # rotNum
                        0,  # rotVel
                        0,  # rotTorque
                    ),
                    "MoveGripper",
                )
                self._last_gripper_cmd = target
            sent["gripper.pos"] = float(target)

        return sent

    # ----------------------------------------------------------------- reads
    def _read_joints_deg(self) -> list[float]:
        if self.config.use_fast_state_read:
            pkg = self.robot.robot_state_pkg
            return [float(pkg.jt_cur_pos[i]) for i in range(6)]
        _err, pos = self.robot.GetActualJointPosDegree(1)  # flag 1 = non-blocking
        return [float(x) for x in pos]

    def _read_tcp(self) -> list[float]:
        if self.config.use_fast_state_read:
            pkg = self.robot.robot_state_pkg
            return [float(pkg.tl_cur_pos[i]) for i in range(6)]
        _err, pose = self.robot.GetActualTCPPose(1)
        return [float(x) for x in pose]
