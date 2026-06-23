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
EE_DELTA_KEYS = ["delta_x", "delta_y", "delta_z"]  # match LeRobot gamepad teleop keys
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
        self._enabled = False  # whether RobotEnable(1) was issued (set in configure)
        self._servo_cmd_type = 0  # transport used for ServoMoveStart/End (set in configure)
        self._last_gripper_cmd: int | None = None
        self._ee_reject_count = 0  # consecutive ServoCart rejections
        self._warned_gripper_dropped = False

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
    def _ee_ft(self) -> dict[str, type]:
        return {k: float for k in EE_DELTA_KEYS}

    @property
    def action_features(self) -> dict[str, type]:
        if self.config.action_mode == "ee_delta":
            ft = dict(self._ee_ft)
            if self.config.use_gripper:
                ft["gripper"] = float  # gamepad emits a discrete open/close/stay command
            return ft
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
        if not self.config.mock and not self.config.require_cnde:
            # The SDK gates every call behind ``is_connect``, which requires the CNDE
            # real-time-state stream (port 20005). This driver reads over XML-RPC and
            # does not use CNDE, so open the gate and verify XML-RPC actually responds.
            type(self.robot).is_connect = True
            self._assert_xmlrpc_alive()
        for cam in self.cameras.values():
            cam.connect()
        self.configure()
        logger.info("FairinoFR5 connected at %s", self.config.ip)

    def _assert_xmlrpc_alive(self) -> None:
        ret = self.robot.GetActualJointPosDegree(1)
        if not (isinstance(ret, (list, tuple)) and ret[0] == 0):
            raise ConnectionError(
                f"FR5 XML-RPC not responding (GetActualJointPosDegree returned {ret!r}). "
                f"Check the controller IP ({self.config.ip}) and that the arm is reachable."
            )

    def _make_rpc_cls(self):
        if self.config.mock:
            from ._sdk import MockRPC

            return MockRPC
        return load_rpc()

    def configure(self) -> None:
        r = self.robot
        _ok(r.RobotEnable(1), "RobotEnable")
        self._enabled = True
        _ok(r.Mode(0), "Mode(automatic)")  # 0 = automatic mode, required for ServoJ
        if self.config.use_gripper:
            _ok(
                r.SetGripperConfig(self.config.gripper_company, self.config.gripper_device),
                "SetGripperConfig",
            )
            _ok(r.ActGripper(self.config.gripper_index, 1), "ActGripper")
        # ServoCart is XML-RPC only, so EE mode must start the servo session over XML-RPC (0).
        self._servo_cmd_type = 0 if self.config.action_mode == "ee_delta" else self.config.servo_cmd_type
        _ok(r.ServoMoveStart(self._servo_cmd_type), "ServoMoveStart")
        self._servo_started = True

    def disconnect(self) -> None:
        if self.robot is not None:
            try:
                if self._servo_started:
                    try:
                        # Halt any residual incremental motion before ending the session.
                        if hasattr(self.robot, "StopMotion"):
                            self.robot.StopMotion()
                        self.robot.ServoMoveEnd(self._servo_cmd_type)
                    except Exception:  # pragma: no cover - best-effort teardown
                        logger.exception("StopMotion/ServoMoveEnd failed during disconnect")
                if self.config.disable_on_disconnect and self._enabled:
                    try:
                        self.robot.RobotEnable(0)
                    except Exception:  # pragma: no cover
                        logger.exception("RobotEnable(0) failed during disconnect")
            finally:
                self.robot = None
                self._servo_started = False
                self._enabled = False
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
            speeds = self._unwrap(
                self.robot.GetActualJointSpeedsDegree(1), "GetActualJointSpeedsDegree"
            )
            obs.update({f"joint{i + 1}_vel.pos": float(speeds[i]) for i in range(6)})

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        check_safety(self.robot)

        if self.config.action_mode == "ee_delta":
            return self._send_ee_delta(action)
        return self._send_joint(action)

    def _servo_cmd_t(self) -> float:
        """Servo interpolation period, clamped to the controller's accepted ceiling.

        1/fps at low fps (e.g. 0.033 s @30Hz) exceeds the FR5 servo window; clamp it.
        Run fps >= 60 for the smoothest motion (so 1/fps stays within the band).
        """
        return min(1.0 / self.config.fps, self.config.servo_cmd_t_ceiling_s)

    def _send_joint(self, action: RobotAction) -> RobotAction:
        present = self._read_joints_deg()
        goal = [float(action[f"{j}.pos"]) for j in JOINTS]
        goal = enforce_joint_limits(goal, self.config.joint_limits_deg)
        goal = clip_joint_step(present, goal, self.config.max_joint_step_deg)

        cmd_t = self._servo_cmd_t()
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

    def _send_ee_delta(self, action: RobotAction) -> RobotAction:
        """Map gamepad EE-translation deltas to an incremental ServoCart command.

        ``delta_x/y/z`` arrive in roughly [-1, 1] (gamepad sticks); they are scaled to
        millimetres (``ee_step_scale``), per-axis sign-flipped (``ee_delta_sign``), and
        hard-clamped per tick (``max_ee_step_mm``). The FR5 controller solves IK onboard.
        """
        mode = 1 if self.config.ee_frame == "base" else 2  # ServoCart: 1=base, 2=tool
        scale = self.config.ee_step_scale
        cap = self.config.max_ee_step_mm
        delta_mm: list[float] = []
        for key, sgn in zip(EE_DELTA_KEYS, self.config.ee_delta_sign):
            d = float(action.get(key, 0.0)) * float(sgn) * scale
            delta_mm.append(max(-cap, min(cap, d)))

        sent: RobotAction = {key: delta_mm[i] for i, key in enumerate(EE_DELTA_KEYS)}

        # Idle deadband: don't stream zero increments when the stick is centered.
        if any(abs(d) > 1e-6 for d in delta_mm):
            desc_pos = [delta_mm[0], delta_mm[1], delta_mm[2], 0.0, 0.0, 0.0]
            code = self.robot.ServoCart(
                mode,
                desc_pos,
                NO_EXT_AXES,
                pos_gain=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 1:1; we scale/clamp ourselves
                acc=0.0,
                vel=0.0,
                cmdT=self._servo_cmd_t(),
                filterT=self.config.servo_filter_t,
                gain=0.0,
            )
            code = int(code[0] if isinstance(code, (list, tuple)) else code)
            if code != 0:
                self._ee_reject_count += 1
                logger.warning(
                    "ServoCart rejected increment %s (code %s, %d consecutive)",
                    desc_pos[:3], code, self._ee_reject_count,
                )
                # ServoCart returns the safety code when the controller is faulted, so
                # re-check: this raises and aborts the loop on a genuine safety/protective
                # stop, while a benign singularity/limit rejection is skipped.
                check_safety(self.robot)
                if self._ee_reject_count >= self.config.max_consecutive_servo_errors:
                    raise RuntimeError(
                        f"ServoCart rejected {self._ee_reject_count} consecutive ticks; aborting"
                    )
            else:
                self._ee_reject_count = 0

        if "gripper" in action:
            if self.config.use_gripper:
                self._send_gamepad_gripper(action["gripper"])
                sent["gripper"] = float(action["gripper"])
            elif not self._warned_gripper_dropped:
                logger.warning(
                    "Received a 'gripper' action but use_gripper=False; ignoring gripper "
                    "input. Set --robot.use_gripper=true (or --teleop.use_gripper=false)."
                )
                self._warned_gripper_dropped = True
        return sent

    def _send_gamepad_gripper(self, command: float) -> None:
        """Handle the gamepad's discrete gripper command (0=close, 1=stay, 2=open).

        Untested without a gripper; gated behind ``use_gripper``.
        """
        cmd = int(round(float(command)))
        if cmd == 1:  # STAY
            return
        target = 0 if cmd == 0 else 100  # CLOSE -> 0, OPEN -> 100
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
                    0,
                    self.config.gripper_type,
                    0,
                    0,
                    0,
                ),
                "MoveGripper",
            )
            self._last_gripper_cmd = target

    # ----------------------------------------------------------------- reads
    @staticmethod
    def _unwrap(ret, what: str) -> list[float]:
        """Validate an SDK getter return of the form ``(0, [..])``.

        SDK getters return ``(0, [values])`` on success but a bare int error code on
        failure, so unpack defensively and surface a clear error.
        """
        if not (isinstance(ret, (list, tuple)) and len(ret) >= 2 and ret[0] == 0):
            raise ConnectionError(f"{what} failed (returned {ret!r})")
        return [float(x) for x in ret[1]]

    def _read_joints_deg(self) -> list[float]:
        if self.config.use_fast_state_read:
            pkg = self.robot.robot_state_pkg
            return [float(pkg.jt_cur_pos[i]) for i in range(6)]
        return self._unwrap(self.robot.GetActualJointPosDegree(1), "GetActualJointPosDegree")

    def _read_tcp(self) -> list[float]:
        if self.config.use_fast_state_read:
            pkg = self.robot.robot_state_pkg
            return [float(pkg.tl_cur_pos[i]) for i in range(6)]
        return self._unwrap(self.robot.GetActualTCPPose(1), "GetActualTCPPose")
