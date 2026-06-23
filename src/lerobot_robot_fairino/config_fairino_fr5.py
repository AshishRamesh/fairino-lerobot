"""Configuration for the Fairino FR5 LeRobot robot driver."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("fairino_fr5")
@dataclass
class FairinoFR5Config(RobotConfig):
    """Config for a Fairino FR5 collaborative arm controlled over TCP/IP.

    Joint values flow as raw degrees: ``get_observation``/``send_action`` use the FR5's
    native joint-angle units. Normalization is computed at training time from dataset
    statistics — do not normalize here.
    """

    # --- connection ---
    ip: str = "192.168.58.2"           # FR controller default; confirm on your subnet
    mock: bool = False                 # use the in-process MockRPC (no hardware)
    # The SDK gates all calls behind is_connect, which requires the CNDE state stream
    # (port 20005). This driver reads over XML-RPC and does not use CNDE, so by default
    # we don't require it. Set True only if your setup genuinely depends on CNDE.
    require_cnde: bool = False

    # --- control loop ---
    fps: int = 30                      # control rate; servo cmdT = min(1/fps, servo_cmd_t_ceiling_s)
    servo_cmd_type: int = 0            # 0 = XML-RPC ServoJ, 1 = UDP passthrough (faster)
    servo_filter_t: float = 0.0        # ServoJ/ServoCart filterT low-pass; raise for noisy chunks
    # Controller interpolation period ceiling. The FR5 servo firmware expects cmdT in
    # ~[0.001, 0.016] s; 1/fps at 30 Hz (0.033 s) is out of range, so cmdT is clamped to
    # this. For smooth motion run fps >= 60 (so 1/fps falls inside the band).
    servo_cmd_t_ceiling_s: float = 0.016
    max_consecutive_servo_errors: int = 5  # abort after this many back-to-back servo rejects
    # Read joint/TCP state from the SDK's real-time UDP struct (robot_state_pkg.*) instead
    # of XML-RPC getters. Faster, but only correct if that struct is live-updated on your
    # controller/firmware — leave False until verified, then enable to cut read latency.
    use_fast_state_read: bool = False

    # --- action mode ---
    # "joint":    action = joint{1..6}.pos (degrees) -> ServoJ. Default; used by policies.
    # "ee_delta": action = delta_x/delta_y/delta_z (matches the LeRobot gamepad teleop)
    #             -> ServoCart incremental Cartesian. The FR5 controller solves IK onboard
    #             (no URDF/solver needed). Use this for gamepad/Xbox teleoperation.
    action_mode: str = "joint"
    # Kinesthetic teaching: when True, connect() puts the arm in free-drive
    # (DragTeachSwitch) instead of a servo session, and send_action() is a no-op so the
    # driver never fights the operator. Use with --teleop.type=fairino_drag_teleop to
    # record demos. Joint trajectories are recorded as the action (keys end in .pos).
    drag_teach: bool = False
    ee_frame: str = "base"             # "base" (ServoCart mode 1) or "tool" (ServoCart mode 2)
    # Conservative defaults for first bring-up: 5 mm/unit => ~150 mm/s at full stick @30Hz,
    # capped at 10 mm/tick (~300 mm/s). Raise once direction/scale are verified on hardware.
    ee_step_scale: float = 5.0         # mm of Cartesian motion per unit gamepad delta (delta in [-1,1])
    max_ee_step_mm: float = 10.0       # hard per-tick Cartesian translation clamp (safety backstop)
    ee_delta_sign: tuple[float, float, float] = (1.0, 1.0, 1.0)  # flip an axis (set -1) if inverted

    # --- safety (degrees) ---
    max_joint_step_deg: float = 4.0    # hard per-tick delta clip (~120 deg/s @ 30 Hz)
    joint_vel_limit_deg_s: float = 60.0  # passed to ServoJ vel
    joint_acc_limit: float = 0.0       # passed to ServoJ acc (0 = controller default)
    # Per-joint (min, max) software limits; None -> FR5 factory defaults (see safety.py).
    joint_limits_deg: dict[str, tuple[float, float]] | None = None

    # --- optional observation extras ---
    # NOTE: every scalar observation/action key MUST end in ".pos" or LeRobot's rollout
    # pipeline drops it. These extras therefore use tcp_x.pos / joint1_vel.pos etc.
    include_tcp_pose: bool = False     # adds tcp_{x,y,z,rx,ry,rz}.pos
    include_joint_velocity: bool = False  # adds joint{1..6}_vel.pos

    # --- gripper (off by default; 6-DoF arm) ---
    use_gripper: bool = False
    gripper_company: int = 4           # SetGripperConfig vendor id (model-specific)
    gripper_device: int = 0            # SetGripperConfig device id
    gripper_type: int = 0              # 0 = parallel, 1 = rotary
    gripper_index: int = 1
    gripper_speed: int = 50            # 0-100
    gripper_force: int = 50            # 0-100
    gripper_max_time_ms: int = 30000
    gripper_command_epsilon: float = 2.0  # re-command only if |delta| > eps (0-100 scale)

    # --- teardown ---
    disable_on_disconnect: bool = True  # RobotEnable(0) on disconnect

    # --- cameras --- (empty by default; pass via --robot.cameras at runtime)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()  # validates each camera has width/height/fps
        if self.action_mode not in ("joint", "ee_delta"):
            raise ValueError(
                f"action_mode must be 'joint' or 'ee_delta', got {self.action_mode!r}"
            )
        if self.ee_frame not in ("base", "tool"):
            raise ValueError(f"ee_frame must be 'base' or 'tool', got {self.ee_frame!r}")
