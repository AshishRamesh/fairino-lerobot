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

    # --- control loop ---
    fps: int = 30                      # control rate; ServoJ cmdT is derived as 1/fps
    servo_cmd_type: int = 0            # 0 = XML-RPC ServoJ, 1 = UDP passthrough (faster)
    servo_filter_t: float = 0.0        # ServoJ filterT low-pass; raise for noisy VLA chunks
    # Read joint/TCP state from the SDK's real-time UDP struct (robot_state_pkg.*) instead
    # of XML-RPC getters. Faster, but only correct if that struct is live-updated on your
    # controller/firmware — leave False until verified, then enable to cut read latency.
    use_fast_state_read: bool = False

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
