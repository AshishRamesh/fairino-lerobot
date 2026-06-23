# lerobot_robot_fairino — Fairino FR5 plugin for LeRobot

A [LeRobot](https://github.com/huggingface/lerobot) robot driver for the **Fairino FR5**
6-axis collaborative arm. It implements LeRobot's `Robot` interface on top of the
`fairino` Python SDK (`Robot.RPC`, TCP/IP), so the standard LeRobot tools — `lerobot-record`,
`lerobot-train`, `lerobot-rollout` — work against the FR5 without any changes to LeRobot.

LeRobot auto-discovers this package (its distribution name starts with `lerobot_robot_`)
and exposes it as `--robot.type=fairino_fr5`.

> **Scope (v0.1):** the FR5 **robot driver** — connect, read joint state + cameras, stream
> joint servo commands. Enough to run a trained policy (inference). **Deferred:** a
> drag-teach teleoperator for demo collection, and the gripper (arm is treated as 6-DoF;
> gripper code exists but is off by default).

## Requirements

- A Fairino FR5 reachable over the network (default controller IP `192.168.58.2`).
- Python 3.12 + LeRobot ≥ 0.5 (built/tested against the `main` branch, `0.5.2-dev`).
- Linux (the live SDK talks to the controller over XML-RPC/UDP).

## Install

```bash
# 1. environment with the latest LeRobot (editable from a local clone, recommended)
conda create -y -n fr5-lerobot python=3.12
conda activate fr5-lerobot
pip install -e /path/to/lerobot            # add extras as needed, e.g. ".[smolvla]"

# 2. this plugin (editable)
cd /path/to/fairno-lerobot
pip install -e .
```

### Fairino SDK

The `fairino` SDK is **not on PyPI**. It is **vendored** in this repo
(`src/lerobot_robot_fairino/_vendor/fairino/`, pure-Python) and added to `sys.path`
lazily at connect time, so no extra install step is needed.

The vendored copy is pinned to **`v2.2.5_robot_v3.9.5`** (matches FR controller firmware
**3.9.5**). The SDK must match your controller's firmware — check the firmware version on
the teach pendant, then if it differs, swap the file for the matching release tag:

```bash
curl -fSL "https://raw.githubusercontent.com/FAIR-INNOVATION/fairino-python-sdk/<TAG>/linux/fairino/Robot.py" \
  -o src/lerobot_robot_fairino/_vendor/fairino/Robot.py   # TAG e.g. v2.2.5_robot_v3.9.5
```

> **CNDE note:** the SDK gates all calls behind `is_connect`, which requires the CNDE
> real-time-state stream (port 20005). This driver reads over XML-RPC and does **not**
> need CNDE, so `require_cnde=False` (default) opens that gate after verifying XML-RPC.
> If CNDE times out on your controller (e.g. only one state client allowed), the driver
> still works.

## Verify (no hardware)

```bash
# plugin is discovered + registered
python -c "from lerobot.utils.import_utils import register_third_party_plugins as r; r()"
python -c "import importlib.metadata as m; print([d.metadata['Name'] for d in m.distributions() if (d.metadata['Name'] or '').startswith('lerobot_robot_')])"
# -> ['lerobot_robot_fairino']

pytest -q                                   # mock-based unit tests
```

Mock end-to-end in Python:

```python
from lerobot_robot_fairino import FairinoFR5, FairinoFR5Config

robot = FairinoFR5(FairinoFR5Config(mock=True, fps=30))
robot.connect()
obs = robot.get_observation()               # {'joint1.pos': 0.0, ... 'joint6.pos': 0.0}
act = {k: v for k, v in obs.items() if k.endswith(".pos")}
act["joint1.pos"] += 3.0
robot.send_action(act)                       # clipped + servo'd
robot.disconnect()
```

## Live smoke test (real arm)

```python
from lerobot_robot_fairino import FairinoFR5, FairinoFR5Config

robot = FairinoFR5(FairinoFR5Config(ip="192.168.58.2", fps=30))
robot.connect()
print(robot.get_observation())               # 6 live joint angles (degrees)
act = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
act["joint1.pos"] += 3.0                      # jog J1 +3 deg (within the per-tick clip)
robot.send_action(act)
robot.disconnect()
```

Keep the physical e-stop in reach. The driver enforces a per-tick joint-delta clip
(`max_joint_step_deg`, default 4°) and software joint limits as backstops; the controller
enforces the true hard limits.

## Xbox / gamepad teleoperation (EE-delta mode)

Drive the arm live with a gamepad — the FR5 solves Cartesian→joint onboard (no URDF/IK).

1. **Pair the controller** (Bluetooth via `bluetoothctl`, or the Xbox wireless dongle via the
   `xone` driver) and confirm Linux sees it: `ls /dev/input/js*` then `jstest /dev/input/js0`.
2. **Install the gamepad deps:** `pip install "pygame>=2.5.1,<2.7.0" "hidapi>=0.14.0,<0.15.0"`.
3. **Run** (LeRobot's built-in `gamepad` teleop emits `delta_x/y/z`; `ee_delta` maps them to `ServoCart`):

```bash
lerobot-teleoperate \
  --robot.type=fairino_fr5 --robot.action_mode=ee_delta --robot.ip=192.168.58.2 --robot.fps=30 \
  --teleop.type=gamepad --teleop.use_gripper=false
```

Left stick → X/Y, right stick → Z. **First-run safety:** defaults are conservative
(`ee_step_scale=5`, `max_ee_step_mm=10` ≈ ≤300 mm/s) — **jog each axis at low scale first** and
confirm directions (use `ee_delta_sign` to flip an inverted axis), keep the e-stop in reach, then
raise the scale. For smoother motion run `--robot.fps=60` (servo `cmdT` is clamped to
`servo_cmd_t_ceiling_s=0.016`, so at 30 Hz motion is interpolated in ≤16 ms bursts). A genuine
controller safety fault aborts the loop; benign singularity/limit rejections are skipped (and abort
after `max_consecutive_servo_errors`). The gamepad's trigger gripper is ignored unless
`--robot.use_gripper=true`.

## Data collection — drag-teach (kinesthetic)

Record demos by hand-guiding the arm. The robot enters free-drive (`DragTeachSwitch`) and
suppresses motion; the `fairino_drag_teleop` teleop reads the same arm's live joints (over
the robot's shared connection) and records them as the action. Recorded action keys are
`joint1.pos … joint6.pos`, so the dataset trains and rolls out unchanged.

```bash
lerobot-record \
  --robot.type=fairino_fr5 --robot.drag_teach=true --robot.ip=192.168.58.2 --robot.fps=30 \
  --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }' \
  --robot.id=fr5_main \
  --teleop.type=fairino_drag_teleop --teleop.ip=192.168.58.2 \
  --dataset.repo_id=$HF_USER/fr5_pick --dataset.num_episodes=10 \
  --dataset.single_task="Pick up the cube" --dataset.fps=30 \
  --display_data=true
```

Notes:
- `--robot.drag_teach=true` and `--teleop.ip` **must equal** `--robot.ip` (one shared connection).
- Keep `--robot.action_mode=joint` (the default) — drag-teach records joint trajectories.
- If the arm isn't free to move by hand, enable manual/drag mode on the teach pendant.
- For **rollout**, drop `--robot.drag_teach` (default `false`) so the policy drives via `ServoJ`.

## Inference (after you have a trained policy)

```bash
# ACT (sync inference)
lerobot-rollout --strategy.type=base --inference.type=sync \
  --policy.path=/path/to/checkpoint \
  --robot.type=fairino_fr5 --robot.ip=192.168.58.2 --robot.fps=30 \
  --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }' \
  --task="pick up the cube" --duration=60

# SmolVLA / Pi0 (real-time chunking for slow VLAs)
lerobot-rollout --strategy.type=base --inference.type=rtc --inference.rtc.execution_horizon=10 \
  --policy.path=/path/to/vla_checkpoint \
  --robot.type=fairino_fr5 --robot.ip=192.168.58.2 --robot.fps=30 \
  --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }' \
  --task="pick up the cube" --duration=60
```

## Configuration highlights

| Field | Default | Notes |
|---|---|---|
| `ip` | `192.168.58.2` | FR controller address |
| `mock` | `False` | use the in-process `MockRPC` (no hardware) |
| `fps` | `30` | control rate; `ServoJ cmdT = 1/fps` for smooth motion |
| `servo_cmd_type` | `0` | `0` XML-RPC, `1` UDP passthrough (faster) |
| `servo_filter_t` | `0.0` | ServoJ low-pass; raise for noisy VLA action chunks |
| `use_fast_state_read` | `False` | read `robot_state_pkg.*`; enable once verified live |
| `action_mode` | `joint` | `joint` (ServoJ) or `ee_delta` (ServoCart, gamepad) |
| `ee_step_scale` | `5.0` | mm of Cartesian motion per unit gamepad delta |
| `max_ee_step_mm` | `10.0` | per-tick Cartesian translation clamp (safety) |
| `ee_frame` | `base` | `base` (ServoCart mode 1) or `tool` (mode 2) |
| `ee_delta_sign` | `(1,1,1)` | per-axis sign flip if a direction is inverted |
| `servo_cmd_t_ceiling_s` | `0.016` | servo interpolation period ceiling |
| `max_joint_step_deg` | `4.0` | per-tick joint-delta clip (safety) |
| `joint_limits_deg` | `None` | `None` → FR5 factory limits (see `safety.py`) |
| `include_tcp_pose` | `False` | add `tcp_*.pos` to observations |
| `include_joint_velocity` | `False` | add `joint*_vel.pos` to observations |
| `use_gripper` | `False` | enable gripper (set `gripper_company`/`device`/`type`) |
| `cameras` | `{}` | pass via `--robot.cameras` |

**Feature keys:** observations/actions are `joint1.pos … joint6.pos` (degrees), plus any
cameras and enabled extras. Every scalar key ends in `.pos` because LeRobot's rollout
pipeline only keeps `.pos`-suffixed scalars. Joint values are **raw degrees** — LeRobot
normalizes at training time from dataset statistics; the driver never normalizes.

## Roadmap (deferred)

1. Gripper bring-up once the model is known (`SetGripperConfig` company/device/type).
2. Camera wiring (OpenCV / RealSense) for the actual rig.
3. Full 6-DoF gamepad control (map the right stick to `drx/dry/drz` rotation increments).

Already supported: joint + EE-delta (gamepad) control, and **drag-teach recording**
(`--robot.drag_teach=true` + `--teleop.type=fairino_drag_teleop`).

Then the full loop is: record → `lerobot-train --policy.type=act|smolvla` → `lerobot-rollout`.
