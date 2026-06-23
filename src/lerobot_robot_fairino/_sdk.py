"""Fairino SDK access layer: lazy loader, small helpers, and an offline MockRPC.

The real SDK (`fairino` package) is vendored under ``_vendor/`` and is pure-Python
(stdlib only), but it is imported lazily — only when a real connection is opened — so
this plugin imports cleanly for plugin discovery, mock runs, and unit tests on machines
that do not have the SDK or a controller present.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FairinoSDKError(RuntimeError):
    """Raised when a Fairino SDK call returns a non-zero error code."""

    def __init__(self, op: str, code: int):
        self.op = op
        self.code = code
        super().__init__(f"Fairino SDK call {op!r} failed with error code {code}")


def _ok(ret: Any, op: str) -> int:
    """Validate the return of an action-type SDK call.

    Fairino action methods return an ``int`` error code (0 = success); a few return a
    tuple whose first element is the code. Raises :class:`FairinoSDKError` on non-zero.
    """
    code = ret[0] if isinstance(ret, (list, tuple)) else ret
    try:
        code = int(code)
    except (TypeError, ValueError):
        # Unexpected shape — log and treat as success rather than abort a live episode.
        logger.warning("Fairino %s returned unparseable value %r; assuming ok", op, ret)
        return 0
    if code != 0:
        raise FairinoSDKError(op, code)
    return code


def _to_gripper_0_100(value: float) -> int:
    """Map a gripper action to the SDK's 0-100 position scale.

    Accepts a normalized 0-1 value (scaled x100) or a raw 0-100 value; clamps to [0,100].
    """
    v = float(value)
    if 0.0 <= v <= 1.0:
        v = v * 100.0
    return int(max(0.0, min(100.0, v)))


def _ensure_vendor_on_path() -> None:
    vendor = Path(__file__).resolve().parent / "_vendor"
    if (vendor / "fairino").is_dir() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


def load_rpc():
    """Lazily import the fairino SDK and return the ``Robot.RPC`` class."""
    _ensure_vendor_on_path()
    try:
        from fairino import Robot  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover - exercised only without the SDK
        raise ImportError(
            "Could not import the fairino SDK. It is vendored under "
            "lerobot_robot_fairino/_vendor/fairino; if removed, reinstall it from "
            "https://github.com/FAIR-INNOVATION/fairino-python-sdk (linux/fairino) "
            "or place it on PYTHONPATH. See the README."
        ) from e
    return Robot.RPC


# --------------------------------------------------------------------------------------
# Offline mock
# --------------------------------------------------------------------------------------
class _MockStatePkg:
    """Stand-in for the SDK's ctypes ``robot_state_pkg`` real-time struct."""

    def __init__(self) -> None:
        self.jt_cur_pos = [0.0] * 6                 # joint positions, degrees
        self.tl_cur_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # TCP pose [x,y,z,rx,ry,rz]
        self.gripper_position = 0                   # 0-100


class MockRPC:
    """In-process stand-in for ``fairino.Robot.RPC`` for offline tests and dry runs.

    Implements only the subset of methods the :class:`FairinoFR5` driver calls.
    ``ServoJ`` and ``MoveGripper`` write back into the fake state package so that
    subsequent reads are closed-loop (what you commanded is what you read back).
    """

    def __init__(self, ip: str = "192.168.58.2"):
        self.ip_address = ip
        self.robot_state_pkg = _MockStatePkg()
        # Test instrumentation / fault injection:
        self.safety_code = 0           # returned by GetSafetyCode()
        self.servocart_return_code = 0  # if non-zero, ServoCart rejects without moving
        self.last_servoj = None         # captured kwargs of the last ServoJ call
        self.last_servocart = None      # captured kwargs of the last ServoCart call
        self.move_gripper_calls = 0
        self.stop_motion_calls = 0
        self.robot_enable_states = []   # history of RobotEnable() args

    # connection / mode -----------------------------------------------------------------
    def RobotEnable(self, state):  # noqa: N802 (match SDK casing)
        self.robot_enable_states.append(int(state))
        return 0

    def Mode(self, state):  # noqa: N802
        return 0

    def ServoMoveStart(self, cmdType=0):  # noqa: N802, N803
        return 0

    def ServoMoveEnd(self, cmdType=0):  # noqa: N802, N803
        return 0

    def StopMotion(self):  # noqa: N802
        self.stop_motion_calls += 1
        return 0

    def GetSafetyCode(self):  # noqa: N802
        return self.safety_code

    # motion ----------------------------------------------------------------------------
    def ServoJ(self, joint_pos, axisPos, acc=0.0, vel=0.0, cmdT=0.008,  # noqa: N802, N803
               filterT=0.0, gain=0.0, id=0, cmdType=0):
        self.last_servoj = {"joint_pos": list(joint_pos), "cmdT": cmdT, "vel": vel}
        self.robot_state_pkg.jt_cur_pos = [float(x) for x in joint_pos][:6]
        return 0

    def ServoCart(self, mode, desc_pos, exaxis, pos_gain=None, acc=0.0,  # noqa: N802, N803
                  vel=0.0, cmdT=0.008, filterT=0.0, gain=0.0):
        self.last_servocart = {"mode": mode, "desc_pos": list(desc_pos), "cmdT": cmdT,
                               "pos_gain": pos_gain}
        # Like the real SDK, reject (return a code) without moving when "faulted".
        if self.servocart_return_code != 0:
            return self.servocart_return_code
        cur = self.robot_state_pkg.tl_cur_pos
        if mode == 0:  # absolute (base frame)
            self.robot_state_pkg.tl_cur_pos = [float(x) for x in desc_pos][:6]
        else:  # 1/2 = incremental (base/tool)
            self.robot_state_pkg.tl_cur_pos = [c + float(d) for c, d in zip(cur, desc_pos)]
        return 0

    # state reads -----------------------------------------------------------------------
    def GetActualJointPosDegree(self, flag=1):  # noqa: N802
        return 0, list(self.robot_state_pkg.jt_cur_pos)

    def GetActualJointPosRadian(self, flag=1):  # noqa: N802
        return 0, [math.radians(x) for x in self.robot_state_pkg.jt_cur_pos]

    def GetActualTCPPose(self, flag=1):  # noqa: N802
        return 0, list(self.robot_state_pkg.tl_cur_pos)

    def GetActualJointSpeedsDegree(self, flag=1):  # noqa: N802
        return 0, [0.0] * 6

    # gripper ---------------------------------------------------------------------------
    def SetGripperConfig(self, company, device, softversion=0, bus=0):  # noqa: N802
        return 0

    def ActGripper(self, index, action):  # noqa: N802
        return 0

    def MoveGripper(self, index, pos, vel, force, maxtime, block, type,  # noqa: N802, A002
                    rotNum, rotVel, rotTorque):  # noqa: N803
        self.move_gripper_calls += 1
        self.robot_state_pkg.gripper_position = int(pos)
        return 0

    def GetGripperMotionDone(self):  # noqa: N802
        return 0, [0, 1]

    # drag-teach (used by the future teleop phase) --------------------------------------
    def DragTeachSwitch(self, state):  # noqa: N802
        return 0

    def IsInDragTeach(self):  # noqa: N802
        return 0, 0
