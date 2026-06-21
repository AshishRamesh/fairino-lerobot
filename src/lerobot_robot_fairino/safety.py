"""Safety helpers for the Fairino FR5 driver.

These are software backstops layered on top of the controller's own hard joint limits
and safety system. The most important one is :func:`clip_joint_step`, which bounds how
far any joint can move in a single control tick — this protects against large jumps from
a policy (or a fat-fingered manual command) that would otherwise request a huge transient
joint velocity.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# FAIRINO FR5 factory joint motion ranges, degrees (min, max).
# Source: FR5 datasheet — J1 ±175, J2 +85/-265, J3 ±160, J4 +85/-265, J5 ±175, J6 ±175.
# Max joint speed is ±180 deg/s, so at 30 Hz the hardware ceiling is ~6 deg/tick.
FR5_DEFAULT_JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "joint1": (-175.0, 175.0),
    "joint2": (-265.0, 85.0),
    "joint3": (-160.0, 160.0),
    "joint4": (-265.0, 85.0),
    "joint5": (-175.0, 175.0),
    "joint6": (-175.0, 175.0),
}

JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]


def resolve_joint_limits(
    limits: dict[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]]:
    """Return per-joint (min, max) limits, falling back to the FR5 factory defaults."""
    if limits is None:
        return dict(FR5_DEFAULT_JOINT_LIMITS_DEG)
    merged = dict(FR5_DEFAULT_JOINT_LIMITS_DEG)
    merged.update(limits)
    return merged


def enforce_joint_limits(
    goal_deg: list[float],
    limits: dict[str, tuple[float, float]] | None,
) -> list[float]:
    """Clamp each joint goal (degrees) into its software limit, warning on clamp."""
    resolved = resolve_joint_limits(limits)
    out: list[float] = []
    for name, value in zip(JOINT_NAMES, goal_deg):
        lo, hi = resolved[name]
        clamped = min(max(value, lo), hi)
        if clamped != value:
            logger.warning(
                "%s goal %.2f deg outside software limit [%.1f, %.1f]; clamped to %.2f",
                name, value, lo, hi, clamped,
            )
        out.append(clamped)
    return out


def clip_joint_step(
    present_deg: list[float],
    goal_deg: list[float],
    max_step_deg: float,
) -> list[float]:
    """Cap the per-joint move relative to the *measured* present position.

    With a control loop at ``fps`` Hz, ``max_step_deg`` bounds joint velocity to roughly
    ``max_step_deg * fps`` deg/s. The cap is taken against the present (measured) position
    rather than the previous goal, so a stalled or lagging arm cannot accumulate a runaway
    error.
    """
    out: list[float] = []
    for present, goal in zip(present_deg, goal_deg):
        delta = goal - present
        if delta > max_step_deg:
            delta = max_step_deg
        elif delta < -max_step_deg:
            delta = -max_step_deg
        out.append(present + delta)
    return out


def check_safety(robot, mock: bool = False) -> None:
    """Gate before sending motion: raise if the controller reports a safety fault.

    Uses ``GetSafetyCode()`` when available; a non-zero code aborts the tick so the
    caller's ``finally`` can disconnect gracefully rather than driving a faulted arm.
    """
    if mock:
        return
    get_code = getattr(robot, "GetSafetyCode", None)
    if get_code is None:
        return
    try:
        code = get_code()
        code = code[0] if isinstance(code, (list, tuple)) else code
        code = int(code)
    except Exception:  # pragma: no cover - defensive; never block on a read hiccup
        logger.debug("GetSafetyCode() read failed; skipping safety gate this tick")
        return
    if code != 0:
        raise RuntimeError(f"Fairino controller reports safety code {code}; aborting motion")
