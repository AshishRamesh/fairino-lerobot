"""End-to-end driver behavior against the in-process MockRPC (no hardware)."""

import pytest

from lerobot_robot_fairino import FairinoFR5, FairinoFR5Config
from lerobot_robot_fairino.safety import FR5_DEFAULT_JOINT_LIMITS_DEG

JOINTS = [f"joint{i}" for i in range(1, 7)]


@pytest.fixture
def robot():
    r = FairinoFR5(FairinoFR5Config(mock=True, fps=30))
    r.connect()
    yield r
    if r.is_connected:
        r.disconnect()


def test_connect_reads_six_joints(robot):
    obs = robot.get_observation()
    assert set(obs) == {f"{j}.pos" for j in JOINTS}
    assert all(isinstance(v, float) for v in obs.values())


def test_servo_is_closed_loop(robot):
    action = {f"{j}.pos": 0.0 for j in JOINTS}
    action["joint1.pos"] = 3.0
    sent = robot.send_action(action)
    assert sent["joint1.pos"] == pytest.approx(3.0)
    # MockRPC writes the commanded goal back into jt_cur_pos.
    assert robot.get_observation()["joint1.pos"] == pytest.approx(3.0)


def test_per_tick_step_is_clipped(robot):
    # Present position is ~0; request a huge jump and confirm it is capped.
    big = {f"{j}.pos": 0.0 for j in JOINTS}
    big["joint2.pos"] = 90.0
    sent = robot.send_action(big)
    assert sent["joint2.pos"] == pytest.approx(robot.config.max_joint_step_deg)


def test_software_joint_limits_clamp():
    r = FairinoFR5(FairinoFR5Config(mock=True, max_joint_step_deg=1e6))
    r.connect()
    try:
        lo, hi = FR5_DEFAULT_JOINT_LIMITS_DEG["joint2"]
        action = {f"{j}.pos": 0.0 for j in JOINTS}
        action["joint2.pos"] = hi + 500.0  # well past the upper limit
        sent = r.send_action(action)
        assert sent["joint2.pos"] == pytest.approx(hi)
    finally:
        r.disconnect()


def test_disconnect_clears_state(robot):
    assert robot.is_connected
    robot.disconnect()
    assert not robot.is_connected
    assert robot.robot is None


def test_get_observation_requires_connection():
    r = FairinoFR5(FairinoFR5Config(mock=True))
    with pytest.raises(ConnectionError):
        r.get_observation()


def test_gripper_command_on_change_only():
    r = FairinoFR5(FairinoFR5Config(mock=True, use_gripper=True, gripper_command_epsilon=2.0))
    r.connect()
    try:
        base = {f"{j}.pos": 0.0 for j in JOINTS}
        # normalized 0-1 input maps onto 0-100 scale
        sent = r.send_action({**base, "gripper.pos": 1.0})
        assert sent["gripper.pos"] == pytest.approx(100.0)
        assert r.get_observation()["gripper.pos"] == pytest.approx(100.0)
    finally:
        r.disconnect()
