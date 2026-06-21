"""observation_features / action_features shape and the .pos-suffix invariant."""

from lerobot_robot_fairino import FairinoFR5, FairinoFR5Config

JOINT_KEYS = {f"joint{i}.pos" for i in range(1, 7)}


def _scalar_keys(features):
    return {k for k, v in features.items() if v is float}


def test_joints_only_features():
    robot = FairinoFR5(FairinoFR5Config(mock=True))
    assert _scalar_keys(robot.observation_features) == JOINT_KEYS
    assert set(robot.action_features) == JOINT_KEYS


def test_all_scalar_features_end_in_pos():
    # LeRobot's rollout pipeline keeps only scalar features whose key ends in ".pos".
    cfg = FairinoFR5Config(
        mock=True, include_tcp_pose=True, include_joint_velocity=True, use_gripper=True
    )
    robot = FairinoFR5(cfg)
    for key, val in robot.observation_features.items():
        if val is float:
            assert key.endswith(".pos"), key
    for key in robot.action_features:
        assert key.endswith(".pos"), key


def test_optional_extras_toggle_features():
    base = FairinoFR5(FairinoFR5Config(mock=True))
    full = FairinoFR5(
        FairinoFR5Config(
            mock=True, include_tcp_pose=True, include_joint_velocity=True, use_gripper=True
        )
    )
    assert "gripper.pos" not in base.observation_features
    assert "gripper.pos" in full.observation_features
    assert "gripper.pos" in full.action_features
    assert {f"tcp_{a}.pos" for a in ("x", "y", "z", "rx", "ry", "rz")} <= set(
        full.observation_features
    )
    assert {f"joint{i}_vel.pos" for i in range(1, 7)} <= set(full.observation_features)


def test_features_available_before_connect():
    # Must not depend on a live connection.
    robot = FairinoFR5(FairinoFR5Config(mock=True))
    assert not robot.is_connected
    assert robot.observation_features
    assert robot.action_features
