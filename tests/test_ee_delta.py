"""EE-delta (Cartesian / gamepad) action mode against MockRPC."""

import pytest

from lerobot_robot_fairino import FairinoFR5, FairinoFR5Config

EE_KEYS = {"delta_x", "delta_y", "delta_z"}


def _ee_robot(**kw):
    kw.setdefault("fps", 30)
    r = FairinoFR5(FairinoFR5Config(mock=True, action_mode="ee_delta", **kw))
    r.connect()
    return r


def test_ee_action_features_match_gamepad_keys():
    r = FairinoFR5(FairinoFR5Config(mock=True, action_mode="ee_delta"))
    assert set(r.action_features) == EE_KEYS  # exactly the gamepad teleop's keys


def test_ee_scales_delta_to_mm_and_servocarts():
    r = _ee_robot(ee_step_scale=10.0, max_ee_step_mm=20.0)
    try:
        before = list(r.robot.robot_state_pkg.tl_cur_pos)
        sent = r.send_action({"delta_x": 1.0, "delta_y": 0.0, "delta_z": -0.5})
        # 1.0 * 10 mm = 10; -0.5 * 10 = -5 (both within the 20 mm cap)
        assert sent["delta_x"] == pytest.approx(10.0)
        assert sent["delta_z"] == pytest.approx(-5.0)
        after = r.robot.robot_state_pkg.tl_cur_pos
        assert after[0] == pytest.approx(before[0] + 10.0)  # incremental ServoCart applied
        assert after[2] == pytest.approx(before[2] - 5.0)
    finally:
        r.disconnect()


def test_ee_per_tick_clamp():
    r = _ee_robot(ee_step_scale=1000.0, max_ee_step_mm=20.0)
    try:
        sent = r.send_action({"delta_x": 1.0, "delta_y": 1.0, "delta_z": 1.0})
        assert sent["delta_x"] == pytest.approx(20.0)  # clamped from 1000 mm
    finally:
        r.disconnect()


def test_ee_delta_sign_flip():
    r = _ee_robot(ee_step_scale=10.0, ee_delta_sign=(-1.0, 1.0, 1.0))
    try:
        sent = r.send_action({"delta_x": 1.0, "delta_y": 0.0, "delta_z": 0.0})
        assert sent["delta_x"] == pytest.approx(-10.0)
    finally:
        r.disconnect()


def test_ee_missing_axis_defaults_to_zero():
    r = _ee_robot()
    try:
        sent = r.send_action({"delta_x": 0.0})  # delta_y / delta_z absent
        assert sent["delta_y"] == pytest.approx(0.0)
        assert sent["delta_z"] == pytest.approx(0.0)
    finally:
        r.disconnect()


def test_ee_servo_session_starts_over_xmlrpc():
    # ServoCart is XML-RPC-only, so EE mode must start the servo session with cmdType 0
    # even if servo_cmd_type=1 was requested.
    r = _ee_robot(servo_cmd_type=1)
    try:
        assert r._servo_cmd_type == 0
    finally:
        r.disconnect()


def test_invalid_action_mode_rejected():
    with pytest.raises(ValueError):
        FairinoFR5Config(mock=True, action_mode="bogus")


def test_invalid_ee_frame_rejected():
    with pytest.raises(ValueError):
        FairinoFR5Config(mock=True, ee_frame="world")


def test_joint_mode_still_default():
    r = FairinoFR5(FairinoFR5Config(mock=True))
    assert r.config.action_mode == "joint"
    assert "joint1.pos" in r.action_features


# --- servo command parameters -------------------------------------------------------

def test_ee_cmd_t_clamped_to_ceiling():
    r = _ee_robot(fps=30)  # 1/30 = 0.033 s exceeds the 0.016 s ceiling
    try:
        r.send_action({"delta_x": 1.0})
        assert r.robot.last_servocart["cmdT"] == pytest.approx(0.016)
    finally:
        r.disconnect()


def test_ee_base_frame_uses_mode_1_with_unit_pos_gain():
    r = _ee_robot()
    try:
        r.send_action({"delta_x": 1.0})
        assert r.robot.last_servocart["mode"] == 1
        assert r.robot.last_servocart["pos_gain"] == [1.0] * 6
    finally:
        r.disconnect()


def test_ee_tool_frame_uses_mode_2():
    r = _ee_robot(ee_frame="tool")
    try:
        r.send_action({"delta_x": 1.0})
        assert r.robot.last_servocart["mode"] == 2
    finally:
        r.disconnect()


def test_ee_idle_deadband_skips_servocart():
    r = _ee_robot()
    try:
        r.send_action({"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0})
        assert r.robot.last_servocart is None  # no command streamed when centered
    finally:
        r.disconnect()


# --- fault handling -----------------------------------------------------------------

def test_ee_benign_rejection_is_skipped_not_raised():
    r = _ee_robot(max_consecutive_servo_errors=5)
    try:
        r.robot.servocart_return_code = 14  # rejected, but no active safety fault
        r.send_action({"delta_x": 1.0})     # should log-and-skip, not raise
        assert r._ee_reject_count == 1
    finally:
        r.disconnect()


def test_ee_safety_fault_aborts_immediately():
    r = _ee_robot()
    try:
        r.robot.servocart_return_code = 7
        r.robot.safety_code = 7  # genuine safety/protective stop
        with pytest.raises(RuntimeError):
            r.send_action({"delta_x": 1.0})
    finally:
        r.disconnect()


def test_ee_consecutive_rejections_abort():
    r = _ee_robot(max_consecutive_servo_errors=3)
    try:
        r.robot.servocart_return_code = 14  # benign reject, safety ok
        r.send_action({"delta_x": 1.0})
        r.send_action({"delta_x": 1.0})
        with pytest.raises(RuntimeError):
            r.send_action({"delta_x": 1.0})
    finally:
        r.disconnect()


# --- gripper (discrete gamepad command) ---------------------------------------------

def test_ee_gripper_open_close_stay():
    r = _ee_robot(use_gripper=True)
    try:
        base = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0}
        r.send_action({**base, "gripper": 2})  # OPEN
        assert r.robot.robot_state_pkg.gripper_position == 100
        r.send_action({**base, "gripper": 0})  # CLOSE
        assert r.robot.robot_state_pkg.gripper_position == 0
        calls = r.robot.move_gripper_calls
        r.send_action({**base, "gripper": 1})  # STAY -> no MoveGripper
        assert r.robot.move_gripper_calls == calls
    finally:
        r.disconnect()


def test_ee_gripper_command_on_change_only():
    r = _ee_robot(use_gripper=True)
    try:
        base = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0}
        r.send_action({**base, "gripper": 2})
        n = r.robot.move_gripper_calls
        r.send_action({**base, "gripper": 2})  # identical -> deduped
        assert r.robot.move_gripper_calls == n
    finally:
        r.disconnect()


def test_ee_gripper_dropped_warns_when_disabled(caplog):
    import logging

    r = _ee_robot(use_gripper=False)
    try:
        with caplog.at_level(logging.WARNING):
            r.send_action({"delta_x": 0.0, "gripper": 2})
        assert any("use_gripper=False" in rec.message for rec in caplog.records)
    finally:
        r.disconnect()


def test_mockrpc_servocart_modes():
    from lerobot_robot_fairino._sdk import MockRPC

    m = MockRPC()
    m.ServoCart(0, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0], [0, 0, 0, 0])  # absolute
    assert list(m.robot_state_pkg.tl_cur_pos) == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
    m.ServoCart(1, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0])  # incremental
    assert m.robot_state_pkg.tl_cur_pos[0] == pytest.approx(2.0)
