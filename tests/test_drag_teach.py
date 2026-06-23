"""Drag-teach robot mode + shared-connection teleop, against MockRPC."""

import pytest

from lerobot.teleoperators.config import TeleoperatorConfig

from lerobot_robot_fairino import (
    FairinoDragTeleop,
    FairinoDragTeleopConfig,
    FairinoFR5,
    FairinoFR5Config,
)
from lerobot_robot_fairino.fairino_fr5 import _ROBOT_BY_IP

JOINTS = [f"joint{i}" for i in range(1, 7)]


def _drag_robot(ip):
    r = FairinoFR5(FairinoFR5Config(mock=True, drag_teach=True, ip=ip))
    r.connect()
    return r


# --- robot drag mode ----------------------------------------------------------------

def test_drag_mode_enters_freedrive_no_servo():
    r = _drag_robot("10.0.0.1")
    try:
        assert r.robot.drag_state == 1      # DragTeachSwitch(1) issued
        assert r._drag_active
        assert not r._servo_started         # no servo session in drag mode
        assert r._enabled                   # but the arm is enabled
    finally:
        r.disconnect()


def test_drag_mode_switches_to_manual():
    r = _drag_robot("10.0.0.10")
    try:
        assert 1 in r.robot.mode_states   # Mode(1) = manual issued for free-drive
    finally:
        r.disconnect()


def test_drag_entry_failure_gives_clear_error(monkeypatch):
    # Simulate the controller refusing free-drive (e.g. still in automatic mode).
    from lerobot_robot_fairino import _sdk

    monkeypatch.setattr(
        _sdk.MockRPC, "DragTeachSwitch",
        lambda self, state: -1 if int(state) == 1 else 0,
    )
    r = FairinoFR5(FairinoFR5Config(mock=True, drag_teach=True, ip="10.0.0.11"))
    with pytest.raises(RuntimeError, match="MANUAL"):
        r.connect()
    assert "10.0.0.11" not in _ROBOT_BY_IP   # failed connect must not leave it registered


def test_drag_mode_send_action_is_noop():
    r = _drag_robot("10.0.0.2")
    try:
        before = list(r.robot.robot_state_pkg.jt_cur_pos)
        r.send_action({f"{j}.pos": 99.0 for j in JOINTS})  # would ServoJ in normal mode
        assert list(r.robot.robot_state_pkg.jt_cur_pos) == before  # unchanged
    finally:
        r.disconnect()


def test_drag_mode_registers_and_unregisters():
    r = _drag_robot("10.0.0.3")
    assert _ROBOT_BY_IP.get("10.0.0.3") is r
    mock = r.robot
    r.disconnect()
    assert "10.0.0.3" not in _ROBOT_BY_IP
    assert mock.drag_state == 0             # exited drag on disconnect


# --- teleop -------------------------------------------------------------------------

def test_drag_teleop_registered():
    assert TeleoperatorConfig.get_choice_class("fairino_drag_teleop") is FairinoDragTeleopConfig


def test_drag_teleop_reads_robot_joints():
    r = _drag_robot("10.0.0.4")
    teleop = FairinoDragTeleop(FairinoDragTeleopConfig(ip="10.0.0.4"))
    try:
        teleop.connect()
        # simulate the operator hand-guiding the arm
        r.robot.robot_state_pkg.jt_cur_pos = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        act = teleop.get_action()
        assert set(act) == set(r.action_features)   # keys match the robot's action space
        assert act["joint1.pos"] == pytest.approx(10.0)
        assert act["joint6.pos"] == pytest.approx(60.0)
    finally:
        teleop.disconnect()
        r.disconnect()


def test_drag_teleop_requires_connected_robot():
    teleop = FairinoDragTeleop(FairinoDragTeleopConfig(ip="10.0.0.99"))
    with pytest.raises(ConnectionError):
        teleop.connect()


def test_drag_teleop_action_keys_end_in_pos():
    teleop = FairinoDragTeleop(FairinoDragTeleopConfig(ip="10.0.0.5"))
    assert all(k.endswith(".pos") for k in teleop.action_features)
