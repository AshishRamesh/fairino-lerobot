"""The plugin registers with LeRobot and its config resolves to the driver class."""

from lerobot.robots.config import RobotConfig
from lerobot.utils.import_utils import make_device_from_device_class

from lerobot_robot_fairino import FairinoFR5, FairinoFR5Config


def test_choice_string_registered():
    # draccus ChoiceRegistry should know the "fairino_fr5" subclass after import.
    cfg = RobotConfig.get_choice_class("fairino_fr5")
    assert cfg is FairinoFR5Config


def test_config_builds_with_defaults():
    cfg = FairinoFR5Config()
    assert cfg.ip == "192.168.58.2"
    assert cfg.fps == 30
    assert cfg.type == "fairino_fr5"


def test_make_device_resolves_driver():
    # The factory strips "Config" -> "FairinoFR5" and imports it from the package.
    robot = make_device_from_device_class(FairinoFR5Config(mock=True))
    assert isinstance(robot, FairinoFR5)
