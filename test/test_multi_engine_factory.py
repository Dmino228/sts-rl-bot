import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine_factory import create_game_engine, normalize_game_version
from env import SlayTheSpireEnv
from sts2.process_manager import StS2CliProcessManager


def test_game_version_normalization_supports_cli_values():
    assert normalize_game_version(1) == "sts1"
    assert normalize_game_version("sts1") == "sts1"
    assert normalize_game_version(2) == "sts2"
    assert normalize_game_version("sts2") == "sts2"


def test_engine_factory_creates_separate_engine_strategies():
    sts1 = create_game_engine(1)
    sts2 = create_game_engine(2)

    assert sts1.game_version == "sts1"
    assert sts2.game_version == "sts2"
    assert sts1.create_state_encoder().shape == (205,)
    assert sts2.create_state_encoder().shape == (205,)


def test_env_defaults_to_sts1_but_can_construct_sts2_stub():
    sts1_env = SlayTheSpireEnv()
    assert sts1_env.game_version == "sts1"
    assert sts1_env.action_space.n == 100
    assert sts1_env.observation_space.shape == (205,)
    sts1_env.close()

    sts2_env = SlayTheSpireEnv(
        game_version=2,
        sts2_cli_path="fake-sts2-cli",
    )
    assert sts2_env.game_version == "sts2"
    assert isinstance(sts2_env.process_manager, StS2CliProcessManager)
    assert sts2_env.action_space.n == 100

    obs = sts2_env.state_encoder.encode({"observation": [2.0, -2.0, 0.25]})
    assert obs[:3].tolist() == [1.0, -1.0, 0.25]
    assert np.all(obs >= -1.0)
    assert np.all(obs <= 1.0)
    sts2_env.close()
