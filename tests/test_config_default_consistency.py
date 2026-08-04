import sys

import pytest

from scripts.compare_system_baselines import (
    build_default_train_config,
    build_env_config_from_train_config,
    parse_args as parse_compare_args,
)
from scripts.run_multiuser_scaling_suite import (
    MultiUserConfig,
    build_paths,
    build_train_command,
    parse_args as parse_suite_args,
)
from scripts.train import TrainConfig, parse_args
from src.environment.gym_env import EnvConfig, build_env_config


REWARD_DEFAULTS = {
    "reward_delay_weight": 0.60,
    "reward_energy_weight": 0.10,
    "reward_energy_reference_j": 1.0,
    "reward_interruption_weight": 0.30,
    "reward_failed_handover_penalty": 0.20,
    "reward_load_balance_weight": 0.05,
}

TRAINING_DEFAULTS = {
    "max_steps": 512,
    "total_timesteps": 150_000,
    "n_steps": 1024,
    "batch_size": 512,
    "learning_rate": 1e-4,
    "n_epochs": 4,
    "eval_interval": 25_000,
    "eval_episodes": 5,
    "save_interval": 50_000,
    "graph_update_interval": 1,
    "log_interval": 1,
    "best_model_metric": "reward",
}


def assert_defaults(config, expected):
    for name, value in expected.items():
        actual = getattr(config, name) if not isinstance(config, dict) else config[name]
        if isinstance(value, float):
            assert actual == pytest.approx(value), name
        else:
            assert actual == value, name


def test_environment_defaults_match_reference_training():
    config = EnvConfig()

    assert config.max_steps == TRAINING_DEFAULTS["max_steps"]
    assert_defaults(config, REWARD_DEFAULTS)


def test_training_defaults_match_reference_training():
    config = TrainConfig()

    assert_defaults(config, REWARD_DEFAULTS)
    assert_defaults(config, TRAINING_DEFAULTS)
    assert config.algorithm == "mappo"
    assert config.exp_name.startswith("han_mappo")


def test_all_environment_fields_are_copied_from_training_config():
    source = TrainConfig(
        pre_handover_rvt_sec=12.5,
        handover_min_snr_db=3.5,
        num_users=7,
    )

    config = build_env_config(source)

    assert config.pre_handover_rvt_sec == pytest.approx(12.5)
    assert config.handover_min_snr_db == pytest.approx(3.5)
    assert config.num_users == 7


def test_comparison_uses_the_same_environment_config_builder():
    source = {
        **build_default_train_config(
            objective="multi_objective",
            seed=42,
            max_steps=600,
            num_users=20,
            best_model_metric="avg_delay",
        ),
        "pre_handover_rvt_sec": 9.0,
        "handover_min_snr_db": 4.0,
        "num_users": 6,
    }

    config = build_env_config_from_train_config(
        source,
        seed=123,
        max_steps=80,
    )

    assert config.seed == 123
    assert config.max_steps == 80
    assert config.pre_handover_rvt_sec == pytest.approx(9.0)
    assert config.handover_min_snr_db == pytest.approx(4.0)
    assert config.num_users == 6


def test_training_cli_defaults_match_train_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])

    args = parse_args()

    assert_defaults(args, REWARD_DEFAULTS)
    assert_defaults(args, TRAINING_DEFAULTS)
    assert args.device == "auto"


def test_comparison_cli_defaults_match_reference_training(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["compare_system_baselines.py"])

    args = parse_compare_args()

    assert args.total_timesteps == TRAINING_DEFAULTS["total_timesteps"]
    assert args.episodes == TRAINING_DEFAULTS["eval_episodes"]
    assert args.best_model_metric == TRAINING_DEFAULTS["best_model_metric"]
    assert args.compare_ranking_metric == TRAINING_DEFAULTS["best_model_metric"]


def test_comparison_defaults_inherit_reference_training():
    config = build_default_train_config(
        objective="multi_objective",
        seed=42,
        max_steps=TRAINING_DEFAULTS["max_steps"],
        num_users=20,
        best_model_metric="reward",
    )

    assert_defaults(config, REWARD_DEFAULTS)
    assert_defaults(config, TRAINING_DEFAULTS)


@pytest.mark.parametrize(
    "config",
    [
        MultiUserConfig(run_id="config-test"),
    ],
)
def test_suite_defaults_match_reference_training(config):
    assert_defaults(config, TRAINING_DEFAULTS)


def test_suite_cli_defaults_and_train_command_propagate_reference_training(tmp_path):
    args = parse_suite_args(["--run-id", "config-test"])
    assert_defaults(args, TRAINING_DEFAULTS)

    config = MultiUserConfig(run_id="config-test")
    paths = build_paths(tmp_path, config.run_id, num_users=20)
    command = build_train_command(paths, config, num_users=20)
    assert command[command.index("--algorithm") + 1] == "mappo"
    assert command[command.index("--reward-load-balance-weight") + 1] == "0.05"

    for option, field_name in (
        ("--total_timesteps", "total_timesteps"),
        ("--max_steps", "max_steps"),
        ("--n_steps", "n_steps"),
        ("--batch_size", "batch_size"),
        ("--learning_rate", "learning_rate"),
        ("--n_epochs", "n_epochs"),
        ("--eval_interval", "eval_interval"),
        ("--eval_episodes", "eval_episodes"),
        ("--save_interval", "save_interval"),
    ):
        option_index = command.index(option)
        assert command[option_index + 1] == str(TRAINING_DEFAULTS[field_name])
