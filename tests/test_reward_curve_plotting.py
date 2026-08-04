import json

from scripts import compare_system_baselines as compare
from scripts.plot_training_artifacts import (
    _training_xy,
    parse_args as parse_artifact_args,
)
from scripts.run_multiuser_scaling_suite import (
    MultiUserConfig,
    _load_training_curve,
    parse_args as parse_suite_args,
)


def test_comparison_reward_curve_prefers_raw_mean_reward():
    records = [
        {"total_steps": 10, "mean_reward": 1.0, "recent_mean_reward": 101.0},
        {"total_steps": 20, "mean_reward": 3.0, "recent_mean_reward": 103.0},
        {
            "total_steps": 30,
            "mean_reward": 99.0,
            "recent_mean_reward": 199.0,
            "partial_episode": True,
        },
    ]

    steps, rewards = compare.extract_training_reward_curve(records)

    assert steps.tolist() == [10.0, 20.0]
    assert rewards.tolist() == [1.0, 3.0]


def test_step_metric_reward_does_not_fall_back_to_recent_mean():
    reward = compare.metric_record_value(
        {"recent_mean_reward": 100.0, "reward": 7.0},
        "mean_reward",
    )

    assert reward == 7.0


def test_multiuser_curve_loader_prefers_raw_mean_reward(tmp_path):
    history = tmp_path / "training_history.json"
    history.write_text(
        json.dumps(
            {
                "training": [
                    {
                        "total_steps": 10,
                        "mean_reward": 2.0,
                        "recent_mean_reward": 102.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    steps, rewards = _load_training_curve(history)

    assert steps == [10.0]
    assert rewards == [2.0]


def test_artifact_curve_prefers_raw_mean_reward(tmp_path):
    history = tmp_path / "training_history.json"
    history.write_text(
        json.dumps(
            {
                "training": [
                    {
                        "total_steps": 10,
                        "mean_reward": 4.0,
                        "eval_mean_reward": 44.0,
                        "recent_mean_reward": 104.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    steps, rewards = _training_xy(history)

    assert steps.tolist() == [10.0]
    assert rewards.tolist() == [4.0]


def test_all_plot_window_defaults_are_three():
    assert compare.DEFAULT_PLOT_WINDOW == 3
    assert MultiUserConfig(run_id="window-test").plot_window == 3
    assert parse_suite_args(["--run-id", "window-test"]).plot_window == 3
    assert parse_artifact_args([]).plot_window == 3
