import json

import numpy as np

from scripts.compare_system_baselines import (
    load_training_curve_from_path,
    reward_component_step_metrics_for_history,
)


def test_reward_curve_prefers_evaluation_rewards_when_available(tmp_path):
    history_path = tmp_path / "training_history.json"
    history_path.write_text(
        json.dumps(
            {
                "training": [
                    {"total_steps": 100, "mean_reward": 1.0},
                    {"total_steps": 200, "mean_reward": 2.0},
                ],
                "evaluation": [
                    {"total_steps": 100, "eval_mean_reward": 10.0},
                    {"total_steps": 200, "eval_mean_reward": 20.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    steps, rewards, evaluation = load_training_curve_from_path(history_path)

    assert np.array_equal(steps, np.array([100.0, 200.0]))
    assert np.array_equal(rewards, np.array([10.0, 20.0]))
    assert [record["eval_mean_reward"] for record in evaluation] == [10.0, 20.0]


def test_reward_component_plot_keeps_legacy_continuity_label_for_old_artifacts(tmp_path):
    history_path = tmp_path / "training_history.json"
    history_path.write_text(
        json.dumps(
            {
                "config": {"reward_service_continuity_weight": 0.5},
                "training": [
                    {"total_steps": 100, "reward_service_continuity": 988.14},
                ],
            }
        ),
        encoding="utf-8",
    )

    specs = reward_component_step_metrics_for_history(history_path)
    service_spec = next(spec for spec in specs if spec[0] == "reward_service_continuity")

    assert service_spec[1] == "Service Continuity Reward"
    assert service_spec[2] == "Reward Term"


def test_reward_component_plot_uses_interruption_penalty_label_for_new_artifacts(tmp_path):
    history_path = tmp_path / "training_history.json"
    history_path.write_text(
        json.dumps(
            {
                "config": {
                    "reward_delay_weight": 0.25,
                    "reward_service_continuity_weight": 0.15,
                    "reward_deadline_penalty": 0.30,
                },
                "training": [
                    {"total_steps": 100, "reward_service_continuity": -0.12},
                ],
            }
        ),
        encoding="utf-8",
    )

    specs = reward_component_step_metrics_for_history(history_path)
    service_spec = next(spec for spec in specs if spec[0] == "reward_service_continuity")

    assert service_spec[1] == "Service Interruption Penalty"
    assert service_spec[2] == "Penalty Term"
