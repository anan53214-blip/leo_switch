import json

import numpy as np
import pytest

from scripts.compare_system_baselines import (
    action_diagnostics,
    load_training_curve_from_path,
    reward_component_step_metrics_for_history,
)


def test_reward_curve_uses_raw_training_rewards_and_returns_eval_markers(tmp_path):
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
    assert np.array_equal(rewards, np.array([1.0, 2.0]))
    assert [record["eval_mean_reward"] for record in evaluation] == [10.0, 20.0]


def test_action_diagnostics_reports_degenerate_local_policy():
    diagnostics = action_diagnostics(
        [
            np.array([[0.0, 0.0], [1.0, 0.2]], dtype=np.float32),
            np.array([[0.0, 0.04]], dtype=np.float32),
        ],
        min_effective_offload_ratio=0.05,
    )

    assert diagnostics["handover_action_rate"] == pytest.approx(1 / 3)
    assert diagnostics["local_compute_rate"] == pytest.approx(2 / 3)
    assert diagnostics["mean_offload_ratio"] == pytest.approx(0.08)


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
