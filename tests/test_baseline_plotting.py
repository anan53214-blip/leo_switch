import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.compare_system_baselines import (
    action_diagnostics,
    annotate_priority_metrics,
    draw_metric_bar_panel,
    load_training_curve_from_path,
    reward_component_step_metrics_for_history,
    save_results_csv,
    save_results_json,
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


def test_comparison_summary_outputs_action_diagnostics_for_legacy_methods(tmp_path):
    methods = annotate_priority_metrics(
        [
            {
                "method": "legacy_method",
                "display_name": "Legacy Method",
                "is_system": False,
                "episodes": 1,
                "mean_reward": 1.0,
                "std_reward": 0.0,
                "avg_delay": 1.0,
                "service_continuity_rate": 1.0,
                "task_completion_rate": 1.0,
                "avg_load_balance_score": 0.0,
                "total_energy": 1.0,
                "resolved_tasks": 1.0,
            }
        ],
        "latency_priority_score",
    )

    json_path = save_results_json(tmp_path, {"methods": methods})
    csv_path = save_results_csv(tmp_path, methods)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    method = payload["methods"][0]
    assert method["handover_action_rate"] == pytest.approx(0.0)
    assert method["local_compute_rate"] == pytest.approx(0.0)
    assert method["mean_offload_ratio"] == pytest.approx(0.0)

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "handover_action_rate" in csv_text.splitlines()[0]
    assert ",0.0,0.0,0.0," in csv_text


def test_tiny_load_balance_bar_labels_stay_inside_axes():
    methods = [
        {
            "method": "han_mappo",
            "display_name": "HAN+MAPPO",
            "is_system": True,
            "avg_load_balance_score": 0.001235,
            "episode_metrics": [{"avg_load_balance_score": 0.0011}, {"avg_load_balance_score": 0.0013}],
        },
        {
            "method": "attn_mappo",
            "display_name": "Attn+MAPPO",
            "avg_load_balance_score": 0.001097,
            "episode_metrics": [{"avg_load_balance_score": 0.0010}, {"avg_load_balance_score": 0.0012}],
        },
        {
            "method": "full_local",
            "display_name": "Full-Local",
            "avg_load_balance_score": 0.0,
            "episode_metrics": [{"avg_load_balance_score": 0.0}, {"avg_load_balance_score": 0.0}],
        },
    ]

    fig, ax = plt.subplots()
    try:
        draw_metric_bar_panel(
            ax,
            methods,
            metric_key="avg_load_balance_score",
            title="Avg Load Balance Score",
            ylabel="Avg Load Balance Score",
            compact=True,
        )

        y_top = ax.get_ylim()[1]
        assert all(text.get_position()[1] <= y_top for text in ax.texts)
    finally:
        plt.close(fig)
