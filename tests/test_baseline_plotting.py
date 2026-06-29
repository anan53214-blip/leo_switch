import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.compare_system_baselines import (
    action_diagnostics,
    annotate_priority_metrics,
    CORE_BAR_METRICS,
    draw_metric_bar_panel,
    load_training_curve_from_path,
    reward_component_step_metrics_for_history,
    save_results_csv,
    save_results_json,
)


def test_comparison_plot_outputs_are_png_only():
    source = Path("scripts/compare_system_baselines.py").read_text(encoding="utf-8")
    saved_outputs = re.findall(r"save_figure\([^)]*?output_dir / \"([^\"]+)\"", source, flags=re.DOTALL)

    assert saved_outputs
    assert all(output.endswith(".png") for output in saved_outputs)
    assert not any(output.endswith(".pdf") for output in saved_outputs)


def test_metric_plots_do_not_show_better_direction_in_titles():
    source = Path("scripts/compare_system_baselines.py").read_text(encoding="utf-8")

    assert "Higher is better" not in source
    assert "Lower is better" not in source


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
                "total_handovers": 2,
                "total_user_seconds": 10.0,
                "handover_frequency": 0.2,
                "service_continuity_rate": 1.0,
                "task_completion_rate": 1.0,
                "task_success_rate": 1.0,
                "mec_load_fairness": 0.5,
                "total_energy": 1.0,
                "resolved_tasks": 1.0,
                "completed_tasks": 1.0,
            }
        ],
        "avg_delay",
    )

    json_path = save_results_json(tmp_path, {"methods": methods})
    csv_path = save_results_csv(tmp_path, methods)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    method = payload["methods"][0]
    assert method["handover_action_rate"] == pytest.approx(0.0)
    assert method["local_compute_rate"] == pytest.approx(0.0)
    assert method["mean_offload_ratio"] == pytest.approx(0.0)
    assert method["handover_frequency"] == pytest.approx(0.2)
    assert method["mec_load_fairness"] == pytest.approx(0.5)
    assert method["active_load_balance_score"] == pytest.approx(0.5)
    assert method["energy_per_successful_task"] == pytest.approx(1.0)
    assert "mec_activity_score" not in method
    assert "mec_load_mean" not in method

    csv_text = csv_path.read_text(encoding="utf-8")
    header = csv_text.splitlines()[0]
    assert "handover_action_rate" in header
    assert "handover_frequency" in header
    assert "mec_load_fairness" in header
    assert "energy_per_successful_task" in header
    assert "service_downtime_rate" not in header
    assert "active_load_balance_score" in header
    assert "mec_activity_score" not in header
    assert "mec_load_mean" not in header
    assert "effective_latency_score" not in header
    assert "latency_priority_score" not in header
    assert ",0.0,0.0,0.0," in csv_text


def test_han_attn_baseline_uses_compact_display_name():
    from scripts.compare_system_baselines import (
        DEFAULT_BASELINES,
        DISPLAY_NAME_MAP,
        pretty_method_name,
    )

    assert "han_attn" in DEFAULT_BASELINES
    assert DISPLAY_NAME_MAP["han_attn"] == "HAN+Attn"
    assert pretty_method_name("han_attn", is_system=False) == "HAN+Attn"


def test_han_attn_system_run_uses_han_attn_display_name():
    from scripts.compare_system_baselines import pretty_method_name

    assert (
        pretty_method_name(
            "han_attn_latency_priority_g1_300k_600s_u30_new_metrics",
            is_system=True,
        )
        == "HAN+Attn"
    )


def test_method_comparison_title_uses_system_display_name(monkeypatch, tmp_path):
    import scripts.compare_system_baselines as baseline_script

    captured = {}

    def capture_figure(fig, output_path):
        captured["title"] = fig._suptitle.get_text()
        plt.close(fig)
        return output_path

    monkeypatch.setattr(baseline_script, "save_figure", capture_figure)
    methods = [
        {
            "method": "han_attn_latency_priority_g1_300k_600s_u30_new_metrics",
            "display_name": "HAN+Attn",
            "is_system": True,
            "avg_delay": 2.7,
            "task_success_rate": 0.82,
            "deadline_violation_rate": 0.18,
            "service_continuity_rate": 0.95,
            "energy_per_successful_task": 0.31,
            "episode_metrics": [],
        }
    ]

    baseline_script.plot_method_comparison(methods, tmp_path)

    assert captured["title"] == "HAN+Attn vs. Baselines: Core Metrics"


def test_reward_curve_uses_system_display_name_for_training_label(tmp_path):
    import scripts.compare_system_baselines as baseline_script

    history_path = tmp_path / "training_history.json"
    history_path.write_text(
        json.dumps(
            {
                "training": [
                    {"total_steps": 100, "mean_reward": -10.0},
                    {"total_steps": 200, "mean_reward": -8.0},
                ],
                "evaluation": [],
            }
        ),
        encoding="utf-8",
    )
    methods = [
        {
            "method": "han_attn_latency_priority_g1_300k_600s_u30_new_metrics",
            "display_name": "HAN+Attn",
            "is_system": True,
        }
    ]

    fig, ax = plt.subplots()
    try:
        assert baseline_script.draw_reward_curve_panel(
            ax,
            history_path,
            methods,
            window=3,
        )
        _, labels = ax.get_legend_handles_labels()
        assert "HAN+Attn" in labels
        assert "HAN+MAPPO" not in labels
    finally:
        plt.close(fig)


def test_core_metric_comparison_uses_paper_kpis_without_custom_composite_score():
    metric_keys = [metric[0] for metric in CORE_BAR_METRICS]

    assert metric_keys == [
        "avg_delay",
        "task_success_rate",
        "deadline_violation_rate",
        "service_continuity_rate",
        "energy_per_successful_task",
    ]
    assert "effective_latency_score" not in metric_keys
    assert "latency_priority_score" not in metric_keys
    assert "active_load_balance_score" not in metric_keys
    assert "mec_activity_score" not in metric_keys


def test_tiny_load_balance_bar_labels_stay_inside_axes():
    methods = [
        {
            "method": "han_mappo",
            "display_name": "HAN+MAPPO",
            "is_system": True,
            "mec_load_fairness": 0.001235,
            "episode_metrics": [{"mec_load_fairness": 0.0011}, {"mec_load_fairness": 0.0013}],
        },
        {
            "method": "attn_mappo",
            "display_name": "Attn+MAPPO",
            "mec_load_fairness": 0.001097,
            "episode_metrics": [{"mec_load_fairness": 0.0010}, {"mec_load_fairness": 0.0012}],
        },
        {
            "method": "full_local",
            "display_name": "Full-Local",
            "mec_load_fairness": 0.0,
            "episode_metrics": [{"mec_load_fairness": 0.0}, {"mec_load_fairness": 0.0}],
        },
    ]

    fig, ax = plt.subplots()
    try:
        draw_metric_bar_panel(
            ax,
            methods,
            metric_key="mec_load_fairness",
            title="MEC Load Fairness",
            ylabel="MEC Load Fairness",
            compact=True,
        )

        y_top = ax.get_ylim()[1]
        assert ax.get_title() == "MEC Load Fairness"
        assert all(text.get_position()[1] <= y_top for text in ax.texts)
    finally:
        plt.close(fig)
