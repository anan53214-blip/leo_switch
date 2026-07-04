import json
from pathlib import Path

import pytest

from scripts import plot_training_artifacts as plotter


def write_history(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parse_history_spec_accepts_label_assignment(tmp_path):
    history_path = tmp_path / "run_a" / "training_history.json"

    label, path = plotter.parse_history_spec(f"HAN+MAPPO={history_path}")

    assert label == "HAN+MAPPO"
    assert path == history_path


def test_method_from_history_uses_best_evaluation_record_by_selection_metric(tmp_path):
    history_path = write_history(
        tmp_path / "han" / "training_history.json",
        {
            "config": {
                "exp_name": "han_mappo_latency_priority",
                "best_model_metric": "avg_delay",
                "eval_episodes": 3,
            },
            "training": [
                {"total_steps": 100, "mean_reward": -20.0, "avg_delay": 3.0},
                {"total_steps": 200, "mean_reward": -12.0, "avg_delay": 2.2},
            ],
            "evaluation": [
                {
                    "total_steps": 100,
                    "eval_mean_reward": -18.0,
                    "eval_std_reward": 1.0,
                    "avg_delay": 3.1,
                    "task_completion_rate": 0.7,
                    "service_continuity_rate": 0.8,
                    "total_energy": 9.0,
                },
                {
                    "total_steps": 200,
                    "eval_mean_reward": -14.0,
                    "eval_std_reward": 0.5,
                    "avg_delay": 2.0,
                    "task_completion_rate": 0.9,
                    "service_continuity_rate": 0.95,
                    "total_energy": 8.0,
                },
            ],
        },
    )

    method = plotter.method_from_history(history_path, label="System", is_system=True)

    assert method["display_name"] == "System"
    assert method["is_system"] is True
    assert method["mean_reward"] == pytest.approx(-14.0)
    assert method["avg_delay"] == pytest.approx(2.0)
    assert method["task_completion_rate"] == pytest.approx(0.9)
    assert method["training_history"] == str(history_path)
    assert method["source"] == "training_history_evaluation_best_avg_delay"


def test_method_from_history_falls_back_to_training_records_without_eval(tmp_path):
    history_path = write_history(
        tmp_path / "mappo" / "training_history.json",
        {
            "config": {"exp_name": "mappo_no_han", "best_model_metric": "reward"},
            "training": [
                {
                    "total_steps": 100,
                    "mean_reward": -30.0,
                    "avg_delay": 4.0,
                    "task_completion_rate": 0.3,
                    "service_continuity_rate": 0.5,
                },
                {
                    "total_steps": 200,
                    "mean_reward": -10.0,
                    "avg_delay": 2.5,
                    "task_completion_rate": 0.8,
                    "service_continuity_rate": 0.9,
                },
            ],
        },
    )

    method = plotter.method_from_history(history_path, label=None, is_system=False)

    assert method["display_name"] == "MAPPO"
    assert method["is_system"] is False
    assert method["mean_reward"] == pytest.approx(-10.0)
    assert method["avg_delay"] == pytest.approx(2.5)
    assert method["source"] == "training_history_training_best_reward"


def test_generate_from_histories_writes_summary_and_plot_manifest(tmp_path, monkeypatch):
    system_history = write_history(
        tmp_path / "system" / "training_history.json",
        {
            "config": {"exp_name": "han_mappo", "best_model_metric": "reward"},
            "training": [
                {
                    "total_steps": 100,
                    "mean_reward": 1.0,
                    "avg_delay": 1.0,
                    "task_completion_rate": 0.8,
                    "service_continuity_rate": 0.9,
                }
            ],
        },
    )
    baseline_history = write_history(
        tmp_path / "baseline" / "training_history.json",
        {
            "config": {"exp_name": "mappo_no_han", "best_model_metric": "reward"},
            "training": [
                {
                    "total_steps": 100,
                    "mean_reward": 0.5,
                    "avg_delay": 1.2,
                    "task_completion_rate": 0.7,
                    "service_continuity_rate": 0.85,
                }
            ],
        },
    )
    output_dir = tmp_path / "figures"

    called = []

    def fake_plot(name):
        def _plot(*_args, **_kwargs):
            path = output_dir / f"{name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
            called.append(name)
            return path

        return _plot

    monkeypatch.setattr(plotter, "plot_method_comparison", fake_plot("method_comparison"))
    monkeypatch.setattr(plotter, "plot_training_curve_vs_baselines", fake_plot("reward_curve_vs_baselines"))
    monkeypatch.setattr(plotter, "plot_delay_energy_tradeoff", fake_plot("delay_energy_tradeoff"))
    monkeypatch.setattr(plotter, "plot_success_continuity_scatter", fake_plot("success_continuity_tradeoff"))
    monkeypatch.setattr(plotter, "plot_performance_radar", fake_plot("performance_radar"))
    monkeypatch.setattr(plotter, "plot_paper_dashboard", fake_plot("paper_baseline_dashboard"))
    monkeypatch.setattr(plotter, "plot_step_metric_curves", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(plotter, "plot_additional_metric_curves", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plotter, "plot_reward_distribution", lambda *_args, **_kwargs: None)

    generated = plotter.generate_from_histories(
        histories=[("HAN+MAPPO", system_history), ("MAPPO", baseline_history)],
        output_dir=output_dir,
        system_history=system_history,
        selection_metric="reward",
        plot_window=3,
    )

    summary = json.loads((output_dir / "comparison_summary.json").read_text(encoding="utf-8"))
    assert [method["display_name"] for method in summary["methods"]] == ["HAN+MAPPO", "MAPPO"]
    assert (output_dir / "comparison_summary.csv").exists()
    assert set(called) == {
        "method_comparison",
        "reward_curve_vs_baselines",
        "delay_energy_tradeoff",
        "success_continuity_tradeoff",
        "performance_radar",
        "paper_baseline_dashboard",
    }
    assert output_dir / "plot_manifest.json" in generated


def test_generate_from_histories_creates_real_plot_files(tmp_path):
    system_history = write_history(
        tmp_path / "system_real" / "training_history.json",
        {
            "config": {"exp_name": "han_mappo", "best_model_metric": "reward"},
            "training": [
                {
                    "total_steps": 100,
                    "mean_reward": 1.0,
                    "avg_delay": 1.4,
                    "task_completion_rate": 0.65,
                    "service_continuity_rate": 0.75,
                    "mec_load_fairness": 0.55,
                    "total_energy": 3.5,
                    "completed_tasks": 5,
                },
                {
                    "total_steps": 200,
                    "mean_reward": 2.0,
                    "avg_delay": 1.1,
                    "task_completion_rate": 0.82,
                    "service_continuity_rate": 0.88,
                    "mec_load_fairness": 0.68,
                    "total_energy": 3.0,
                    "completed_tasks": 6,
                },
            ],
        },
    )
    baseline_history = write_history(
        tmp_path / "baseline_real" / "training_history.json",
        {
            "config": {"exp_name": "mappo_no_han", "best_model_metric": "reward"},
            "training": [
                {
                    "total_steps": 100,
                    "mean_reward": 0.7,
                    "avg_delay": 1.8,
                    "task_completion_rate": 0.55,
                    "service_continuity_rate": 0.7,
                    "mec_load_fairness": 0.45,
                    "total_energy": 4.0,
                    "completed_tasks": 4,
                },
                {
                    "total_steps": 200,
                    "mean_reward": 1.1,
                    "avg_delay": 1.5,
                    "task_completion_rate": 0.7,
                    "service_continuity_rate": 0.78,
                    "mec_load_fairness": 0.5,
                    "total_energy": 3.7,
                    "completed_tasks": 5,
                },
            ],
        },
    )
    output_dir = tmp_path / "real_figures"

    plotter.generate_from_histories(
        histories=[("HAN+MAPPO", system_history), ("MAPPO", baseline_history)],
        output_dir=output_dir,
        selection_metric="reward",
        plot_window=2,
    )

    for filename in [
        "comparison_summary.json",
        "comparison_summary.csv",
        "method_comparison.png",
        "reward_curve_vs_baselines.png",
        "training_qos_metrics_vs_steps.png",
        "reward_components_vs_steps.png",
        "delay_energy_tradeoff.png",
        "success_continuity_tradeoff.png",
        "performance_radar.png",
        "paper_baseline_dashboard.png",
        "plot_manifest.json",
    ]:
        path = output_dir / filename
        assert path.exists()
        assert path.stat().st_size > 0
