import json
from pathlib import Path

import numpy as np
import pytest

from scripts import compare_system_baselines as compare
from scripts.paper_metrics import (
    PRIMARY_COMPARE_METRICS,
    bootstrap_mean_ci,
    derive_paper_metrics,
)
from scripts.run_multiuser_scaling_suite import (
    CORE_SCALING_METRICS,
    MultiUserConfig,
    _aggregate_seed_rows,
    _aggregate_training_curves,
    _plot_fixed_user_seed_summary,
    _read_comparison_rows,
    _validate_comparison_schema,
    has_any_training_artifacts,
    has_training_artifacts,
    plot_scaling_metrics,
)
from scripts.train import (
    ENVIRONMENT_SCHEMA_VERSION,
    HANMAPPOTrainer,
    compute_model_selection_score,
    summarize_env_stats_with_load_balance,
)
from src.environment.gym_env import LEOSatelliteEnv, summarize_env_stats


def test_success_delay_metrics_only_use_successful_task_samples():
    summary = summarize_env_stats(
        {
            "total_tasks": 4,
            "completed_tasks": 3,
            "deadline_violations": 1,
            "total_delay": 104.0,
            "successful_task_delay_samples": [1.0, 2.0, 3.0],
        }
    )

    assert summary["avg_success_delay"] == pytest.approx(2.0)
    assert summary["p95_success_delay"] == pytest.approx(2.9)
    assert summary["avg_delay"] == pytest.approx(26.0)


def test_blocked_time_ratio_complements_service_availability():
    summary = summarize_env_stats(
        {
            "total_user_seconds": 200.0,
            "blocked_user_seconds": 30.0,
            "handover_interruption_seconds": 0.0,
        }
    )

    assert summary["blocked_time_ratio"] == pytest.approx(0.15)
    assert (
        summary["blocked_time_ratio"] + summary["service_availability_rate"]
    ) == pytest.approx(1.0)


def test_derived_energy_and_handover_metrics_use_paper_denominators():
    result = derive_paper_metrics(
        {
            "total_energy": 12.0,
            "completed_tasks": 3,
            "total_user_seconds": 120.0,
            "blocked_user_seconds": 24.0,
            "handover_committed": 4,
        }
    )

    assert result["energy_per_successful_task"] == pytest.approx(4.0)
    assert result["blocked_time_ratio"] == pytest.approx(0.2)
    assert result["handovers_per_user_minute"] == pytest.approx(2.0)


def test_training_accumulator_preserves_new_paper_metric_inputs():
    target = HANMAPPOTrainer._empty_env_stats()
    HANMAPPOTrainer._accumulate_env_stats(
        target,
        {
            "total_tasks": 2,
            "completed_tasks": 2,
            "successful_task_delay_samples": [1.0, 3.0],
            "jain_load_fairness_sum": 0.75,
            "jain_load_fairness_samples": 1,
        },
    )

    summary = summarize_env_stats_with_load_balance(target)

    assert summary["avg_success_delay"] == pytest.approx(2.0)
    assert summary["p95_success_delay"] == pytest.approx(2.9)
    assert summary["jain_mec_load_fairness"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    "metric_name",
    [
        "avg_success_delay",
        "p95_success_delay",
        "energy_per_successful_task",
    ],
)
def test_success_based_model_selection_rejects_zero_success(metric_name):
    zero_success_score = compute_model_selection_score(
        {
            "completed_tasks": 0,
            metric_name: 0.0,
        },
        metric_name,
    )
    successful_score = compute_model_selection_score(
        {
            "completed_tasks": 2,
            metric_name: 1.0,
            "total_energy": 2.0,
        },
        metric_name,
    )

    assert zero_success_score == float("-inf")
    assert np.isfinite(successful_score)


def test_systemwide_jain_fairness_penalizes_one_busy_node():
    fairness = LEOSatelliteEnv._jain_fairness(np.array([1.0, 0.0, 0.0, 0.0]))

    assert fairness == pytest.approx(0.25)
    assert fairness < 1.0


def test_bootstrap_ci_and_seed_aggregation_are_reproducible():
    first = bootstrap_mean_ci([1.0, 2.0, 3.0])
    second = bootstrap_mean_ci([1.0, 2.0, 3.0])
    rows = [
        {
            "num_users": "20",
            "method": "han_mappo",
            "display_name": "HAN+MAPPO",
            "seed": str(seed),
            "task_success_rate": str(value),
        }
        for seed, value in [(1, 0.8), (2, 0.9), (3, 1.0)]
    ]

    aggregated = _aggregate_seed_rows(rows)[0]

    assert first == second
    assert first[0] == pytest.approx(2.0)
    assert float(aggregated["task_success_rate"]) == pytest.approx(0.9)
    assert float(aggregated["task_success_rate_ci_low"]) <= 0.9
    assert float(aggregated["task_success_rate_ci_high"]) >= 0.9


def _write_history(path: Path, records: list[tuple[int, float]]) -> None:
    path.write_text(
        json.dumps(
            {
                "training": [
                    {"total_steps": step, "mean_reward": reward}
                    for step, reward in records
                ]
            }
        ),
        encoding="utf-8",
    )


def test_reward_curves_align_only_on_common_training_steps(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_history(first, [(10, 1.0), (20, 2.0), (30, 3.0)])
    _write_history(second, [(20, 4.0), (30, 5.0), (40, 6.0)])

    steps, means, lows, highs = _aggregate_training_curves([first, second])

    assert steps == [20.0, 30.0]
    assert means == pytest.approx([3.0, 4.0])
    assert len(lows) == len(highs) == 2


def test_new_paper_figures_reject_old_or_incomplete_schema(tmp_path):
    compare_dir = tmp_path / "u20"
    compare_dir.mkdir()
    summary_path = compare_dir / "comparison_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "environment_schema_version": 4,
                "metric_schema_version": 2,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=f"environment_schema_version={ENVIRONMENT_SCHEMA_VERSION}",
    ):
        _validate_comparison_schema(compare_dir)

    summary_path.write_text(
        json.dumps(
            {
                "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
                "metric_schema_version": 2,
                "env_config": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unified reward configuration"):
        _validate_comparison_schema(compare_dir)


def test_fixed_and_multiuser_scripts_share_primary_metric_definition():
    assert tuple(compare.PRIMARY_COMPARE_METRICS) == PRIMARY_COMPARE_METRICS


def test_fixed_plot_confidence_interval_only_uses_seed_records():
    episode_only = {
        "episode_metrics": [
            {"task_success_rate": 0.8},
            {"task_success_rate": 0.9},
        ]
    }
    seeded = {
        **episode_only,
        "seed_metrics": [
            {"task_success_rate": 0.7, "completed_tasks": 7},
            {"task_success_rate": 0.95, "completed_tasks": 9},
        ],
    }

    assert compare.paper_metric_samples(
        episode_only,
        "task_success_rate",
    ).size == 0
    assert compare.paper_metric_samples(
        seeded,
        "task_success_rate",
    ) == pytest.approx([70.0, 95.0])


def _write_complete_training_history(
    run_dir: Path,
    config: MultiUserConfig,
    *,
    total_steps: int,
    seed: int = 42,
) -> None:
    from scripts.train import TrainConfig

    defaults = TrainConfig()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "final_model.pt").write_bytes(b"final")
    history_config = {
        "total_timesteps": config.total_timesteps,
        "max_steps": config.max_steps,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "n_epochs": config.n_epochs,
        "algorithm": "mappo",
        "eval_interval": config.eval_interval,
        "eval_episodes": config.eval_episodes,
        "save_interval": config.save_interval,
        "early_stop_patience": config.early_stop_patience,
        "graph_update_interval": config.graph_update_interval,
        "best_model_metric": config.best_model_metric,
        "num_users": 20,
        "seed": seed,
        "reward_delay_weight": defaults.reward_delay_weight,
        "reward_energy_weight": defaults.reward_energy_weight,
        "reward_energy_reference_j": defaults.reward_energy_reference_j,
        "reward_interruption_weight": defaults.reward_interruption_weight,
        "reward_failed_handover_penalty": defaults.reward_failed_handover_penalty,
        "reward_load_balance_weight": config.reward_load_balance_weight,
    }
    (run_dir / "training_history.json").write_text(
        json.dumps(
            {
                "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
                "config": history_config,
                "summary": {"total_steps": total_steps},
            }
        ),
        encoding="utf-8",
    )


def test_training_reuse_requires_complete_matching_run(tmp_path):
    config = MultiUserConfig(
        run_id="test",
        total_timesteps=100,
        best_model_metric="reward",
    )
    best_only = tmp_path / "best_only"
    best_only.mkdir()
    (best_only / "best_model.pt").write_bytes(b"partial")

    assert has_any_training_artifacts(best_only)
    assert not has_training_artifacts(
        best_only,
        config=config,
        num_users=20,
        seed=42,
    )

    complete = tmp_path / "complete"
    _write_complete_training_history(
        complete,
        config,
        total_steps=100,
    )
    assert has_training_artifacts(
        complete,
        config=config,
        num_users=20,
        seed=42,
    )
    assert not has_training_artifacts(
        complete,
        config=config,
        num_users=20,
        seed=43,
    )


def test_training_reuse_rejects_incomplete_total_steps(tmp_path):
    config = MultiUserConfig(
        run_id="test",
        total_timesteps=100,
        best_model_metric="reward",
    )
    run_dir = tmp_path / "partial_steps"
    _write_complete_training_history(
        run_dir,
        config,
        total_steps=50,
    )

    assert not has_training_artifacts(
        run_dir,
        config=config,
        num_users=20,
        seed=42,
    )


def test_training_reuse_accepts_orderly_early_stopping(tmp_path):
    config = MultiUserConfig(
        run_id="test",
        total_timesteps=100,
        early_stop_patience=2,
        best_model_metric="reward",
    )
    run_dir = tmp_path / "early_stopped"
    _write_complete_training_history(
        run_dir,
        config,
        total_steps=50,
    )

    assert has_training_artifacts(
        run_dir,
        config=config,
        num_users=20,
        seed=42,
    )


def test_comparison_csv_preserves_task_counts_and_zero_success_is_undefined(
    tmp_path,
):
    summary_path = compare.save_results_csv(
        tmp_path,
        [
            {
                "method": "zero_success",
                "display_name": "Zero Success",
                "total_tasks": 10,
                "completed_tasks": 0,
                "resolved_tasks": 10,
                "deadline_violations": 10,
                "avg_success_delay": 0.0,
                "p95_success_delay": 0.0,
                "energy_per_successful_task": 0.0,
            }
        ],
    )

    rows = _read_comparison_rows(summary_path, num_users=20, seed=42)

    assert rows[0]["total_tasks"] == "10"
    assert rows[0]["completed_tasks"] == "0"
    assert rows[0]["avg_success_delay"] == ""
    assert rows[0]["p95_success_delay"] == ""
    assert rows[0]["energy_per_successful_task"] == ""


def test_zero_success_method_cannot_win_success_dependent_primary_metrics():
    methods = compare.annotate_priority_metrics(
        [
            {
                "method": "zero",
                "completed_tasks": 0,
                "avg_success_delay": 0.0,
                "p95_success_delay": 0.0,
                "energy_per_successful_task": 0.0,
            },
            {
                "method": "valid",
                "completed_tasks": 5,
                "avg_success_delay": 1.0,
                "p95_success_delay": 2.0,
                "total_energy": 10.0,
                "energy_per_successful_task": 2.0,
            },
        ],
        metric_name="reward",
    )

    assert "Successful-Task Delay" not in methods[0]["primary_metric_wins"]
    assert "P95 Successful-Task Delay" not in methods[0]["primary_metric_wins"]
    assert "Energy per Successful Task" not in methods[0]["primary_metric_wins"]


def test_fixed_user_aggregate_forwards_output_suffix(tmp_path, monkeypatch):
    captured_suffixes = []

    def fake_plot(*_args, **kwargs):
        captured_suffixes.append(kwargs.get("output_suffix"))
        return tmp_path / f"plot_{len(captured_suffixes)}.png"

    for name in (
        "plot_method_comparison",
        "plot_training_curve_vs_baselines",
        "plot_paper_dashboard",
        "plot_delay_energy_tradeoff",
        "plot_success_continuity_scatter",
        "plot_performance_radar",
    ):
        monkeypatch.setattr(compare, name, fake_plot)

    seed_rows = [
        {
            "method": "han_mappo",
            "display_name": "HAN+MAPPO",
            "is_system": "True",
        }
    ]
    aggregated_rows = [
        {
            "method": "han_mappo",
            "display_name": "HAN+MAPPO",
            "is_system": "True",
        }
    ]

    _plot_fixed_user_seed_summary(
        tmp_path,
        seed_rows,
        aggregated_rows,
        output_suffix="no_han",
    )

    assert captured_suffixes == ["no_han"] * 6


def test_fixed_and_multiuser_paper_plots_accept_seed_level_confidence_data(
    tmp_path,
):
    seed_records = [
        {
            "avg_success_delay": delay,
            "p95_success_delay": delay * 1.2,
            "task_success_rate": success,
            "deadline_violation_rate": 1.0 - success,
            "service_continuity_rate": 0.98,
            "energy_per_successful_task": 4.0,
            "completed_tasks": 8,
        }
        for delay, success in [(0.1, 0.8), (0.12, 0.85)]
    ]
    methods = [
        {
            "method": "han_mappo",
            "display_name": "HAN+MAPPO",
            "is_system": True,
            **seed_records[0],
            "seed_metrics": seed_records,
        }
    ]
    fixed_path = compare.plot_method_comparison(methods, tmp_path)

    scaling_rows = [
        {
            "num_users": str(users),
            "method": "han_mappo",
            "display_name": "HAN+MAPPO",
            "avg_success_delay": str(delay),
            "avg_success_delay_ci_low": str(delay * 0.9),
            "avg_success_delay_ci_high": str(delay * 1.1),
        }
        for users, delay in [(20, 0.1), (30, 0.2)]
    ]
    scaling_path = plot_scaling_metrics(
        scaling_rows,
        tmp_path,
        CORE_SCALING_METRICS[:1],
        "scaling.png",
    )

    assert fixed_path is not None and fixed_path.exists()
    assert scaling_path is not None and scaling_path.exists()
