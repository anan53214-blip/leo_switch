import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import run_multiuser_scaling_suite as multiuser
from scripts.run_multiuser_scaling_suite import (
    DEFAULT_BASELINES,
    DEFAULT_PYTHON,
    DEFAULT_USER_COUNTS,
    LEARNED_REWARD_METHODS,
    METHOD_DISPLAY_NAMES,
    MultiUserConfig,
    aggregate_user_summaries,
    build_compare_command,
    build_paths,
)


def test_default_python_executable_uses_current_interpreter_without_personal_path():
    source = Path("scripts/run_multiuser_scaling_suite.py").read_text(encoding="utf-8")

    assert r"C:\Users\19704" not in source
    assert DEFAULT_PYTHON == sys.executable
    assert MultiUserConfig(run_id="demo").python_executable == sys.executable


def test_build_paths_keeps_each_user_count_isolated(tmp_path):
    paths = build_paths(project_root=tmp_path, run_id="demo", num_users=30)

    assert paths.system_run_dir == tmp_path / "results" / "full_train_latency_priority_multiuser_u30_demo"
    assert paths.compare_output_dir == tmp_path / "results" / "baseline_compare" / "multiuser_scaling_demo" / "u30"
    assert paths.suite_dir == tmp_path / "results" / "baseline_compare" / "multiuser_scaling_demo"


def test_build_compare_command_uses_requested_algorithms_and_short_display_names(tmp_path):
    config = MultiUserConfig(
        run_id="demo",
        python_executable="python",
        device="cpu",
        total_timesteps=123,
        max_steps=45,
        compare_episodes=6,
        baselines=DEFAULT_BASELINES,
    )
    paths = build_paths(project_root=tmp_path, run_id=config.run_id, num_users=40)

    command = build_compare_command(paths, config, num_users=40)

    assert command[:3] == ["python", "scripts/compare_system_baselines.py", "--run-mode"]
    assert command[command.index("--num-users") + 1] == "40"
    assert command[command.index("--episodes") + 1] == "6"
    assert command[command.index("--max-steps") + 1] == "45"
    assert command[command.index("--total-timesteps") + 1] == "123"
    assert command[command.index("--baselines") + 1 :] == list(DEFAULT_BASELINES)
    assert METHOD_DISPLAY_NAMES["mappo_no_han"] == "MAPPO"
    assert METHOD_DISPLAY_NAMES["maddpg"] == "MADDPG"


def test_default_suite_includes_intermediate_user_counts_and_han_offpolicy_methods():
    assert DEFAULT_USER_COUNTS == (20, 25, 30, 35, 40)
    assert "han_maddpg" in DEFAULT_BASELINES
    assert "han_pdqn" in DEFAULT_BASELINES
    assert METHOD_DISPLAY_NAMES["han_maddpg"] == "HAN+MADDPG"
    assert METHOD_DISPLAY_NAMES["han_pdqn"] == "HAN+PDQN"
    assert "han_maddpg" in LEARNED_REWARD_METHODS
    assert "han_pdqn" in LEARNED_REWARD_METHODS


def test_aggregate_only_rebuilds_suite_artifacts_without_running_user_counts(tmp_path, monkeypatch):
    suite_dir = tmp_path / "results" / "baseline_compare" / "multiuser_scaling_demo"
    for num_users in (20, 25):
        user_dir = suite_dir / f"u{num_users}"
        user_dir.mkdir(parents=True)
        with (user_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "method",
                    "display_name",
                    "avg_delay",
                    "task_success_rate",
                    "deadline_violation_rate",
                    "service_continuity_rate",
                    "mec_load_fairness",
                    "energy_per_successful_task",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "method": "han_maddpg",
                    "display_name": "HAN+MADDPG",
                    "avg_delay": 2.0 + num_users / 100,
                    "task_success_rate": 0.9,
                    "deadline_violation_rate": 0.1,
                    "service_continuity_rate": 0.95,
                    "mec_load_fairness": 0.3,
                    "energy_per_successful_task": 1.1,
                }
            )

    args = Namespace(
        run_id="demo",
        user_counts=[20, 25],
        python_executable="python",
        seed=42,
        device="cpu",
        total_timesteps=300_000,
        max_steps=600,
        n_steps=1024,
        eval_interval=50_000,
        eval_episodes=3,
        save_interval=100_000,
        graph_update_interval=1,
        compare_episodes=3,
        plot_window=5,
        early_stop_patience=0,
        best_model_metric="avg_delay",
        compare_ranking_metric="avg_delay",
        baselines=list(DEFAULT_BASELINES),
        force_system_train=False,
        skip_system_train=False,
        skip_compare=False,
        reuse_learned_checkpoints=False,
        dry_run=False,
        aggregate_only=True,
    )
    monkeypatch.setattr(multiuser, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(multiuser, "parse_args", lambda: args)

    def fail_run_user_count(*_args, **_kwargs):
        raise AssertionError("aggregate-only should not run per-user training or comparison")

    monkeypatch.setattr(multiuser, "run_user_count", fail_run_user_count)

    multiuser.main()

    summary_path = suite_dir / "multiuser_summary.csv"
    assert summary_path.exists()
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    assert [row["num_users"] for row in rows] == ["20", "25"]
    assert {row["display_name"] for row in rows} == {"HAN+MADDPG"}
    assert (suite_dir / "multiuser_core_metrics.png").exists()
    assert (suite_dir / "multiuser_resource_metrics.png").exists()
    manifest = json.loads((suite_dir / "suite_manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["aggregate_only"] is True


def test_aggregate_user_summaries_writes_scaling_csv_and_uses_short_names(tmp_path):
    suite_dir = tmp_path / "suite"
    for num_users, han_delay, mappo_delay in [(20, 2.4, 2.8), (30, 2.9, 3.2)]:
        user_dir = suite_dir / f"u{num_users}"
        user_dir.mkdir(parents=True)
        with (user_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "method",
                    "display_name",
                    "avg_delay",
                    "task_success_rate",
                    "deadline_violation_rate",
                    "service_continuity_rate",
                    "mec_load_fairness",
                    "energy_per_successful_task",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "method": "han_mappo",
                    "display_name": "HAN+MAPPO",
                    "avg_delay": han_delay,
                    "task_success_rate": 0.9,
                    "deadline_violation_rate": 0.1,
                    "service_continuity_rate": 0.95,
                    "mec_load_fairness": 0.3,
                    "energy_per_successful_task": 1.1,
                }
            )
            writer.writerow(
                {
                    "method": "mappo_no_han",
                    "display_name": "MAPPO (no HAN)",
                    "avg_delay": mappo_delay,
                    "task_success_rate": 0.85,
                    "deadline_violation_rate": 0.15,
                    "service_continuity_rate": 0.9,
                    "mec_load_fairness": 0.25,
                    "energy_per_successful_task": 1.3,
                }
            )
        (user_dir / "comparison_summary.json").write_text(
            json.dumps({"methods": []}),
            encoding="utf-8",
        )

    rows, csv_path = aggregate_user_summaries(suite_dir=suite_dir, user_counts=(20, 30))

    assert csv_path == suite_dir / "multiuser_summary.csv"
    assert csv_path.exists()
    assert len(rows) == 4
    mappo_rows = [row for row in rows if row["method"] == "mappo_no_han"]
    assert {row["display_name"] for row in mappo_rows} == {"MAPPO"}
    assert float(mappo_rows[0]["avg_delay"]) == pytest.approx(2.8)
