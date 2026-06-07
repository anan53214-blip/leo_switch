import csv
import json
from pathlib import Path

import pytest

from scripts.run_multiuser_scaling_suite import (
    DEFAULT_BASELINES,
    METHOD_DISPLAY_NAMES,
    MultiUserConfig,
    aggregate_user_summaries,
    build_compare_command,
    build_paths,
)


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
                    "mec_activity_score",
                    "active_load_balance_score",
                    "energy_per_resolved_task",
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
                    "mec_activity_score": 0.2,
                    "active_load_balance_score": 0.3,
                    "energy_per_resolved_task": 1.1,
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
                    "mec_activity_score": 0.15,
                    "active_load_balance_score": 0.25,
                    "energy_per_resolved_task": 1.3,
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
