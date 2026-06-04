import csv
import sys
from pathlib import Path

import pytest

from scripts.run_u20_multiseed_reeval import (
    DEFAULT_BASELINES,
    DEFAULT_METRICS,
    MultiSeedConfig,
    aggregate_seed_summaries,
    build_compare_command,
    run_command,
)


def test_build_compare_command_uses_seed_output_dir_and_reused_checkpoints(tmp_path):
    config = MultiSeedConfig(
        suite_dir=tmp_path / "multiseed",
        system_run_dir=tmp_path / "system_run",
        learned_baselines_source=tmp_path / "learned_baselines",
        seeds=(43,),
        episodes=10,
        max_steps=600,
        num_users=20,
        device="cpu",
        python_executable="python",
    )

    command = build_compare_command(config, seed=43, output_dir=config.suite_dir / "seed_43")

    assert command[:3] == ["python", "scripts/compare_system_baselines.py", "--run-mode"]
    assert "compare_only" in command
    assert command[command.index("--output-dir") + 1] == str(config.suite_dir / "seed_43")
    assert command[command.index("--seed") + 1] == "43"
    assert command[command.index("--episodes") + 1] == "10"
    assert command[command.index("--num-users") + 1] == "20"
    assert "--reuse-learned-checkpoints" in command
    assert command[command.index("--baselines") + 1 :] == list(DEFAULT_BASELINES)


def test_aggregate_seed_summaries_writes_mean_std(tmp_path):
    suite_dir = tmp_path / "multiseed"
    for seed, han_delay, attn_delay in [(42, 2.6, 2.5), (43, 2.8, 2.7)]:
        seed_dir = suite_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        with (seed_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["method", "avg_delay", "task_success_rate"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "method": "han_mappo",
                    "avg_delay": han_delay,
                    "task_success_rate": 0.80,
                }
            )
            writer.writerow(
                {
                    "method": "attn_mappo",
                    "avg_delay": attn_delay,
                    "task_success_rate": 0.85,
                }
            )

    summary_csv, combined_csv = aggregate_seed_summaries(
        suite_dir=suite_dir,
        seeds=(42, 43),
        metrics=DEFAULT_METRICS,
    )

    assert summary_csv.exists()
    assert combined_csv.exists()
    rows = {row["method"]: row for row in csv.DictReader(summary_csv.open(encoding="utf-8"))}
    assert rows["han_mappo"]["num_seeds"] == "2"
    assert float(rows["han_mappo"]["avg_delay_mean"]) == pytest.approx(2.7)
    assert float(rows["han_mappo"]["avg_delay_std"]) == pytest.approx(0.1414213562)
    assert float(rows["attn_mappo"]["task_success_rate_mean"]) == pytest.approx(0.85)


def test_run_command_streams_output_to_log(tmp_path):
    log_path = tmp_path / "command.log"

    run_command(
        [
            sys.executable,
            "-c",
            "import os; print('PYTHONUNBUFFERED=' + str(os.environ.get('PYTHONUNBUFFERED')))",
        ],
        cwd=Path.cwd(),
        log_path=log_path,
        dry_run=False,
    )

    assert "PYTHONUNBUFFERED=1" in log_path.read_text(encoding="utf-8")
