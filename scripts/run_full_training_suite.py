#!/usr/bin/env python3
"""
Run the full HAN+MAPPO training and baseline comparison suite.

This wrapper keeps one reproducible entry point for the paper experiment:
it launches scripts/compare_system_baselines.py with the 1,200K-step setup,
stores logs under results/baseline_compare/<timestamp>, and leaves model
artifacts in the same results layout used by previous runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOTAL_TIMESTEPS = 1_200_000
DEFAULT_EXP_NAME = "han_mappo_latency_priority"
DEFAULT_SELECTION_METRIC = "effective_latency_score"


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full 1,200K-step HAN+MAPPO plus baseline training suite "
            "and generate comparison figures/logs under results/."
        )
    )
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Timestamp suffix for output folders. Defaults to current time.")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python interpreter used to run compare_system_baselines.py.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Training/evaluation device passed through to the comparison script.")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS,
                        help="Training steps for HAN+MAPPO and learned baselines.")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Evaluation episodes for each method.")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Maximum steps per episode, aligned with previous full run.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for training and baseline evaluation.")
    parser.add_argument("--num-users", type=int, default=20,
                        help="Number of users in the environment.")
    parser.add_argument("--plot-window", type=int, default=5,
                        help="Smoothing window for generated paper figures.")
    parser.add_argument("--best-model-metric", type=str, default=DEFAULT_SELECTION_METRIC,
                        help="Metric used to select best_model.pt.")
    parser.add_argument("--compare-ranking-metric", type=str, default=DEFAULT_SELECTION_METRIC,
                        help="Metric used to rank methods and select heuristic variants.")
    parser.add_argument("--exp-name", type=str, default=DEFAULT_EXP_NAME,
                        help="Experiment name written into training history.")
    parser.add_argument("--baselines", nargs="+", default=["all"],
                        help="Baselines passed through to compare_system_baselines.py.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "baseline_compare",
                        help="Root folder for comparison outputs.")
    parser.add_argument("--system-root", type=Path, default=PROJECT_ROOT / "results",
                        help="Root folder for HAN+MAPPO model artifacts.")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Early stopping patience. Default 0 keeps the full 1,200K run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write manifest and command, but do not start training.")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                        help="Additional raw args appended after -- when launching compare_system_baselines.py.")
    return parser.parse_args()


def build_compare_command(args: argparse.Namespace, output_dir: Path, system_run_dir: Path) -> List[str]:
    command = [
        args.python,
        str(PROJECT_ROOT / "scripts" / "compare_system_baselines.py"),
        "--run-mode", "train_compare",
        "--system-run-dir", str(system_run_dir),
        "--output-dir", str(output_dir),
        "--exp-name", args.exp_name,
        "--objective", "multi_objective",
        "--total-timesteps", str(int(args.total_timesteps)),
        "--episodes", str(int(args.episodes)),
        "--max-steps", str(int(args.max_steps)),
        "--seed", str(int(args.seed)),
        "--num-users", str(int(args.num_users)),
        "--device", args.device,
        "--best-model-metric", args.best_model_metric,
        "--compare-ranking-metric", args.compare_ranking_metric,
        "--plot-window", str(int(args.plot_window)),
        "--early-stop-patience", str(int(args.early_stop_patience)),
        "--baselines", *args.baselines,
    ]
    if args.extra_args:
        extra_args = list(args.extra_args)
        if extra_args and extra_args[0] == "--":
            extra_args = extra_args[1:]
        command.extend(extra_args)
    return command


def write_manifest(
    manifest_path: Path,
    args: argparse.Namespace,
    output_dir: Path,
    system_run_dir: Path,
    command: Sequence[str],
    status: str,
    return_code: int | None = None,
) -> None:
    manifest = {
        "status": status,
        "return_code": return_code,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(output_dir),
        "system_run_dir": str(system_run_dir),
        "total_timesteps": int(args.total_timesteps),
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "num_users": int(args.num_users),
        "best_model_metric": args.best_model_metric,
        "compare_ranking_metric": args.compare_ranking_metric,
        "baselines": list(args.baselines),
        "command": list(command),
        "expected_artifacts": {
            "comparison_summary": str(output_dir / "comparison_summary.json"),
            "comparison_csv": str(output_dir / "comparison_summary.csv"),
            "episode_metrics_csv": str(output_dir / "episode_metrics.csv"),
            "suite_log": str(output_dir / "logs" / "full_training_suite.log"),
            "system_training_history": str(system_run_dir / "training_history.json"),
            "system_best_model": str(system_run_dir / "best_model.pt"),
            "learned_baselines_dir": str(output_dir / "learned_baselines"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_with_log(command: Sequence[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("Command:\n")
        log_file.write(" ".join(f'"{part}"' if " " in part else part for part in command))
        log_file.write("\n\nOutput:\n")
        log_file.flush()

        process = subprocess.Popen(
            list(command),
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return int(process.wait())


def main() -> int:
    args = parse_args()
    run_timestamp = args.timestamp or timestamp_string()
    output_dir = args.output_root / run_timestamp
    system_run_dir = args.system_root / f"full_train_latency_priority_{run_timestamp}"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=False)

    command = build_compare_command(args, output_dir=output_dir, system_run_dir=system_run_dir)
    manifest_path = output_dir / "suite_manifest.json"
    command_path = output_dir / "run_command.txt"
    log_path = log_dir / "full_training_suite.log"

    command_path.write_text(
        " ".join(f'"{part}"' if " " in part else part for part in command) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        manifest_path,
        args=args,
        output_dir=output_dir,
        system_run_dir=system_run_dir,
        command=command,
        status="dry_run" if args.dry_run else "running",
    )

    print("Full training suite")
    print(f"  Output dir: {repo_relative(output_dir)}")
    print(f"  System dir: {repo_relative(system_run_dir)}")
    print(f"  Log file:   {repo_relative(log_path)}")
    print(f"  Steps:      {args.total_timesteps:,}")
    print(f"  Baselines:  {' '.join(args.baselines)}")

    if args.dry_run:
        print("Dry run only; training was not started.")
        return 0

    return_code = run_with_log(command, log_path)
    write_manifest(
        manifest_path,
        args=args,
        output_dir=output_dir,
        system_run_dir=system_run_dir,
        command=command,
        status="completed" if return_code == 0 else "failed",
        return_code=return_code,
    )
    if return_code != 0:
        print(f"Training suite failed with exit code {return_code}. See {repo_relative(log_path)}")
    else:
        print(f"Training suite completed. Summary: {repo_relative(output_dir / 'comparison_summary.json')}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
