"""Run the graph-update-1 latency-priority diagnostic suite.

This is a smaller companion to run_latency_priority_full_suite.py for testing
fresh HAN graph encodings:

    graph_update_interval = 1
    total_timesteps = 300000
    max_steps = 600
    num_users = 10

The result layout is labeled so these diagnostic runs do not blend into the
full-suite directories:

    results/full_train_latency_priority_g1_300k_600s_u10_<run_id>
    results/baseline_compare/g1_300k_600s_u10_<run_id>

By default the comparison stage is intentionally lightweight: it compares the
trained HAN+MAPPO system against Attn+MAPPO, MAPPO without HAN, and simple
rule-based baselines only. Off-policy learned baselines such as MADDPG/PDQN are
excluded unless explicitly requested with --baselines.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_LABEL = "g1_300k_600s_u10"
DEFAULT_BASELINES = (
    "mappo_no_han",
    "attn_mappo",
    "random",
    "min_distance",
    "full_local",
    "joint_greedy",
)
TRAIN_ARTIFACTS = ("training_history.json", "best_model.pt", "final_model.pt")


@dataclass(frozen=True)
class SuitePaths:
    system_run_dir: Path
    compare_output_dir: Path
    log_dir: Path


@dataclass(frozen=True)
class SuiteConfig:
    run_id: str
    python_executable: str = "python"
    exp_name: str = "han_mappo_latency_priority_g1_300k_600s_u10"
    algorithm: str = "mappo"
    seed: int = 42
    device: str = "auto"
    num_users: int = 10
    total_timesteps: int = 300_000
    max_steps: int = 600
    n_steps: int = 1024
    eval_interval: int = 50_000
    eval_episodes: int = 3
    save_interval: int = 100_000
    graph_update_interval: int = 1
    log_interval: int = 1
    best_model_metric: str = "latency_priority_score"
    compare_ranking_metric: str = "latency_priority_score"
    compare_episodes: int = 3
    plot_window: int = 5
    early_stop_patience: int = 0
    baselines: tuple[str, ...] = DEFAULT_BASELINES
    skip_system_train: bool = False
    skip_compare: bool = False
    force_system_train: bool = False
    dry_run: bool = False


def build_paths(project_root: Path, run_id: str) -> SuitePaths:
    results_dir = project_root / "results"
    return SuitePaths(
        system_run_dir=results_dir / f"full_train_latency_priority_{RUN_LABEL}_{run_id}",
        compare_output_dir=results_dir / "baseline_compare" / f"{RUN_LABEL}_{run_id}",
        log_dir=results_dir / "logs",
    )


def resolve_run_id(run_id: str | None = None, now: datetime | None = None) -> str:
    if run_id:
        return run_id
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def _path_arg(path: Path) -> str:
    return str(path)


def build_train_command(project_root: Path, paths: SuitePaths, config: SuiteConfig) -> list[str]:
    return [
        config.python_executable,
        "scripts/train.py",
        "--exp_name",
        config.exp_name,
        "--algorithm",
        config.algorithm,
        "--seed",
        str(config.seed),
        "--device",
        config.device,
        "--num_users",
        str(config.num_users),
        "--total_timesteps",
        str(config.total_timesteps),
        "--max_steps",
        str(config.max_steps),
        "--n_steps",
        str(config.n_steps),
        "--eval_interval",
        str(config.eval_interval),
        "--eval_episodes",
        str(config.eval_episodes),
        "--save_interval",
        str(config.save_interval),
        "--graph_update_interval",
        str(config.graph_update_interval),
        "--log_interval",
        str(config.log_interval),
        "--best-model-metric",
        config.best_model_metric,
        "--save_path",
        _path_arg(paths.system_run_dir),
        "--log_path",
        _path_arg(paths.log_dir),
    ]


def build_compare_command(project_root: Path, paths: SuitePaths, config: SuiteConfig) -> list[str]:
    command = [
        config.python_executable,
        "scripts/compare_system_baselines.py",
        "--run-mode",
        "compare_only",
        "--system-run-dir",
        _path_arg(paths.system_run_dir),
        "--output-dir",
        _path_arg(paths.compare_output_dir),
        "--episodes",
        str(config.compare_episodes),
        "--max-steps",
        str(config.max_steps),
        "--total-timesteps",
        str(config.total_timesteps),
        "--seed",
        str(config.seed),
        "--device",
        config.device,
        "--num-users",
        str(config.num_users),
        "--best-model-metric",
        config.best_model_metric,
        "--compare-ranking-metric",
        config.compare_ranking_metric,
        "--plot-window",
        str(config.plot_window),
        "--early-stop-patience",
        str(config.early_stop_patience),
        "--baselines",
    ]
    command.extend(config.baselines)
    return command


def has_training_artifacts(run_dir: Path) -> bool:
    return any((run_dir / filename).exists() for filename in TRAIN_ARTIFACTS)


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: Sequence[str], cwd: Path, dry_run: bool) -> None:
    print(format_command(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the graph-update-1 latency-priority HAN+MAPPO diagnostic run, "
            "then compare it with Attn+MAPPO, MAPPO(no-HAN), and simple rule-based baselines."
        )
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Result timestamp id. Defaults to the script start time, formatted as YYYYMMDD_HHMMSS.",
    )
    parser.add_argument("--python-executable", type=str, default="python")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-users", type=int, default=10)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--save-interval", type=int, default=100_000)
    parser.add_argument(
        "--graph-update-interval",
        "--graph_update_interval",
        dest="graph_update_interval",
        type=int,
        default=1,
    )
    parser.add_argument("--compare-episodes", type=int, default=3)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(DEFAULT_BASELINES),
        help=(
            "Baselines for comparison. Defaults to Attn+MAPPO, MAPPO(no-HAN), "
            "and simple rule-based baselines; pass 'all' only when you also "
            "want learned MADDPG/PDQN-style baselines."
        ),
    )
    parser.add_argument("--best-model-metric", type=str, default="latency_priority_score")
    parser.add_argument("--compare-ranking-metric", type=str, default="latency_priority_score")
    parser.add_argument("--plot-window", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument(
        "--force-system-train",
        action="store_true",
        help="Run scripts/train.py even when the target system run directory already has artifacts.",
    )
    parser.add_argument("--skip-system-train", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> SuiteConfig:
    return SuiteConfig(
        run_id=resolve_run_id(args.run_id),
        python_executable=args.python_executable,
        seed=args.seed,
        device=args.device,
        num_users=args.num_users,
        total_timesteps=args.total_timesteps,
        max_steps=args.max_steps,
        n_steps=args.n_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_interval=args.save_interval,
        graph_update_interval=args.graph_update_interval,
        best_model_metric=args.best_model_metric,
        compare_ranking_metric=args.compare_ranking_metric,
        compare_episodes=args.compare_episodes,
        plot_window=args.plot_window,
        early_stop_patience=args.early_stop_patience,
        baselines=tuple(args.baselines),
        skip_system_train=args.skip_system_train,
        skip_compare=args.skip_compare,
        force_system_train=args.force_system_train,
        dry_run=args.dry_run,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    paths = build_paths(PROJECT_ROOT, config.run_id)

    if not config.dry_run:
        paths.log_dir.mkdir(parents=True, exist_ok=True)
        if not config.skip_compare:
            paths.compare_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run label: {RUN_LABEL}")
    print(f"Run id: {config.run_id}")
    print(f"System run dir: {paths.system_run_dir}")
    print(f"Baseline compare dir: {paths.compare_output_dir}")

    train_needed = not config.skip_system_train
    if train_needed and has_training_artifacts(paths.system_run_dir) and not config.force_system_train:
        train_needed = False
        print(
            "Existing system training artifacts found; reusing them for comparison. "
            "Use --force-system-train to train into this directory again."
        )

    if train_needed:
        if not config.dry_run:
            paths.system_run_dir.mkdir(parents=True, exist_ok=True)
        run_command(build_train_command(PROJECT_ROOT, paths, config), PROJECT_ROOT, config.dry_run)
    else:
        print("System training step skipped.")

    if not config.skip_compare:
        if not config.dry_run and not has_training_artifacts(paths.system_run_dir):
            raise FileNotFoundError(
                f"No system training artifacts found in {paths.system_run_dir}. "
                "Run without --skip-system-train first, or point --run-id to an existing run."
            )
        run_command(build_compare_command(PROJECT_ROOT, paths, config), PROJECT_ROOT, config.dry_run)
    else:
        print("Baseline comparison step skipped.")


if __name__ == "__main__":
    main()
