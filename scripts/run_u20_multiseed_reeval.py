"""Run u20 multi-seed re-evaluation from existing checkpoints.

This script does not retrain learned methods. It reuses:

    results/full_train_latency_priority_g1_300k_600s_u20_20260603_201842
    results/baseline_compare/g1_300k_600s_u20_20260603_201842/learned_baselines

and creates a fresh directory under results/baseline_compare:

    results/baseline_compare/multiseed_u20_reeval_<timestamp>/
        seed_42/
        seed_43/
        ...
        combined_comparison_summary.csv
        multiseed_summary.csv
        multiseed_summary.md

Example:

    python scripts/run_u20_multiseed_reeval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMPARE_ROOT = PROJECT_ROOT / "results" / "baseline_compare"
DEFAULT_SYSTEM_RUN_DIR = (
    PROJECT_ROOT
    / "results"
    / "full_train_latency_priority_g1_300k_600s_u20_20260603_201842"
)
DEFAULT_LEARNED_BASELINES_SOURCE = (
    BASELINE_COMPARE_ROOT
    / "g1_300k_600s_u20_20260603_201842"
    / "learned_baselines"
)

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_BASELINES = (
    "attn_mappo",
    "mappo_no_han",
    "random",
    "min_distance",
    "full_local",
    "joint_greedy",
)
DEFAULT_METRICS = (
    "avg_delay",
    "task_success_rate",
    "deadline_violation_rate",
    "service_continuity_rate",
    "handover_failure_rate",
    "handover_frequency",
    "energy_per_resolved_task",
    "active_load_balance_score",
    "mec_activity_score",
    "mean_offload_ratio",
)


@dataclass(frozen=True)
class MultiSeedConfig:
    suite_dir: Path
    system_run_dir: Path = DEFAULT_SYSTEM_RUN_DIR
    learned_baselines_source: Path = DEFAULT_LEARNED_BASELINES_SOURCE
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    episodes: int = 10
    max_steps: int = 600
    num_users: int = 20
    device: str = "cpu"
    python_executable: str = "python"
    best_model_metric: str = "avg_delay"
    compare_ranking_metric: str = "avg_delay"
    plot_window: int = 5
    baselines: tuple[str, ...] = DEFAULT_BASELINES
    metrics: tuple[str, ...] = DEFAULT_METRICS
    dry_run: bool = False
    overwrite: bool = False
    extra_compare_args: tuple[str, ...] = field(default_factory=tuple)


def timestamp_id(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def default_suite_dir(run_id: str | None = None) -> Path:
    return BASELINE_COMPARE_ROOT / f"multiseed_u20_reeval_{run_id or timestamp_id()}"


def parse_int_list(values: Sequence[str]) -> tuple[int, ...]:
    seeds: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                seeds.append(int(part))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return tuple(seeds)


def build_compare_command(config: MultiSeedConfig, seed: int, output_dir: Path) -> list[str]:
    command = [
        config.python_executable,
        "scripts/compare_system_baselines.py",
        "--run-mode",
        "compare_only",
        "--system-run-dir",
        str(config.system_run_dir),
        "--output-dir",
        str(output_dir),
        "--episodes",
        str(config.episodes),
        "--max-steps",
        str(config.max_steps),
        "--seed",
        str(seed),
        "--num-users",
        str(config.num_users),
        "--device",
        config.device,
        "--best-model-metric",
        config.best_model_metric,
        "--compare-ranking-metric",
        config.compare_ranking_metric,
        "--plot-window",
        str(config.plot_window),
        "--reuse-learned-checkpoints",
        "--baselines",
        *config.baselines,
        *config.extra_compare_args,
    ]
    return command


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def require_inputs(config: MultiSeedConfig) -> None:
    required_paths = [
        config.system_run_dir / "best_model.pt",
        config.system_run_dir / "training_history.json",
        config.learned_baselines_source / "attn_mappo" / "best_model.pt",
        config.learned_baselines_source / "mappo_no_han" / "best_model.pt",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        message = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required checkpoint artifacts:\n{message}")


def prepare_suite_dir(config: MultiSeedConfig) -> None:
    if config.suite_dir.exists() and any(config.suite_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(
            f"{config.suite_dir} already exists and is not empty. "
            "Use --overwrite or choose --output-dir."
        )
    config.suite_dir.mkdir(parents=True, exist_ok=True)


def ensure_learned_baselines_link(seed_dir: Path, source: Path, overwrite: bool) -> Path:
    target = seed_dir / "learned_baselines"
    if target.exists() or target.is_symlink():
        if not overwrite:
            return target
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    try:
        target.symlink_to(source.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(source, target)
    return target


def run_command(command: Sequence[str], cwd: Path, log_path: Path, dry_run: bool) -> None:
    print(format_command(command), flush=True)
    print(f"Log: {log_path}", flush=True)
    if dry_run:
        return
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(format_command(command) + "\n\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def _float_or_nan(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_seed_summary(seed_dir: Path, seed: int, metrics: Sequence[str]) -> list[dict[str, str]]:
    csv_path = seed_dir / "comparison_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing seed summary: {csv_path}")
    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output = {"seed": str(seed), "method": row.get("method", "")}
            for metric in metrics:
                output[metric] = row.get(metric, "")
            rows.append(output)
    return rows


def aggregate_seed_summaries(
    suite_dir: Path,
    seeds: Sequence[int],
    metrics: Sequence[str],
) -> tuple[Path, Path]:
    combined_rows: list[dict[str, str]] = []
    for seed in seeds:
        combined_rows.extend(read_seed_summary(suite_dir / f"seed_{seed}", seed, metrics))

    combined_csv = suite_dir / "combined_comparison_summary.csv"
    combined_fields = ["seed", "method", *metrics]
    with combined_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=combined_fields)
        writer.writeheader()
        writer.writerows(combined_rows)

    methods = sorted({row["method"] for row in combined_rows})
    summary_fields = ["method", "num_seeds"]
    for metric in metrics:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])

    summary_rows: list[dict[str, str]] = []
    for method in methods:
        method_rows = [row for row in combined_rows if row["method"] == method]
        summary: dict[str, str] = {
            "method": method,
            "num_seeds": str(len({row["seed"] for row in method_rows})),
        }
        for metric in metrics:
            values = [
                value
                for value in (_float_or_nan(row.get(metric)) for row in method_rows)
                if not math.isnan(value)
            ]
            summary[f"{metric}_mean"] = f"{mean(values):.10g}" if values else ""
            summary[f"{metric}_std"] = f"{stdev(values):.10g}" if len(values) >= 2 else "0"
        summary_rows.append(summary)

    summary_csv = suite_dir / "multiseed_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    write_markdown_summary(suite_dir / "multiseed_summary.md", summary_rows, metrics)
    return summary_csv, combined_csv


def write_markdown_summary(path: Path, rows: Sequence[dict[str, str]], metrics: Sequence[str]) -> None:
    primary_metrics = [metric for metric in DEFAULT_METRICS if metric in metrics][:6]
    headers = ["Method", *primary_metrics]
    lines = [
        "# U20 Multi-Seed Re-Evaluation Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", *[":---:" for _ in primary_metrics]]) + " |",
    ]
    for row in rows:
        cells = [row["method"]]
        for metric in primary_metrics:
            mean_value = row.get(f"{metric}_mean", "")
            std_value = row.get(f"{metric}_std", "")
            cells.append(f"{mean_value} +/- {std_value}" if mean_value else "")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(config: MultiSeedConfig) -> Path:
    manifest_path = config.suite_dir / "suite_manifest.json"
    payload = asdict(config)
    for key in ("suite_dir", "system_run_dir", "learned_baselines_source"):
        payload[key] = str(payload[key])
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def run_multiseed_reeval(config: MultiSeedConfig) -> tuple[Path, Path]:
    require_inputs(config)
    if not config.dry_run:
        prepare_suite_dir(config)
        write_manifest(config)
    else:
        print(f"Dry run suite dir: {config.suite_dir}")

    for seed in config.seeds:
        seed_dir = config.suite_dir / f"seed_{seed}"
        if not config.dry_run:
            seed_dir.mkdir(parents=True, exist_ok=True)
            ensure_learned_baselines_link(
                seed_dir,
                config.learned_baselines_source,
                overwrite=config.overwrite,
            )
        command = build_compare_command(config, seed=seed, output_dir=seed_dir)
        run_command(command, PROJECT_ROOT, seed_dir / "compare_stdout.log", config.dry_run)

    if config.dry_run:
        return config.suite_dir / "multiseed_summary.csv", config.suite_dir / "combined_comparison_summary.csv"
    return aggregate_seed_summaries(config.suite_dir, config.seeds, config.metrics)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run u20 multi-seed comparison-only re-evaluation under "
            "results/baseline_compare without retraining learned methods."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--system-run-dir", type=Path, default=DEFAULT_SYSTEM_RUN_DIR)
    parser.add_argument("--learned-baselines-source", type=Path, default=DEFAULT_LEARNED_BASELINES_SOURCE)
    parser.add_argument("--seeds", nargs="+", default=[str(seed) for seed in DEFAULT_SEEDS])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--num-users", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--python-executable", type=str, default="python")
    parser.add_argument("--best-model-metric", type=str, default="avg_delay")
    parser.add_argument("--compare-ranking-metric", type=str, default="avg_delay")
    parser.add_argument("--plot-window", type=int, default=5)
    parser.add_argument("--baselines", nargs="+", default=list(DEFAULT_BASELINES))
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--extra-compare-args", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> MultiSeedConfig:
    suite_dir = args.output_dir or default_suite_dir(args.run_id)
    if not suite_dir.is_absolute():
        suite_dir = PROJECT_ROOT / suite_dir
    return MultiSeedConfig(
        suite_dir=suite_dir,
        system_run_dir=args.system_run_dir if args.system_run_dir.is_absolute() else PROJECT_ROOT / args.system_run_dir,
        learned_baselines_source=(
            args.learned_baselines_source
            if args.learned_baselines_source.is_absolute()
            else PROJECT_ROOT / args.learned_baselines_source
        ),
        seeds=parse_int_list(args.seeds),
        episodes=args.episodes,
        max_steps=args.max_steps,
        num_users=args.num_users,
        device=args.device,
        python_executable=args.python_executable,
        best_model_metric=args.best_model_metric,
        compare_ranking_metric=args.compare_ranking_metric,
        plot_window=args.plot_window,
        baselines=tuple(args.baselines),
        metrics=tuple(args.metrics),
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        extra_compare_args=tuple(args.extra_compare_args),
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = config_from_args(parse_args(argv))
    print(f"Suite output dir: {config.suite_dir}")
    print(f"Seeds: {', '.join(str(seed) for seed in config.seeds)}")
    print(f"Episodes per seed: {config.episodes}")
    summary_csv, combined_csv = run_multiseed_reeval(config)
    print(f"Combined seed CSV: {combined_csv}")
    print(f"Multi-seed summary CSV: {summary_csv}")
    print(f"Multi-seed summary Markdown: {config.suite_dir / 'multiseed_summary.md'}")


if __name__ == "__main__":
    main()
