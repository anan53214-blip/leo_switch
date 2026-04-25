#!/usr/bin/env python3
"""
Cleanup helper for stale training artifacts under results/.

Default behavior mirrors the current repo policy:
- keep `results/full_train_v4`
- keep the newest delay-only run under `results/delay_only_train`
- keep `results/energy_only_train`
- keep `results/logs`
- remove older `full_train*`, smoke runs, empty logs, and stale delay-only runs
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"

TOP_LEVEL_DELETE_CANDIDATES = (
    "full_train",
    "full_train_v2",
    "full_train_v3",
    "quick_test",
    "smoke_reward_sensitivity",
    "test_han_integration",
    "profile_tmp",
)

RESULT_DIR_PATTERNS = (
    "baseline_smoke_*",
    "plot_*_smoke",
    "han_integration*",
)

PROJECT_ROOT_PATTERNS = (
    "pytest-cache-files-*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup stale training result directories.")
    parser.add_argument(
        "--keep-delay-runs",
        type=int,
        default=1,
        help="Number of newest delay-only runs to keep.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag the script only prints a plan.",
    )
    return parser.parse_args()


def ensure_within_results(path: Path) -> None:
    resolved = path.resolve()
    results_root = RESULTS_ROOT.resolve()
    if not str(resolved).startswith(str(results_root)):
        raise ValueError(f"Refusing to touch path outside results/: {resolved}")


def ensure_within_project(path: Path) -> None:
    resolved = path.resolve()
    project_root = PROJECT_ROOT.resolve()
    if not str(resolved).startswith(str(project_root)):
        raise ValueError(f"Refusing to touch path outside project: {resolved}")


def matching_paths(root: Path, patterns: Iterable[str]) -> List[Path]:
    matches: List[Path] = []
    if not root.exists():
        return matches
    for pattern in patterns:
        matches.extend(path for path in root.glob(pattern) if path.exists())
    return matches


def gather_delay_only_deletions(keep_runs: int) -> List[Path]:
    delay_root = RESULTS_ROOT / "delay_only_train"
    if not delay_root.exists():
        return []

    run_dirs = sorted(
        [path for path in delay_root.iterdir() if path.is_dir() and path.name.startswith("han_mappo_delay_only_")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return run_dirs[max(keep_runs, 0):]


def gather_cleanup_targets(keep_delay_runs: int) -> List[Path]:
    targets: List[Path] = []

    for name in TOP_LEVEL_DELETE_CANDIDATES:
        path = RESULTS_ROOT / name
        if path.exists():
            targets.append(path)

    targets.extend(gather_delay_only_deletions(keep_delay_runs))
    targets.extend(matching_paths(RESULTS_ROOT, RESULT_DIR_PATTERNS))
    targets.extend(matching_paths(PROJECT_ROOT, PROJECT_ROOT_PATTERNS))

    logs_root = RESULTS_ROOT / "logs"
    if logs_root.exists():
        targets.extend(path for path in logs_root.iterdir() if path.is_file() and path.stat().st_size == 0)
    return targets


def delete_targets(targets: Iterable[Path]) -> None:
    for path in targets:
        ensure_within_project(path)
        if path.is_relative_to(RESULTS_ROOT):
            ensure_within_results(path)
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        except PermissionError as exc:
            print(f"Skipped (permission denied): {path.relative_to(PROJECT_ROOT)} - {exc}")


def main() -> None:
    args = parse_args()
    targets = gather_cleanup_targets(args.keep_delay_runs)

    if not targets:
        print("No stale result directories matched the cleanup policy.")
        return

    print("Cleanup targets:")
    for path in targets:
        print(f"- {path.relative_to(PROJECT_ROOT)}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to delete these paths.")
        return

    delete_targets(targets)
    print("\nCleanup complete.")


if __name__ == "__main__":
    main()
