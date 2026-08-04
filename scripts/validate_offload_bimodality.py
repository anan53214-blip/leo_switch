#!/usr/bin/env python3
"""Paired diagnostic experiment for the local/full-offload bimodality claim."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_system_baselines import (  # noqa: E402
    JointGreedyPolicy,
    detect_objective,
    evaluate_policy,
    load_run_config,
)
from scripts.task_offload_diagnostics import (  # noqa: E402
    build_bimodality_summary,
    flatten_method_traces,
    save_task_offload_diagnostics,
)


DEFAULT_SOURCE_RUN = (
    PROJECT_ROOT
    / "results"
    / "full_train_latency_priority_multiuser_u20_multiuser_single_seed_150k_20260803"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a full offload grid with endpoint-only and interior-only "
            "grids using paired episode seeds."
        )
    )
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_run = args.source_run.resolve()
    history_path = source_run / "training_history.json"
    checkpoint_path = source_run / "best_model.pt"
    config = load_run_config(
        checkpoint_path if checkpoint_path.exists() else None,
        history_path if history_path.exists() else None,
    )
    objective = detect_objective(config)
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    max_steps = int(
        args.max_steps
        if args.max_steps is not None
        else config.get("max_steps", 512)
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else PROJECT_ROOT
        / "results"
        / "diagnostics"
        / f"offload_bimodality_u{config.get('num_users', 20)}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    policies = [
        ("joint_greedy_full_grid", [0.0, 0.25, 0.5, 0.75, 1.0]),
        ("joint_greedy_endpoints", [0.0, 1.0]),
        ("joint_greedy_interior", [0.25, 0.5, 0.75]),
    ]
    methods = []
    for name, grid in policies:
        policy = JointGreedyPolicy(objective=objective, offload_grid=grid)
        policy.name = name
        result = evaluate_policy(
            policy=policy,
            objective=objective,
            config=config,
            episodes=int(args.episodes),
            seed=seed,
            max_steps=max_steps,
            collect_task_trace=True,
        )
        result["offload_grid"] = grid
        methods.append(result)
        print(
            f"{name}: reward={result['mean_reward']:.6f}, "
            f"success={result.get('task_success_rate', 0.0):.4f}, "
            f"deadline_miss={result.get('deadline_violation_rate', 0.0):.4f}"
        )

    diagnostic_paths = save_task_offload_diagnostics(output_dir, methods)
    trace_rows = flatten_method_traces(methods)
    bimodality = build_bimodality_summary(trace_rows)
    compact_methods = [
        {
            key: value
            for key, value in method.items()
            if key != "task_trace"
        }
        for method in methods
    ]
    report = {
        "source_run": str(source_run),
        "objective": objective,
        "seed": seed,
        "paired_episode_seeds": [
            seed + 1_000_000 + episode_index
            for episode_index in range(int(args.episodes))
        ],
        "episodes": int(args.episodes),
        "max_steps": max_steps,
        "methods": compact_methods,
        "bimodality_summary": bimodality,
        "artifacts": {
            name: str(path)
            for name, path in diagnostic_paths.items()
        },
    }
    report_path = output_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Validation report: {report_path}")


if __name__ == "__main__":
    main()
