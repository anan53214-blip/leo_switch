#!/usr/bin/env python3
"""Paired diagnostics for offload modes and handover target selection."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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
from scripts.train import ENVIRONMENT_SCHEMA_VERSION, TrainConfig  # noqa: E402


DEFAULT_SOURCE_RUN = (
    PROJECT_ROOT
    / "results"
    / "full_train_latency_priority_multiuser_u20_multiuser_single_seed_150k_20260803"
)
PAIRED_METRICS = (
    "reward",
    "task_success_rate",
    "deadline_violation_rate",
    "energy_per_successful_task",
    "active_mec_load_fairness",
    "blocked_time_ratio",
)


def write_rows(path: Path, rows: list[dict]) -> Path:
    fields = sorted(
        {key for row in rows for key in row},
        key=lambda key: (key not in {"control", "method"}, key),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def paired_method_deltas(methods: list[dict]) -> list[dict]:
    reference, *controls = methods
    rows = []
    for control in controls:
        row = {
            "reference": reference["method"],
            "control": control["method"],
        }
        for metric in PAIRED_METRICS:
            deltas = [
                float(left.get(metric, 0.0)) - float(right.get(metric, 0.0))
                for left, right in zip(
                    reference["episode_metrics"],
                    control["episode_metrics"],
                )
            ]
            row[f"{metric}_delta_mean"] = statistics.mean(deltas)
            row[f"{metric}_delta_sample_std"] = (
                statistics.stdev(deltas) if len(deltas) > 1 else 0.0
            )
        rows.append(row)
    return rows


class StayJointGreedyPolicy(JointGreedyPolicy):
    """Keep the current satellite; use joint selection only to reconnect."""

    def _handover_candidate_actions(
        self,
        env,
        user,
        visible_sats,
        legal_mask,
    ):
        if int(user.serving_satellite) >= 0:
            return [0]
        return super()._handover_candidate_actions(
            env,
            user,
            visible_sats,
            legal_mask,
        )


class ElevationJointGreedyPolicy(JointGreedyPolicy):
    """Choose the highest-elevation legal link, independent of MEC load."""

    def _handover_candidate_actions(
        self,
        env,
        user,
        visible_sats,
        legal_mask,
    ):
        legal = super()._handover_candidate_actions(
            env,
            user,
            visible_sats,
            legal_mask,
        )
        current = (
            env._get_satellite_visibility(user, int(user.serving_satellite))
            if int(user.serving_satellite) >= 0
            else None
        )

        def elevation(action):
            if action == 0:
                return (
                    float(current.elevation_deg)
                    if current is not None and current.is_visible
                    else -float("inf")
                )
            return float(visible_sats[action - 1].elevation_deg)

        return [max(legal, key=elevation)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired offload-grid or handover-selection diagnostics using "
            "the same episode seeds."
        )
    )
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--use-current-defaults",
        action="store_true",
        help=(
            "Use the current TrainConfig environment instead of the source "
            "run's historical environment configuration."
        ),
    )
    parser.add_argument(
        "--handover-ablation",
        action="store_true",
        help=(
            "Compare joint load-aware satellite selection with stay/reconnect "
            "and elevation-only selection using the full offload grid."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_run = args.source_run.resolve()
    history_path = source_run / "training_history.json"
    checkpoint_path = source_run / "best_model.pt"
    config = (
        vars(TrainConfig()).copy()
        if args.use_current_defaults
        else load_run_config(
            checkpoint_path if checkpoint_path.exists() else None,
            history_path if history_path.exists() else None,
        )
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

    full_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    policies = (
        [
            (
                "joint_load_aware",
                JointGreedyPolicy(objective=objective, offload_grid=full_grid),
            ),
            (
                "stay_or_reconnect",
                StayJointGreedyPolicy(
                    objective=objective,
                    offload_grid=full_grid,
                ),
            ),
            (
                "elevation_only",
                ElevationJointGreedyPolicy(
                    objective=objective,
                    offload_grid=full_grid,
                ),
            ),
        ]
        if args.handover_ablation
        else [
            (
                "joint_greedy_full_grid",
                JointGreedyPolicy(
                    objective=objective,
                    offload_grid=full_grid,
                ),
            ),
            (
                "joint_greedy_endpoints",
                JointGreedyPolicy(
                    objective=objective,
                    offload_grid=[0.0, 1.0],
                ),
            ),
            (
                "joint_greedy_interior",
                JointGreedyPolicy(
                    objective=objective,
                    offload_grid=[0.25, 0.5, 0.75],
                ),
            ),
        ]
    )
    methods = []
    for name, policy in policies:
        policy.name = name
        result = evaluate_policy(
            policy=policy,
            objective=objective,
            config=config,
            episodes=int(args.episodes),
            seed=seed,
            max_steps=max_steps,
            collect_task_trace=True,
            collect_handover_actionability=True,
        )
        result["offload_grid"] = policy.offload_grid
        methods.append(result)
        print(
            f"{name}: reward={result['mean_reward']:.6f}, "
            f"success={result.get('task_success_rate', 0.0):.4f}, "
            f"deadline_miss={result.get('deadline_violation_rate', 0.0):.4f}"
        )
        actionability = result["handover_actionability"]
        print(
            "  handover actionability: "
            f"gate_open={actionability['pre_handover_gate_open_rate']:.4f}, "
            "ungated_feasible="
            f"{actionability['ungated_feasible_switch_user_rate']:.4f}, "
            "blocked_relief="
            f"{actionability['gate_blocked_congestion_relief_user_rate']:.4f}"
        )

    diagnostic_paths = save_task_offload_diagnostics(output_dir, methods)
    actionability_rows = []
    for method in methods:
        row = {
            "method": method["method"],
            **{
                key: value
                for key, value in method["handover_actionability"].items()
                if key not in {
                    "gate_open_reason_rates",
                    "raw_candidate_status_rates",
                }
            },
        }
        for reason, rate in method["handover_actionability"][
            "gate_open_reason_rates"
        ].items():
            row[f"gate_reason_{reason}_rate"] = rate
        for status, rate in method["handover_actionability"][
            "raw_candidate_status_rates"
        ].items():
            row[f"candidate_status_{status}_rate"] = rate
        actionability_rows.append(row)
    actionability_path = output_dir / "handover_actionability_summary.csv"
    write_rows(actionability_path, actionability_rows)
    diagnostic_paths["handover_actionability_summary"] = actionability_path

    paired_deltas = (
        paired_method_deltas(methods)
        if args.handover_ablation
        else []
    )
    if paired_deltas:
        paired_path = write_rows(
            output_dir / "handover_paired_deltas.csv",
            paired_deltas,
        )
        diagnostic_paths["handover_paired_deltas"] = paired_path

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
        "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "source_run": str(source_run),
        "used_current_defaults": bool(args.use_current_defaults),
        "config": config,
        "objective": objective,
        "experiment": (
            "handover_ablation"
            if args.handover_ablation
            else "offload_bimodality"
        ),
        "seed": seed,
        "paired_episode_seeds": [
            seed + 1_000_000 + episode_index
            for episode_index in range(int(args.episodes))
        ],
        "episodes": int(args.episodes),
        "max_steps": max_steps,
        "methods": compact_methods,
        "paired_method_deltas": paired_deltas,
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
