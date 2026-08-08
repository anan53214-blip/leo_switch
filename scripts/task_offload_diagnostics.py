#!/usr/bin/env python3
"""Task-level summaries and figures for offloading-mode diagnostics."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OFFLOAD_BIN_ORDER = [
    "local",
    "very_low",
    "medium_low",
    "medium_high",
    "near_full",
    "full",
]

OFFLOAD_BIN_LABELS = {
    "local": "Local [0,.05)",
    "very_low": "Very low [.05,.25)",
    "medium_low": "Medium-low [.25,.50)",
    "medium_high": "Medium-high [.50,.75)",
    "near_full": "Near-full [.75,.95)",
    "full": "Full [.95,1]",
}


def offload_ratio_bin(value: object) -> str:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return "undecided"
    if not math.isfinite(ratio):
        return "undecided"
    if ratio < 0.05:
        return "local"
    if ratio < 0.25:
        return "very_low"
    if ratio < 0.50:
        return "medium_low"
    if ratio < 0.75:
        return "medium_high"
    if ratio < 0.95:
        return "near_full"
    return "full"


def _finite(values: Iterable[object]) -> np.ndarray:
    parsed = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            parsed.append(number)
    return np.asarray(parsed, dtype=np.float64)


def _mean(values: Iterable[object]) -> float:
    samples = _finite(values)
    return float(np.mean(samples)) if len(samples) else 0.0


def _percentile(values: Iterable[object], percentile: float) -> float:
    samples = _finite(values)
    return float(np.percentile(samples, percentile)) if len(samples) else 0.0


def flatten_method_traces(methods: Sequence[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for method in methods:
        method_name = str(method.get("method", ""))
        display_name = str(method.get("display_name", method_name))
        for source in method.get("task_trace", []):
            row = dict(source)
            row["method"] = method_name
            row["display_name"] = display_name
            row["offload_bin"] = offload_ratio_bin(
                row.get("actual_offload_ratio")
            )
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_task_type_offload_summary(rows: Sequence[Dict]) -> List[Dict]:
    decided = [row for row in rows if bool(row.get("decision_made"))]
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    totals = Counter(
        (str(row.get("method", "")), str(row.get("task_type", "")))
        for row in decided
    )
    for row in decided:
        grouped[
            (
                str(row.get("method", "")),
                str(row.get("task_type", "")),
                str(row.get("offload_bin", "undecided")),
            )
        ].append(row)

    summary: List[Dict] = []
    for (method, task_type, ratio_bin), group in sorted(grouped.items()):
        attempts = [row for row in group if row.get("mec_admission_attempted")]
        rejections = [row for row in attempts if row.get("mec_admission_rejected")]
        settled = [
            row for row in group
            if str(row.get("outcome", "")).startswith(("completed", "deadline_miss", "forced"))
        ]
        misses = [row for row in settled if not bool(row.get("success"))]
        summary.append({
            "method": method,
            "task_type": task_type,
            "offload_bin": ratio_bin,
            "tasks": len(group),
            "share_within_task_type": len(group) / max(totals[(method, task_type)], 1),
            "settled_tasks": len(settled),
            "success_rate": (
                sum(bool(row.get("success")) for row in settled) / max(len(settled), 1)
            ),
            "deadline_miss_rate": len(misses) / max(len(settled), 1),
            "mec_admission_attempts": len(attempts),
            "mec_rejections": len(rejections),
            "mec_rejection_rate": len(rejections) / max(len(attempts), 1),
            "mean_mec_queue_wait_sec": _mean(
                row.get("mec_queue_wait_sec") for row in settled
            ),
            "p95_mec_queue_wait_sec": _percentile(
                (row.get("mec_queue_wait_sec") for row in settled),
                95,
            ),
            "mean_total_delay_sec": _mean(
                row.get("total_delay_sec") for row in settled
            ),
            "p95_total_delay_sec": _percentile(
                (row.get("total_delay_sec") for row in settled),
                95,
            ),
            "mean_energy_j": _mean(row.get("total_energy_j") for row in settled),
            "mean_local_energy_j": _mean(
                row.get("local_energy_j") for row in settled
            ),
            "mean_upload_energy_j": _mean(
                row.get("upload_energy_j") for row in settled
            ),
            "mean_uplink_bandwidth_mhz": _mean(
                row.get("uplink_bandwidth_mhz")
                for row in attempts
                if row.get("mec_admission_accepted")
            ),
            "mean_uplink_concurrent_users": _mean(
                row.get("uplink_concurrent_users")
                for row in attempts
                if row.get("mec_admission_accepted")
            ),
            "mean_task_reward": _mean(row.get("task_reward") for row in settled),
        })
    return summary


def build_bimodality_summary(rows: Sequence[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        if bool(row.get("decision_made")):
            grouped[str(row.get("method", ""))].append(row)

    result: List[Dict] = []
    for method, group in sorted(grouped.items()):
        settled = [
            row for row in group
            if str(row.get("outcome", "")) not in {
                "episode_pending_mec",
                "episode_pending_user_queue",
            }
        ]
        endpoint = [
            row for row in group
            if row.get("offload_bin") in {"local", "full"}
        ]
        broad_endpoint = [
            row for row in group
            if row.get("offload_bin") in {"local", "near_full", "full"}
        ]
        middle = [
            row for row in group
            if row.get("offload_bin") in {"medium_low", "medium_high"}
        ]
        attempts = [row for row in group if row.get("mec_admission_attempted")]
        rejections = [row for row in attempts if row.get("mec_admission_rejected")]
        result.append({
            "method": method,
            "decided_tasks": len(group),
            "settled_tasks": len(settled),
            "local_share": sum(row.get("offload_bin") == "local" for row in group) / max(len(group), 1),
            "full_share": sum(row.get("offload_bin") == "full" for row in group) / max(len(group), 1),
            "strict_endpoint_share": len(endpoint) / max(len(group), 1),
            "broad_endpoint_share": len(broad_endpoint) / max(len(group), 1),
            "middle_share": len(middle) / max(len(group), 1),
            "success_rate": sum(bool(row.get("success")) for row in settled) / max(len(settled), 1),
            "mean_task_reward": _mean(row.get("task_reward") for row in settled),
            "mean_energy_j": _mean(row.get("total_energy_j") for row in settled),
            "mean_local_energy_j": _mean(
                row.get("local_energy_j") for row in settled
            ),
            "mean_upload_energy_j": _mean(
                row.get("upload_energy_j") for row in settled
            ),
            "mean_uplink_bandwidth_mhz": _mean(
                row.get("uplink_bandwidth_mhz")
                for row in attempts
                if row.get("mec_admission_accepted")
            ),
            "mean_uplink_concurrent_users": _mean(
                row.get("uplink_concurrent_users")
                for row in attempts
                if row.get("mec_admission_accepted")
            ),
            "mec_rejection_rate": len(rejections) / max(len(attempts), 1),
        })
    return result


def build_deadline_reason_summary(rows: Sequence[Dict]) -> List[Dict]:
    misses = [
        row for row in rows
        if row.get("deadline_miss_reason")
        and row.get("deadline_miss_reason") != "episode_pending"
    ]
    counts = Counter(
        (
            str(row.get("method", "")),
            str(row.get("task_type", "")),
            str(row.get("deadline_miss_reason", "")),
        )
        for row in misses
    )
    totals = Counter(str(row.get("method", "")) for row in misses)
    return [
        {
            "method": method,
            "task_type": task_type,
            "deadline_miss_reason": reason,
            "count": count,
            "share_within_method_misses": count / max(totals[method], 1),
        }
        for (method, task_type, reason), count in sorted(counts.items())
    ]


def plot_offload_distribution(rows: Sequence[Dict], output_path: Path) -> Path | None:
    decided = [row for row in rows if bool(row.get("decision_made"))]
    methods = sorted({str(row.get("method", "")) for row in decided})
    task_types = ["light", "medium", "heavy"]
    if not methods:
        return None

    fig, axes = plt.subplots(
        len(methods),
        1,
        figsize=(10, max(3.0, 2.8 * len(methods))),
        squeeze=False,
    )
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(OFFLOAD_BIN_ORDER)))
    for method_index, method in enumerate(methods):
        ax = axes[method_index, 0]
        bottoms = np.zeros(len(task_types), dtype=float)
        for ratio_bin, color in zip(OFFLOAD_BIN_ORDER, colors):
            shares = []
            for task_type in task_types:
                group = [
                    row for row in decided
                    if row.get("method") == method and row.get("task_type") == task_type
                ]
                count = sum(row.get("offload_bin") == ratio_bin for row in group)
                shares.append(count / max(len(group), 1))
            values = np.asarray(shares)
            ax.bar(
                task_types,
                values,
                bottom=bottoms,
                label=OFFLOAD_BIN_LABELS[ratio_bin],
                color=color,
            )
            bottoms += values
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Task share")
        ax.set_title(method)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    axes[0, 0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.34),
        ncol=3,
        fontsize=8,
    )
    axes[-1, 0].set_xlabel("Task type")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_deadline_reasons(rows: Sequence[Dict], output_path: Path) -> Path | None:
    summary = build_deadline_reason_summary(rows)
    methods = sorted({row["method"] for row in summary})
    reasons = sorted({row["deadline_miss_reason"] for row in summary})
    if not methods or not reasons:
        return None
    matrix = np.zeros((len(methods), len(reasons)), dtype=float)
    for row in summary:
        matrix[methods.index(row["method"]), reasons.index(row["deadline_miss_reason"])] += int(row["count"])
    shares = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)

    fig, ax = plt.subplots(figsize=(max(8, len(reasons) * 1.3), max(3.5, len(methods) * 0.8)))
    image = ax.imshow(shares, cmap="magma", vmin=0.0, vmax=max(float(np.max(shares)), 1e-9), aspect="auto")
    ax.set_xticks(range(len(reasons)), reasons, rotation=35, ha="right")
    ax.set_yticks(range(len(methods)), methods)
    ax.set_xlabel("Observed dominant miss stage")
    ax.set_ylabel("Method")
    for row_index in range(len(methods)):
        for column_index in range(len(reasons)):
            ax.text(
                column_index,
                row_index,
                f"{shares[row_index, column_index]:.1%}\n(n={int(matrix[row_index, column_index])})",
                ha="center",
                va="center",
                color="white" if shares[row_index, column_index] > 0.45 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="Share within method misses")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_task_offload_diagnostics(output_dir: Path, methods: Sequence[Dict]) -> Dict[str, Path]:
    rows = flatten_method_traces(methods)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "task_trace": _write_csv(output_dir / "task_trace.csv", rows),
        "task_type_offload_summary": _write_csv(
            output_dir / "task_type_offload_summary.csv",
            build_task_type_offload_summary(rows),
        ),
        "bimodality_summary": _write_csv(
            output_dir / "bimodality_summary.csv",
            build_bimodality_summary(rows),
        ),
        "deadline_miss_reason_summary": _write_csv(
            output_dir / "deadline_miss_reason_summary.csv",
            build_deadline_reason_summary(rows),
        ),
    }
    distribution = plot_offload_distribution(
        rows,
        output_dir / "task_type_offload_distribution.png",
    )
    if distribution is not None:
        paths["task_type_offload_distribution"] = distribution
    deadline = plot_deadline_reasons(
        rows,
        output_dir / "deadline_miss_reasons.png",
    )
    if deadline is not None:
        paths["deadline_miss_reasons"] = deadline
    return paths

def active_queue_remaining_cycles(server) -> float:
    if server is None:
        return 0.0
    return float(
        sum(
            task.get("remaining_cycles", 0.0)
            for task in server.task_queue
            if task.get("status") in ("queued", "processing")
        )
    )


class HandoverActionabilityAccumulator:
    """Read-only diagnostics for whether users can act on satellite load."""

    def __init__(self) -> None:
        self.steps = 0
        self.user_steps = 0
        self.connected_user_steps = 0
        self.gate_open_user_steps = 0
        self.raw_candidate_targets = 0
        self.raw_candidate_user_steps = 0
        self.ungated_legal_targets = 0
        self.gated_legal_targets = 0
        self.ungated_feasible_user_steps = 0
        self.gated_feasible_user_steps = 0
        self.gate_blocked_feasible_user_steps = 0
        self.load_comparison_user_steps = 0
        self.lower_load_candidate_user_steps = 0
        self.congestion_relief_user_steps = 0
        self.gate_blocked_congestion_relief_user_steps = 0
        self.avoidable_full_queue_user_steps = 0
        self.queue_advantage_sum = 0.0
        self.positive_queue_advantage_sum = 0.0
        self.wait_advantage_sum_sec = 0.0
        self.positive_wait_advantage_sum_sec = 0.0
        self.concentration_steps = 0
        self.hhi_sum = 0.0
        self.effective_serving_satellites_sum = 0.0
        self.max_serving_share_sum = 0.0
        self.occupied_satellites_sum = 0.0
        self.gate_reason_counts: Dict[str, int] = defaultdict(int)
        self.candidate_status_counts: Dict[str, int] = defaultdict(int)

    @staticmethod
    def _safe_rate(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator > 0 else 0.0

    @staticmethod
    def _estimated_wait_sec(server) -> float:
        if server is None:
            return 0.0
        capacity_hz = max(float(server.total_capacity_ghz) * 1e9, 1e-9)
        return active_queue_remaining_cycles(server) / capacity_hz

    @staticmethod
    def _gate_reason(env, user) -> str:
        if int(user.serving_satellite) < 0:
            return "disconnected"
        visibility = env._get_satellite_visibility(
            user,
            int(user.serving_satellite),
        )
        if visibility is None or not visibility.is_visible:
            return "current_link_unavailable"
        threshold = float(
            getattr(env.config, "pre_handover_rvt_sec", 60.0)
        )
        if float(visibility.rvt_seconds) < threshold:
            return "low_rvt"
        server = env.mec_manager.get_server(int(user.serving_satellite))
        if server is not None and float(server.utilization) >= 0.95:
            return "high_utilization"
        return "other"

    def observe(self, env) -> None:
        """Sample current state before policy selection without mutating it."""
        self.steps += 1
        self.user_steps += int(env.num_users)

        pre_handover = np.asarray(
            env.get_pre_handover_mask(),
            dtype=bool,
        )
        gated_mask = np.asarray(
            env.get_handover_action_mask(
                env.max_visible_sats,
                apply_pre_handover_gate=True,
            ),
            dtype=bool,
        )
        ungated_mask = np.asarray(
            env.get_handover_action_mask(
                env.max_visible_sats,
                apply_pre_handover_gate=False,
            ),
            dtype=bool,
        )

        serving_counts: Dict[int, int] = defaultdict(int)
        for user in env.user_manager.users:
            if int(user.serving_satellite) >= 0:
                serving_counts[int(user.serving_satellite)] += 1
        connected = sum(serving_counts.values())
        if connected > 0:
            shares = np.asarray(
                [count / connected for count in serving_counts.values()],
                dtype=np.float64,
            )
            hhi = float(np.sum(np.square(shares)))
            self.concentration_steps += 1
            self.hhi_sum += hhi
            self.effective_serving_satellites_sum += 1.0 / max(hhi, 1e-12)
            self.max_serving_share_sum += float(np.max(shares))
            self.occupied_satellites_sum += float(len(serving_counts))

        for user_id, user in enumerate(env.user_manager.users):
            gate_open = bool(pre_handover[user_id])
            if gate_open:
                self.gate_open_user_steps += 1
                self.gate_reason_counts[self._gate_reason(env, user)] += 1

            candidates = env._get_handover_candidates(user)[
                : env.max_visible_sats
            ]
            self.raw_candidate_targets += len(candidates)
            if candidates:
                self.raw_candidate_user_steps += 1
            ungated_indices = [
                index
                for index in range(1, len(candidates) + 1)
                if ungated_mask[user_id, index]
            ]
            gated_indices = [
                index
                for index in range(1, len(candidates) + 1)
                if gated_mask[user_id, index]
            ]
            self.ungated_legal_targets += len(ungated_indices)
            self.gated_legal_targets += len(gated_indices)

            ungated_index_set = set(ungated_indices)
            for index, candidate in enumerate(candidates, start=1):
                if index in ungated_index_set:
                    self.candidate_status_counts["legal"] += 1
                    continue
                link_feasible, failure_reason = (
                    env._check_handover_link_feasibility(candidate)
                )
                if not link_feasible:
                    self.candidate_status_counts[
                        str(failure_reason or "radio_infeasible")
                    ] += 1
                    continue
                migration_plan = (
                    env.mec_manager.prepare_user_task_migration(
                        user_id=int(user.user_id),
                        old_sat_id=int(user.serving_satellite),
                        new_sat_id=int(candidate.sat_id),
                    )
                )
                self.candidate_status_counts[
                    "migration_feasible"
                    if migration_plan.feasible
                    else str(
                        migration_plan.failure_reason
                        or "migration_infeasible"
                    )
                ] += 1

            if ungated_indices:
                self.ungated_feasible_user_steps += 1
            if gated_indices:
                self.gated_feasible_user_steps += 1
            if ungated_indices and not gate_open:
                self.gate_blocked_feasible_user_steps += 1

            serving_satellite = int(user.serving_satellite)
            if serving_satellite < 0:
                continue
            self.connected_user_steps += 1
            current_server = env.mec_manager.get_server(serving_satellite)
            if current_server is None or not ungated_indices:
                continue

            legal_target_servers = [
                env.mec_manager.get_server(candidates[index - 1].sat_id)
                for index in ungated_indices
            ]
            legal_target_servers = [
                server for server in legal_target_servers if server is not None
            ]
            if not legal_target_servers:
                continue

            best_server = min(
                legal_target_servers,
                key=lambda server: (
                    int(server.queue_length),
                    self._estimated_wait_sec(server),
                    int(server.satellite_id),
                ),
            )
            current_queue = int(current_server.queue_length)
            best_queue = int(best_server.queue_length)
            queue_advantage = float(current_queue - best_queue)
            wait_advantage = (
                self._estimated_wait_sec(current_server)
                - self._estimated_wait_sec(best_server)
            )
            self.load_comparison_user_steps += 1
            self.queue_advantage_sum += queue_advantage
            self.wait_advantage_sum_sec += wait_advantage

            if queue_advantage > 0.0 or wait_advantage > 0.0:
                self.lower_load_candidate_user_steps += 1
                self.positive_queue_advantage_sum += max(
                    queue_advantage,
                    0.0,
                )
                self.positive_wait_advantage_sum_sec += max(
                    wait_advantage,
                    0.0,
                )

            max_concurrent = max(
                int(current_server.config.mec_max_concurrent_tasks),
                1,
            )
            current_congested = (
                current_queue >= max_concurrent
                or bool(current_server.is_full)
                or float(current_server.utilization) >= 0.95
            )
            has_relief = current_congested and (
                best_queue < current_queue
                or self._estimated_wait_sec(best_server)
                < self._estimated_wait_sec(current_server)
            )
            if has_relief:
                self.congestion_relief_user_steps += 1
                if not gate_open:
                    self.gate_blocked_congestion_relief_user_steps += 1
            if bool(current_server.is_full) and not bool(best_server.is_full):
                self.avoidable_full_queue_user_steps += 1

    def summary(self) -> Dict[str, object]:
        comparison_count = self.load_comparison_user_steps
        lower_load_count = self.lower_load_candidate_user_steps
        feasible_count = self.ungated_feasible_user_steps
        reason_rates = {
            reason: self._safe_rate(count, self.gate_open_user_steps)
            for reason, count in sorted(self.gate_reason_counts.items())
        }
        candidate_status_rates = {
            status: self._safe_rate(count, self.raw_candidate_targets)
            for status, count in sorted(self.candidate_status_counts.items())
        }
        return {
            "sample_steps": int(self.steps),
            "user_step_samples": int(self.user_steps),
            "connected_user_step_samples": int(self.connected_user_steps),
            "pre_handover_gate_open_rate": self._safe_rate(
                self.gate_open_user_steps,
                self.user_steps,
            ),
            "mean_raw_candidate_targets_per_user": self._safe_rate(
                self.raw_candidate_targets,
                self.user_steps,
            ),
            "raw_candidate_user_rate": self._safe_rate(
                self.raw_candidate_user_steps,
                self.user_steps,
            ),
            "mean_ungated_legal_targets_per_user": self._safe_rate(
                self.ungated_legal_targets,
                self.user_steps,
            ),
            "mean_gated_legal_targets_per_user": self._safe_rate(
                self.gated_legal_targets,
                self.user_steps,
            ),
            "ungated_feasible_switch_user_rate": self._safe_rate(
                feasible_count,
                self.user_steps,
            ),
            "gated_feasible_switch_user_rate": self._safe_rate(
                self.gated_feasible_user_steps,
                self.user_steps,
            ),
            "gate_blocked_feasible_switch_user_rate": self._safe_rate(
                self.gate_blocked_feasible_user_steps,
                self.user_steps,
            ),
            "gate_block_share_of_feasible_switch_user_steps": self._safe_rate(
                self.gate_blocked_feasible_user_steps,
                feasible_count,
            ),
            "lower_load_candidate_user_rate": self._safe_rate(
                lower_load_count,
                comparison_count,
            ),
            "mean_current_minus_best_candidate_queue": self._safe_rate(
                self.queue_advantage_sum,
                comparison_count,
            ),
            "mean_positive_queue_reduction_tasks": self._safe_rate(
                self.positive_queue_advantage_sum,
                lower_load_count,
            ),
            "mean_current_minus_best_candidate_wait_sec": self._safe_rate(
                self.wait_advantage_sum_sec,
                comparison_count,
            ),
            "mean_positive_workload_wait_reduction_sec": self._safe_rate(
                self.positive_wait_advantage_sum_sec,
                lower_load_count,
            ),
            "congestion_relief_opportunity_user_rate": self._safe_rate(
                self.congestion_relief_user_steps,
                self.connected_user_steps,
            ),
            "gate_blocked_congestion_relief_user_rate": self._safe_rate(
                self.gate_blocked_congestion_relief_user_steps,
                self.connected_user_steps,
            ),
            "gate_block_share_of_congestion_relief_opportunities": (
                self._safe_rate(
                    self.gate_blocked_congestion_relief_user_steps,
                    self.congestion_relief_user_steps,
                )
            ),
            "avoidable_full_queue_user_rate": self._safe_rate(
                self.avoidable_full_queue_user_steps,
                self.connected_user_steps,
            ),
            "mean_serving_satellite_hhi": self._safe_rate(
                self.hhi_sum,
                self.concentration_steps,
            ),
            "mean_effective_serving_satellites": self._safe_rate(
                self.effective_serving_satellites_sum,
                self.concentration_steps,
            ),
            "mean_max_serving_satellite_share": self._safe_rate(
                self.max_serving_share_sum,
                self.concentration_steps,
            ),
            "mean_occupied_serving_satellites": self._safe_rate(
                self.occupied_satellites_sum,
                self.concentration_steps,
            ),
            "gate_open_reason_rates": reason_rates,
            "raw_candidate_status_rates": candidate_status_rates,
        }
