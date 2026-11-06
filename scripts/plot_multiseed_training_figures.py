"""One-field training figures, preserving native episode sampling and seed identity.

This module reads experiment histories only.  ``generate_training_figures`` is
intended to be called by the top-level figure export script; it never edits the
histories or copies anything into ``paper/``.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np

from scripts.baseline_plot_config import method_visual_style


USER_COUNTS = (20, 25, 30, 35, 40)
METHOD_NAMES = {
    "han_pdqn": "HAN+PDQN",
    "han_mappo": "HAN+MAPPO",
    "mappo_no_han": "MAPPO",
    "maddpg": "MADDPG",
    "pdqn": "PDQN",
}
METHOD_ORDER = tuple(METHOD_NAMES)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260728

# These are accumulated over the recorded rollout.  PPO records cover two
# episodes here; rates, delays, losses and mean_reward already have their own
# denominators and MUST NOT receive this normalization.
EPISODE_TOTALS = frozenset({
    "rollout_total_reward", "total_energy", "total_handovers",
    "handover_attempts", "handover_committed", "handover_aborted",
    "handover_radio_failures", "migration_rejections", "reconnection_attempts",
    "reconnections", "successful_handovers", "failed_handovers",
    "forced_disconnects", "total_user_seconds", "blocked_user_seconds",
    "handover_interruption_seconds", "service_interruption_seconds",
    "total_tasks", "completed_tasks", "deadline_violations", "failed_tasks",
    "resolved_tasks", "pending_tasks", "reward_task_success",
    "reward_load_balance", "penalty_delay", "penalty_energy",
    "penalty_task_failure", "penalty_service_interruption",
    "penalty_failed_handover",
})
PPO_METHODS = frozenset({"han_mappo", "mappo_no_han"})
METHOD_SUPPORT = {
    "actor_loss": PPO_METHODS | {"maddpg"},
    # HAN+PDQN's critic_loss is a q_loss alias, its actor_loss is a zero
    # placeholder, and exploration_noise is absent from its PDQN algorithm.
    "critic_loss": PPO_METHODS | {"maddpg"},
    "q_loss": {"han_pdqn", "pdqn"},
    "param_loss": {"han_pdqn", "pdqn"},
    "epsilon": {"han_pdqn", "pdqn"},
    "exploration_noise": {"maddpg"},
    "std_reward": PPO_METHODS,
    "entropy": PPO_METHODS,
    "kl_divergence": PPO_METHODS,
    "clip_fraction": PPO_METHODS,
    "han_grad_norm": {"han_mappo"},
    "han_parameter_delta": {"han_mappo"},
    "han_metapaths_executed": {"han_mappo"},
    "q_mean": {"maddpg"},
    "target_q_mean": {"maddpg"},
    "td_abs_mean": {"maddpg"},
    "actor_grad_norm": {"maddpg"},
    "critic_grad_norm": {"maddpg"},
}
SKIP_FIELDS = {
    "update": "optimizer/record index, not an outcome",
    "total_steps": "horizontal coordinate",
    "episodes": "horizontal coordinate",
    "mean_length": "episode-length metadata used for validation",
    "elapsed_sec": "incompatible timing intervals across trainer types",
    "partial_episode": "record validity flag",
    "load_variance_sample_count": "sample-count metadata",
    "eval_mean_reward": "plotted separately from actual evaluation checkpoints",
    "eval_std_reward": "plotted separately from actual evaluation checkpoints",
}
LABELS = {
    "mean_reward": ("Training Episode Return", "Mean episode return"),
    "recent_mean_reward": ("Recorded Recent Episode Return", "Recent mean return"),
    "std_reward": ("Within-Rollout Episode Return Variation", "Return standard deviation"),
    "rollout_total_reward": ("Rollout Return per Episode", "Return / episode"),
    "rollout_mean_reward": ("Mean Step Reward", "Mean reward / step"),
    "avg_delay": ("Resolved-Task Delay", "Delay (ms)"),
    "avg_success_delay": ("Average Successful-Task Delay", "Delay (ms)"),
    "p95_success_delay": ("P95 Successful-Task Delay", "Delay (ms)"),
    "energy_per_successful_task": ("Energy per Successful Task", "Energy (J / successful task)"),
    "energy_per_resolved_task": ("Energy per Resolved Task", "Energy (J / resolved task)"),
    "total_energy": ("Terminal Energy per Episode", "Energy (J / episode)"),
    "successful_task_throughput": ("Per-User Successful-Task Throughput", "Tasks / user-minute"),
    "handover_frequency": ("Handover Frequency", "Handovers / user-second"),
    "handovers_per_user_minute": ("Handover Frequency", "Handovers / user-minute"),
    "load_balance_variance": ("MEC Load Variance", "Load variance"),
    "load_balance_coefficient": ("MEC Load-Balance Coefficient", "Coefficient"),
    "jain_mec_load_fairness": ("MEC Load Jain Fairness", "Jain index"),
    "reachable_jain_mec_load_fairness": ("Reachable MEC Load Jain Fairness", "Jain index"),
    "active_mec_load_fairness": ("Active MEC Load Fairness", "Fairness index"),
    "eval_mean_reward": ("Training-Checkpoint Evaluation Return", "Mean evaluation return"),
    "eval_std_reward": ("Training-Checkpoint Evaluation Return Variation", "Evaluation return standard deviation"),
    "epsilon": ("Discrete Exploration Probability", "Epsilon"),
    "exploration_noise": ("Continuous Exploration Noise", "Noise standard deviation"),
    "han_metapaths_executed": ("HAN Meta-Path Executions", "Executions / update"),
}


@dataclass
class _History:
    path: Path
    seed: int
    users: int
    method: str
    payload: dict[str, Any]
    length: float
    budget: float
    complete_limit: int


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _load_histories(source_dir: Path) -> tuple[list[_History], list[dict[str, str]]]:
    histories: list[_History] = []
    skipped: list[dict[str, str]] = []
    for seed_dir in sorted(source_dir.glob("seed*")):
        if not re.fullmatch(r"seed\d+", seed_dir.name):
            continue
        seed = int(seed_dir.name[4:])
        for users in USER_COUNTS:
            for method in METHOD_ORDER:
                path = seed_dir / f"u{users}" / "learned_baselines" / method / "training_history.json"
                if not path.exists():
                    skipped.append({"path": str(path), "reason": "history missing"})
                    continue
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
                config = payload.get("config", {})
                length = float(config.get("max_steps", 512))
                budget = float(config.get("total_timesteps", 150000))
                if length <= 0 or not math.isfinite(length) or budget <= 0:
                    raise ValueError(f"Invalid episode length/training budget: {path}")
                histories.append(_History(path, seed, users, method, payload, length,
                                          budget, int(budget // length)))
    return histories, skipped


def _episode_position(record: dict[str, Any], history: _History) -> float | None:
    if record.get("partial_episode", False):
        return None
    steps = _number(record.get("total_steps"))
    if steps is None:
        return None
    equivalent = steps / history.length
    recorded = _number(record.get("episodes"))
    # Prefer the real cumulative completed-episode counter after checking it
    # against environment steps.  An update number is never an episode number.
    if recorded is not None and abs(recorded - equivalent) <= 1e-8:
        episode = recorded
    elif abs(equivalent - round(equivalent)) <= 1e-8:
        episode = float(round(equivalent))
    else:
        return None
    mean_length = _number(record.get("mean_length"))
    if mean_length is not None and abs(mean_length - history.length) > 1e-8:
        return None
    if 0 < episode <= history.complete_limit:
        return episode
    return None


def _training_curve(history: _History, metric: str) -> dict[float, float]:
    if metric in METHOD_SUPPORT and history.method not in METHOD_SUPPORT[metric]:
        return {}
    curve: dict[float, float] = {}
    previous_steps = 0.0
    for record in history.payload.get("training", []):
        steps = _number(record.get("total_steps"))
        recorded_steps = steps - previous_steps if steps is not None else None
        if steps is not None:
            previous_steps = steps
        episode = _episode_position(record, history)
        if episode is None:
            continue
        value = _number(record.get(metric))
        # Precisely derive throughput only when its two measured inputs exist.
        if value is None and metric == "successful_task_throughput":
            completed = _number(record.get("completed_tasks"))
            user_seconds = _number(record.get("total_user_seconds"))
            if completed is not None and user_seconds is not None and user_seconds > 0:
                value = 60.0 * completed / user_seconds
        if value is None:
            continue
        if metric in EPISODE_TOTALS and history.method in PPO_METHODS:
            mean_length = _number(record.get("mean_length"))
            if mean_length is None or mean_length <= 0 or recorded_steps is None:
                continue
            episodes_in_record = recorded_steps / mean_length
            if episodes_in_record <= 0:
                continue
            value /= episodes_in_record
        curve[episode] = value
    return curve


def _evaluation_curve(history: _History, metric: str) -> dict[float, float]:
    records = history.payload.get("training_evaluation", [])
    if not records:
        records = [row for row in history.payload.get("evaluation", [])
                   if "total_steps" in row and "eval_mean_reward" in row]
    if not records:
        records = [row for row in history.payload.get("training", [])
                   if "total_steps" in row and "eval_mean_reward" in row]
    curve: dict[float, float] = {}
    for row in records:
        steps, value = _number(row.get("total_steps")), _number(row.get(metric))
        if steps is not None and 0 < steps <= history.budget and value is not None:
            curve[steps / history.length] = value
    return curve


def _aggregate(curves: dict[int, dict[float, float]]) -> tuple[np.ndarray, ...]:
    """Bootstrap seed trajectories on their exact shared native sample grid."""
    seeds = sorted(curves)
    common = sorted(set.intersection(*(set(curves[seed]) for seed in seeds)))
    if not common:
        return tuple(np.array([], dtype=float) for _ in range(4))
    samples = np.asarray([[curves[seed][x] for x in common] for seed in seeds])
    mean = samples.mean(axis=0)
    if len(seeds) == 1:
        return np.asarray(common), mean, mean.copy(), mean.copy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Multinomial weights are equivalent to sampling whole seed trajectories
    # with replacement, while avoiding a resamples x seeds x time array.
    weights = rng.multinomial(len(seeds), np.full(len(seeds), 1.0 / len(seeds)),
                              size=BOOTSTRAP_RESAMPLES) / len(seeds)
    bootstrap = weights @ samples
    low, high = np.percentile(bootstrap, (2.5, 97.5), axis=0)
    return np.asarray(common), mean, low, high


def _labels(metric: str) -> tuple[str, str, float]:
    if metric in LABELS:
        title, label = LABELS[metric]
    else:
        title = metric.replace("_", " ").title().replace("Mec", "MEC").replace("Han", "HAN")
        label = title
        if metric in EPISODE_TOTALS:
            label += " / episode"
    percentage_metrics = {"blocked_time_ratio", "clip_fraction", "mean_offload_ratio", "selected_offload"}
    scale = 100.0 if metric.endswith("_rate") or metric in percentage_metrics else 1.0
    if metric in {"avg_delay", "avg_success_delay", "p95_success_delay"}:
        scale = 1000.0
    if scale == 100.0:
        label = title + " (%)"
    if metric in {"total_user_seconds", "blocked_user_seconds",
                  "handover_interruption_seconds", "service_interruption_seconds"}:
        label = "Time (user-s / episode)"
    return title, label, scale


def _draw_metric(metric: str, grouped: dict, output_dir: Path,
                 evaluation: bool = False) -> dict[str, Any]:
    title, ylabel, scale = _labels(metric)
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.1), sharex=True, sharey=True)
    method_sizes: dict[str, set[int]] = defaultdict(set)
    coverage: dict[str, Any] = {}
    plotted = False
    for ax, users in zip(axes.flat, USER_COUNTS):
        panel = {}
        for method in METHOD_ORDER:
            curves = grouped.get((users, method), {})
            if not curves:
                continue
            x, mean, low, high = _aggregate(curves)
            if not len(x):
                continue
            plotted = True
            method_sizes[method].add(len(curves))
            style = dict(method_visual_style(method))
            style.update(markevery=max(len(x) // 7, 1),
                         markersize=7.0 if method == "han_pdqn" else 4.5,
                         linewidth=2.0 if method == "han_pdqn" else 1.25)
            ax.fill_between(x, low * scale, high * scale, color=style["color"],
                            alpha=0.12, linewidth=0, zorder=1)
            ax.plot(x, mean * scale, **style, zorder=4 if method == "han_pdqn" else 3)
            panel[method] = {"n_seeds": len(curves), "seeds": sorted(curves),
                             "points": len(x), "episode_start": float(x[0]),
                             "episode_end": float(x[-1]),
                             "native_episode_grid_preserved": True}
        ax.set_title(f"N = {users}", fontsize=13)
        ax.set_xlabel("Equivalent training episodes" if evaluation else "Completed training episodes")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=not evaluation))
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.grid(True, alpha=0.23, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_powerlimits((-3, 4))
        ax.yaxis.set_major_formatter(formatter)
        if not panel:
            ax.text(0.5, 0.5, "No recorded data", transform=ax.transAxes,
                    ha="center", va="center", color="#666666")
        coverage[str(users)] = panel
    axes.flat[-1].set_axis_off()
    if not plotted:
        plt.close(fig)
        return {}
    handles, labels = [], []
    for method in METHOD_ORDER:
        if method not in method_sizes:
            continue
        sizes = sorted(method_sizes[method])
        n_text = str(sizes[0]) if len(sizes) == 1 else f"{sizes[0]}-{sizes[-1]}"
        style = dict(method_visual_style(method))
        style.update(markersize=7 if method == "han_pdqn" else 5)
        handles.append(Line2D([], [], **style))
        labels.append(f"{METHOD_NAMES[method]} (n={n_text})")
    fig.suptitle(title, y=0.995, fontsize=17)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.957),
               ncol=min(5, len(handles)), frameon=False, fontsize=11,
               handlelength=2.7, columnspacing=1.6)
    fig.text(0.5, 0.014, "Training-seed mean and 95% bootstrap CI; native sampling; no smoothing.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=0.074, right=0.988, bottom=0.10, top=0.875,
                        wspace=0.28, hspace=0.36)
    filename = f"{metric}.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    return {"filename": filename, "metric": metric, "title": title,
            "ylabel": ylabel, "display_multiplier": scale,
            "source": "training checkpoints" if evaluation else "training",
            "per_episode_normalized": metric in EPISODE_TOTALS,
            "smoothing": "none", "coverage": coverage}


def generate_training_figures(source_dir: Path, output_dir: Path) -> dict:
    """Export one PNG per measured numeric training field and return its audit.

    Training curves use complete episodes within each saved training budget.
    Sparse evaluation curves instead use the exact fractional steps/max_steps
    coordinate, without rounding or using final test episodes as checkpoints.
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    histories, skipped_paths = _load_histories(source_dir)
    fields = sorted({key for history in histories
                     for record in history.payload.get("training", [])
                     for key, value in record.items() if _number(value) is not None})
    if any("completed_tasks" in record and "total_user_seconds" in record
           for history in histories for record in history.payload.get("training", [])):
        fields = sorted(set(fields) | {"successful_task_throughput"})
    specs: list[dict[str, Any]] = []
    skipped: dict[str, Any] = {key: SKIP_FIELDS[key] for key in fields if key in SKIP_FIELDS}
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "axes.linewidth": 0.8}):
        for metric in fields:
            if metric in SKIP_FIELDS:
                continue
            grouped: dict = defaultdict(dict)
            for history in histories:
                curve = _training_curve(history, metric)
                if curve:
                    grouped[history.users, history.method][history.seed] = curve
            spec = _draw_metric(metric, grouped, output_dir)
            if spec:
                specs.append(spec)
            else:
                skipped[metric] = "no supported finite measurements on complete episodes"
        for metric in ("eval_mean_reward", "eval_std_reward"):
            grouped = defaultdict(dict)
            for history in histories:
                curve = _evaluation_curve(history, metric)
                if curve:
                    grouped[history.users, history.method][history.seed] = curve
            spec = _draw_metric(metric, grouped, output_dir, evaluation=True)
            if spec:
                specs.append(spec)
                skipped.pop(metric, None)
    return {
        "files": [spec["filename"] for spec in specs],
        "specs": specs,
        "coverage": {"history_count": len(histories),
                     "seeds": sorted({history.seed for history in histories}),
                     "user_counts": sorted({history.users for history in histories}),
                     "methods": [m for m in METHOD_ORDER if any(h.method == m for h in histories)],
                     "per_metric": {spec["metric"]: spec["coverage"] for spec in specs}},
        "episode_conversion": {
            "rule": "validated cumulative episodes; otherwise exact total_steps/max_steps on complete episodes",
            "max_steps": sorted({h.length for h in histories}),
            "complete_episode_limits": sorted({h.complete_limit for h in histories}),
            "partial_episodes": "excluded, including the final 496-step HAN+PDQN/MADDPG records",
            "ppo_sampling": "two-episode means at native even episode coordinates; no interpolation",
            "ppo_totals": "divide recorded rollout totals by delta(total_steps)/mean_length (2 here)",
            "per_episode_normalized_fields": sorted(EPISODE_TOTALS),
            "evaluation": "exact fractional total_steps/max_steps; checkpoints beyond training budget excluded",
        },
        "confidence_interval": {"unit": "training seed", "confidence": 0.95,
                                "method": "percentile bootstrap of whole seed trajectories",
                                "resamples": BOOTSTRAP_RESAMPLES, "rng_seed": BOOTSTRAP_SEED,
                                "single_seed": "zero-width band; no uncertainty estimated"},
        "notes": [
            "All metrics retain their own native method sampling grids; no smoothing or cross-method interpolation.",
            "Missing fields are omitted rather than filled with zeros; legend n is seed count, and detailed coverage is per panel.",
            "Throughput is derived only from measured completed_tasks and total_user_seconds where both exist.",
            "PDQN seed42 throughput lacks both source inputs and is omitted, leaving n=4 for that method.",
            "Actor/critic losses, HAN gradients and exploration noise exclude trainer placeholders using actual network support.",
            "Loss aggregation and recent-mean reward windows differ by trainer; these are recorded optimization diagnostics.",
            "The current histories include promoted reruns at U20/seed43 and U30/seed45; bootstrap does not remove selection bias.",
        ],
        "skipped": {"fields": skipped, "histories": skipped_paths,
                    "method_restrictions": {k: sorted(v) for k, v in METHOD_SUPPORT.items()}},
    }
