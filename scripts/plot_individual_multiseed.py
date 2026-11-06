"""Export one scientific figure per recorded metric without changing experiment data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baseline_plot_config import method_visual_style


@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    ylabel: str
    scale: float = 1.0
    kind: str = "line"


METRICS = [
    Metric("avg_delay", "Resolved-Task Delay", "Average Delay (ms)", 1000),
    Metric("avg_success_delay", "Successful-Task Delay", "Average Delay (ms)", 1000),
    Metric("p95_success_delay", "P95 Successful-Task Delay", "P95 Delay (ms)", 1000),
    Metric("energy_per_successful_task", "Energy per Successful Task", "Energy / Successful Task (J)"),
    Metric("energy_per_resolved_task", "Energy per Resolved Task", "Energy / Resolved Task (J)"),
    Metric("task_success_rate", "Task Success Rate", "Task Success Rate (%)", 100),
    Metric("deadline_violation_rate", "Deadline Violation Rate", "Deadline Violation Rate (%)", 100),
    Metric("task_failure_rate", "Task Failure Rate", "Task Failure Rate (%)", 100),
    Metric("task_completion_rate", "Resolved-Task Completion Rate", "Task Completion Rate (%)", 100),
    Metric("pending_task_rate", "Pending-Task Rate", "Pending-Task Rate (%)", 100),
    Metric("task_settlement_rate", "Task Settlement Rate", "Task Settlement Rate (%)", 100),
    Metric("task_resolution_rate", "Task Resolution Rate", "Task Resolution Rate (%)", 100),
    Metric("blocked_time_ratio", "Blocked User-Time Ratio", "Blocked User-Time Ratio (%)", 100),
    Metric("service_availability_rate", "Service Availability", "Service Availability (%)", 100),
    Metric("service_continuity_rate", "Service Continuity", "Uninterrupted User-Time (%)", 100),
    Metric("forced_termination_rate", "Forced Disconnection Event Ratio", "Forced Disconnection Events (%)", 100),
    Metric("handover_success_rate", "Handover Success Rate", "Handover Success Rate (%)", 100),
    Metric("handover_failure_rate", "Handover Failure Rate", "Handover Failure Rate (%)", 100),
    Metric("handover_frequency", "Handover Frequency", "Handovers / User-Second"),
    Metric("handovers_per_user_minute", "Handover Frequency", "Handovers / User-Minute"),
    Metric("handover_action_rate", "Handover Action Rate", "Handover Actions (%)", 100),
    Metric("local_compute_rate", "Local-Mode Action Rate", "Local-Mode Actions (%)", 100),
    Metric("mean_offload_ratio", "Mean Offloading Ratio", "Offloading Ratio (%)", 100),
    Metric("jain_mec_load_fairness", "MEC Load Jain Fairness", "Jain Fairness Index"),
    Metric("reachable_jain_mec_load_fairness", "Reachable-MEC Load Jain Fairness", "Jain Fairness Index"),
    Metric("load_balance_variance", "MEC Load Variance", "Normalized Load Variance"),
    Metric("load_balance_coefficient", "MEC Load-Balance Coefficient", "Load-Balance Coefficient"),
    Metric("mec_load_fairness", "MEC Load-Balance Coefficient", "Load-Balance Coefficient"),
    Metric("active_load_balance_score", "Active Load-Balance Score", "Load-Balance Coefficient"),
    Metric("avg_load_balance_score", "Average Load-Balance Score", "Load-Balance Coefficient"),
    Metric("active_mec_load_fairness", "Legacy Active-MEC Load Fairness", "Legacy Load-Fairness Score"),
    Metric("mean_reward", "Evaluation Reward", "Mean Episode Reward"),
    Metric("std_reward", "Evaluation Reward Variability", "Within-Seed Reward SD"),
    Metric("selection_score", "Model Selection Score", "Selection Score (Reward)"),
    Metric("successful_task_throughput", "Per-User Successful-Task Throughput", "Tasks / User-Minute", kind="bar"),
    Metric("total_energy", "Total Terminal Energy", "Energy / Episode (J)", kind="bar"),
    Metric("completed_tasks", "Successful Tasks", "Tasks / Episode", kind="bar"),
    Metric("deadline_violations", "Deadline-Violating Tasks", "Tasks / Episode", kind="bar"),
    Metric("failed_tasks", "Other Failed Tasks", "Tasks / Episode", kind="bar"),
    Metric("pending_tasks", "Pending Tasks", "Tasks / Episode", kind="bar"),
    Metric("resolved_tasks", "Resolved Tasks", "Tasks / Episode", kind="bar"),
    Metric("total_tasks", "Generated Tasks", "Tasks / Episode", kind="bar"),
    Metric("total_handovers", "Total Handover Attempts", "Attempts / Episode", kind="bar"),
    Metric("handover_attempts", "Handover Attempts", "Attempts / Episode", kind="bar"),
    Metric("handover_committed", "Committed Handovers", "Handovers / Episode", kind="bar"),
    Metric("handover_aborted", "Aborted Handovers", "Handovers / Episode"),
    Metric("failed_handovers", "Failed Handovers", "Handovers / Episode"),
    Metric("handover_radio_failures", "Handover Radio Failures", "Failures / Episode"),
    Metric("migration_rejections", "Migration Rejections", "Rejections / Episode"),
    Metric("reconnection_attempts", "Reconnection Attempts", "Attempts / Episode", kind="bar"),
    Metric("reconnections", "Successful Reconnections", "Reconnections / Episode", kind="bar"),
    Metric("blocked_user_seconds", "Blocked User-Time", "Blocked Time / Episode (user-s)", kind="bar"),
    Metric("handover_interruption_seconds", "Handover Interruption Time", "Interruption / Episode (user-s)", kind="bar"),
    Metric("service_interruption_seconds", "Service Interruption Time", "Interruption / Episode (user-s)", kind="bar"),
    Metric("total_user_seconds", "Total User-Time Exposure", "Observation Time / Episode (user-s)", kind="bar"),
    Metric("reward_task_success", "Task-Success Reward", "Reward Contribution / Episode", kind="bar"),
    Metric("reward_load_balance", "Load-Balance Reward", "Reward Contribution / Episode", kind="bar"),
    Metric("penalty_delay", "Delay Penalty", "Reward Contribution / Episode", kind="bar"),
    Metric("penalty_energy", "Energy Penalty", "Reward Contribution / Episode", kind="bar"),
    Metric("penalty_task_failure", "Task-Failure Penalty", "Reward Contribution / Episode", kind="bar"),
    Metric("penalty_service_interruption", "Service-Interruption Penalty", "Reward Contribution / Episode", kind="bar"),
    Metric("penalty_failed_handover", "Failed-Handover Penalty", "Reward Contribution / Episode"),
    Metric("primary_metric_win_count", "Primary-Metric Wins", "Mean Wins per Seed (out of 6)", kind="bar"),
]

METHODS = [
    ("han_pdqn", "HAN+PDQN"),
    ("han_mappo", "HAN+MAPPO"),
    ("mappo_no_han", "MAPPO"),
    ("maddpg", "MADDPG"),
    ("pdqn", "PDQN"),
    ("min_distance", "Min-Distance"),
    ("full_local", "Full-Local"),
]

EXCLUDED = {
    "num_users": "用户规模横轴，不是待比较的结果指标。",
    "seed": "训练种子标识。",
    "seed_count": "有效训练种子数量；用于说明统计覆盖。",
    "episodes": "每次评估的回合数，当前恒为 5。",
    "trained_timesteps": "训练预算元数据，存在缺失且有效值恒定。",
    "best_training_eval_reward": "只在部分方法中保存的模型选择记录，不能替代统一测试结果。",
    "selected_offload": "仅单一基线保存的固定参数，不能形成跨方法比较。",
    "load_variance_sample_count": "负载统计采样数量，不作为性能指标。",
}

ALIASES = {
    "mec_load_fairness": "load_balance_coefficient",
    "active_load_balance_score": "load_balance_coefficient",
    "avg_load_balance_score": "load_balance_coefficient",
    "reachable_jain_mec_load_fairness": "jain_mec_load_fairness",
    "task_resolution_rate": "task_settlement_rate",
    "selection_score": "mean_reward",
    "total_handovers": "handover_attempts",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: object) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "legend.fontsize": 9.5, "lines.linewidth": 1.8,
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.45, "grid.linestyle": "--",
        "axes.axisbelow": True, "axes.formatter.useoffset": False,
    })


def validate_source(summary: list[dict], seeds: list[dict]) -> None:
    expected = {(n, method) for n in (20, 25, 30, 35, 40) for method, _ in METHODS}
    actual = {(int(row["num_users"]), row["method"]) for row in summary}
    if actual != expected or len(summary) != len(expected):
        raise ValueError("Unexpected or duplicate result groups.")
    fields = {spec.key for spec in METRICS}
    numeric = {key for key in summary[0] if any(number(row.get(key)) is not None for row in summary)}
    unaccounted = numeric - fields - set(EXCLUDED)
    unaccounted = {key for key in unaccounted if not key.endswith(("_ci_low", "_ci_high", "_sample_count"))}
    if unaccounted:
        raise ValueError(f"Unaccounted numeric result fields: {sorted(unaccounted)}")
    for row in summary:
        pair = (int(row["num_users"]), row["method"])
        source = [r for r in seeds if (int(r["num_users"]), r["method"]) == pair]
        if {int(r["seed"]) for r in source} != {42, 43, 44, 45, 46} or len(source) != 5:
            raise ValueError(f"Unexpected seed coverage for {pair}")
        for spec in METRICS:
            value = number(row[spec.key])
            values = [number(r.get(spec.key)) for r in source]
            if value is None or any(v is None for v in values):
                raise ValueError(f"Missing {spec.key} for {pair}")
            if not np.isclose(value, np.mean(values), rtol=1e-9, atol=1e-10):
                raise ValueError(f"Seed mean mismatch: {pair}, {spec.key}")
            lo, hi = number(row.get(spec.key + "_ci_low")), number(row.get(spec.key + "_ci_high"))
            if lo is None or hi is None or lo > value + 1e-9 or hi < value - 1e-9:
                raise ValueError(f"Invalid confidence interval: {pair}, {spec.key}")


def plot_metric(rows: list[dict], spec: Metric, output: Path) -> dict:
    is_bar = spec.kind == "bar"
    fig, ax = plt.subplots(figsize=(9.2 if is_bar else 7.6, 5.5))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.14, top=0.77)
    users = sorted({int(row["num_users"]) for row in rows})
    locations = np.arange(len(users), dtype=float)
    width = 0.112
    hatches = ["///", "\\\\", "..", "xx", "", "++", "--"]
    plot_methods = [(method, label) for method, label in METHODS
                    if not (spec.key == "blocked_user_seconds" and method == "full_local")]
    center = (len(plot_methods) - 1) / 2
    plotted = []
    for index, (method, label) in enumerate(plot_methods):
        group = sorted((r for r in rows if r["method"] == method), key=lambda r: int(r["num_users"]))
        x = np.asarray([int(r["num_users"]) for r in group])
        y = np.asarray([float(r[spec.key]) * spec.scale for r in group])
        lo = np.asarray([float(r[spec.key + "_ci_low"]) * spec.scale for r in group])
        hi = np.asarray([float(r[spec.key + "_ci_high"]) * spec.scale for r in group])
        style = method_visual_style(method)
        if is_bar:
            xx = locations + (index - center) * width
            ax.bar(xx, y, width=width, label=label, color=style["color"],
                   edgecolor="white", linewidth=0.45, hatch=hatches[index],
                   yerr=np.maximum(0, np.stack([y - lo, hi - y])),
                   error_kw={"elinewidth": 0.8, "capsize": 1.8, "capthick": 0.7, "ecolor": "#333333"},
                   zorder=3)
        else:
            ax.plot(x, y, label=label, **style, clip_on=False,
                    zorder=4 if method == "han_pdqn" else 3)
            ax.fill_between(x, lo, hi, color=style["color"], alpha=0.14, linewidth=0)
        plotted.append({"method": method, "users": x.tolist(), "mean": y.tolist(),
                        "ci_low": lo.tolist(), "ci_high": hi.tolist()})
    if is_bar:
        ax.set_xticks(locations, labels=[str(n) for n in users])
        ax.set_xlim(-0.6, len(users) - 0.4)
        low, high = ax.get_ylim()
        ax.set_ylim(min(low, 0), max(high, 0))
        ax.axhline(0, color="#555555", linewidth=0.6)
        ax.grid(axis="x", visible=False)
    else:
        ax.set_xticks(users)
        ax.set_xlim(min(users) - 1, max(users) + 1)
        if spec.scale == 100:
            lower, upper = ax.get_ylim()
            ax.set_ylim(max(0, lower), min(100, upper))
        elif all(min(s["ci_low"]) >= 0 for s in plotted):
            lower, upper = ax.get_ylim()
            if lower < 0:
                ax.set_ylim(0, upper)
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel("Number of Users")
    ax.set_ylabel(spec.ylabel)
    ax.set_title(spec.title, pad=11)
    sparse_metrics = {"migration_rejections", "failed_handovers", "handover_aborted",
                      "handover_radio_failures", "penalty_failed_handover"}
    if spec.key in sparse_metrics:
        zeros = [dict(METHODS)[series["method"]] for series in plotted
                 if all(value == 0 for value in series["mean"])]
        if zeros:
            ax.text(0.02, 0.97, textwrap.fill("Zero throughout: " + ", ".join(zeros), 65),
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    if spec.key in {"blocked_time_ratio",
                    "service_interruption_seconds", "penalty_service_interruption"}:
        inset = ax.inset_axes([0.12, 0.22 if spec.key == "penalty_service_interruption" else 0.30, 0.76, 0.49])
        for index, series in enumerate(plotted):
            y, lo, hi = (np.asarray(series[key]) for key in ("mean", "ci_low", "ci_high"))
            style = dict(method_visual_style(series["method"]))
            if is_bar:
                xx = locations + (index - center) * width
                inset.bar(xx, y, width=width, color=style["color"], edgecolor="white",
                          linewidth=0.3, hatch=hatches[index],
                          yerr=np.maximum(0, np.stack([y - lo, hi - y])),
                          error_kw={"elinewidth": 0.6, "capsize": 1, "ecolor": "#333333"})
            else:
                style.update(markersize=5, linewidth=1.3)
                inset.plot(users, y, **style)
                inset.fill_between(users, lo, hi, color=style["color"], alpha=0.14)
        detail = [series for series in plotted if series["method"] != "full_local"]
        lower = min(min(series["ci_low"]) for series in detail)
        upper = max(max(series["ci_high"]) for series in detail)
        inset.set_ylim(min(0, lower * 1.12), max(0, upper * 1.12))
        inset.set_xticks(locations if is_bar else users, labels=[str(n) for n in users])
        inset.tick_params(labelsize=8)
        inset.yaxis.set_major_locator(MaxNLocator(nbins=4))
        inset.set_title("Detail near zero", fontsize=9, pad=3)
        inset.patch.set_alpha(0.97)
        inset.grid(axis="x", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.995),
               ncol=3 if len(handles) == 6 else 4,
               frameon=True, columnspacing=1.05, handletextpad=0.45)
    path = output / f"{spec.key}.png"
    fig.savefig(path, facecolor="white", metadata={"Description": f"{spec.key}; 5 training seeds; 95% bootstrap CI"})
    plt.close(fig)
    with Image.open(path) as image:
        image.verify()
    return {"file": path.name, "metric": spec.key, "title": spec.title,
            "ylabel": spec.ylabel, "scale": spec.scale, "kind": spec.kind,
            "series": plotted,
            "excluded_methods": [method for method, _ in METHODS if method not in dict(plot_methods)]}


def write_index(output: Path, report: dict) -> None:
    result_figures = report.get("results", [])
    training = report.get("training", {})
    lines = [
        "# multiseed_0907 指标独立图", "",
        "来源：`results/multiseed_0907` 当前结果；5 个训练种子 42–46、用户规模 20/25/30/35/40。",
        "结果图默认包含 7 种方法；按用户要求，`blocked_user_seconds.png` 仅比较除 Full-Local 以外的 6 种方法。训练图只包含实际保存了该项轨迹的方法。", "",
        "- 根目录：评估结果随用户规模变化，一指标一图，文件名使用 CSV 指标名。",
        "- `training/`：训练过程随 episode 变化，同一指标的五个用户规模放在一张图中。",
        "- 沿用原图 Times New Roman/serif、算法颜色与标记；红色星形为 HAN+PDQN。",
        "- 结果图直接使用原表均值及训练种子层面的 95% bootstrap CI。",
        "- 比例转百分比，时延转 ms，吞吐量保持 tasks/(user·min)，奖励惩罚保留负值。",
        "- 柱状图从零起；有序规模趋势采用折线，吞吐量、计数和奖励贡献采用分组柱状，稀疏事件采用点线。",
        "- 阻塞比例及中断相关图保留完整主轴，并附靠近零的局部放大；`blocked_user_seconds.png` 已移除 Full-Local 及放大窗，直接展示其余 6 种方法。",
        "- 原始结果、历史记录及原始图片均未修改。", "",
        "## 统计口径", "",
        "当前汇总包含 U20/seed43、U30/seed45 两次 HAN+PDQN 复跑替换，图中沿用该来源，不视为未经选择的新实验。",
        "成功率以生成任务数为分母；完成率以已结算任务数为分母，两者不能互换。",
        "每成功任务能耗包含失败及已启动未完成任务已记账的能量。P95 是各评估回合 P95 的均值。",
        "Full-Local 没有切换或 MEC 负载，其零失败率及公平性极值不表示切换或负载管理更优。", "",
        "## 结果图清单", "", "| 指标文件 | 标题 | 图型 | 单位/纵轴 |", "|---|---|---|---|",
    ]
    for item in result_figures:
        lines.append(f"| [{item['metric']}]({item['file']}) | {item['title']} | {'分组柱状＋95% CI' if item['kind'] == 'bar' else '折线＋95% CI'} | {item['ylabel']} |")
    lines += ["", "## 重复字段", "", "按要求保留每个有数据的指标文件；以下字段在本批结果中是重复表达，不应在论文中当作独立证据：", ""]
    lines += [f"- `{key}` 与 `{value}` 数值一致。" for key, value in ALIASES.items()]
    lines += ["", "`failed_handovers` 与 `handover_aborted` 在本批结果中也数值一致，但保留各自事件名称。", "",
              "## 不单独绘图的元数据", ""]
    lines += [f"- `{key}`：{reason}" for key, reason in EXCLUDED.items()]
    lines += ["- `*_ci_low`、`*_ci_high` 用于误差范围；`*_sample_count` 用于统计覆盖，不单独出图。", ""]
    if training:
        lines += ["## 训练曲线", "", "每个 episode 为 512 环境步；优先使用记录的真实累计 `episodes`，并排除 `partial_episode=True` 的未完整回合。",
                  "训练指标仅保留预算内的完整回合（最多 292）。PPO 每 1024 步记录一次，保留 2、4、… 的实际采样位置，不插造逐回合数据。",
                  "PPO 的累计计数、总能耗和奖励分项除以记录覆盖的回合数，统一为每回合；已归一化比例、时延和平均奖励不重复除。",
                  "训练曲线不平滑，置信区间按完整训练种子轨迹 bootstrap 2000 次得到。稀疏训练评估的横轴采用准确 steps/512，可出现小数 episode 等价位置。",
                  "缺失字段不补零。PDQN 的吞吐训练轨迹只有 4 个种子，图例明确标注；其余覆盖见下表及 plot_manifest.json。",
                  "不同算法的 loss 和 recent_mean_reward 窗口定义可能不同，这些图仅为各自优化过程的诊断。", "",
                  "| 指标文件 | 标题 | 实际支持的方法与训练种子数 |", "|---|---|---|"]
        for item in training.get("specs", []):
            counts = {}
            for panel in item.get("coverage", {}).values():
                for method, details in panel.items():
                    counts.setdefault(method, set()).add(details["n_seeds"])
            descriptions = []
            names = dict(METHODS)
            for method, _ in METHODS:
                if method not in counts:
                    continue
                values = sorted(counts[method])
                n = str(values[0]) if len(values) == 1 else f"{values[0]}–{values[-1]}"
                descriptions.append(f"{names[method]} (n={n})")
            filename = item["filename"]
            lines.append(f"| [{item['metric']}](training/{filename}) | {item['title']} | {', '.join(descriptions)} |")
        lines += [""]
    lines += ["## 图型参考", "",
              "- [Shen et al., IEEE TMC 2024](https://shen-hang.github.io/files/TMC2024.pdf)：Fig.9 完成率分组柱状；Fig.10–11 失败率与完成率随参数变化的折线。",
              "- [Dai et al., IEEE TMC 2025](https://www.cs.newpaltz.edu/~lik/publications/Longbao-Dai-IEEE-TMC-2025.pdf)：Fig.7(b) 吞吐分组柱状与误差棒；Fig.6(c) 吞吐时序折线。",
              "- [Huang et al., IEEE JSAC 2024](https://arxiv.org/pdf/2401.01140)：Fig.7 能耗、Fig.11 时延随用户数变化的带标记折线。", "",
              "## 复现", "", "```powershell", "python scripts/plot_individual_multiseed.py", "```", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "results" / "multiseed_0907")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "figure")
    parser.add_argument("--section", choices=("all", "results", "training"), default="all")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = [source / "multiuser_summary.csv", source / "multiuser_seed_records.csv", source / "suite_manifest.json"]
    fingerprints = {str(p): sha256(p) for p in inputs}
    manifest_path = output / "plot_manifest.json"
    report = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    report.update({"source": str(source), "source_sha256": fingerprints, "training_seeds": [42, 43, 44, 45, 46],
                   "user_counts": [20, 25, 30, 35, 40], "aliases": ALIASES, "excluded_metadata": EXCLUDED})
    if args.section in ("all", "results"):
        with inputs[0].open(encoding="utf-8-sig", newline="") as handle:
            summary = list(csv.DictReader(handle))
        with inputs[1].open(encoding="utf-8-sig", newline="") as handle:
            seed_rows = list(csv.DictReader(handle))
        validate_source(summary, seed_rows)
        setup_style()
        report["results"] = []
        for index, spec in enumerate(METRICS, 1):
            report["results"].append(plot_metric(summary, spec, output))
            if index % 10 == 0 or index == len(METRICS):
                print(f"Result figures: {index}/{len(METRICS)}", flush=True)
    if args.section in ("all", "training"):
        from scripts.plot_multiseed_training_figures import generate_training_figures
        history_files = sorted(source.glob("seed*/u*/learned_baselines/*/training_history.json"))
        history_hashes = {str(path): sha256(path) for path in history_files}
        report["training"] = generate_training_figures(source, output / "training")
        for path, digest in history_hashes.items():
            if sha256(Path(path)) != digest:
                raise RuntimeError(f"Training history changed during export: {path}")
        report["training_source_sha256"] = history_hashes
    for path, digest in fingerprints.items():
        if sha256(Path(path)) != digest:
            raise RuntimeError(f"Source changed during export: {path}")
    report["source_files_verified_unchanged"] = True
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_index(output, report)
    print(json.dumps({"output": str(output), "result_figures": len(report.get("results", [])),
                      "training_figures": len(report.get("training", {}).get("files", [])),
                      "sources_unchanged": True}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
