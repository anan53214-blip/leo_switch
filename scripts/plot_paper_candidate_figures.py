"""Create the two selected IEEE double-column figures from verified seed means."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.baseline_plot_config import method_visual_style
from scripts.plot_individual_multiseed import METHODS, validate_source

WIDTH_IN = 7.16
HEIGHT_IN = 2.72
DPI = 600
USERS = [20, 25, 30, 35, 40]
HATCHES = {method: pattern for (method, _), pattern in zip(
    METHODS, ("///", "\\\\", "..", "xx", "", "++", "--"))}


@dataclass(frozen=True)
class Panel:
    key: str
    label: str
    ylabel: str
    multiplier: float = 1.0
    exclude: tuple[str, ...] = ()


LINE_PANELS = (
    Panel("avg_success_delay", "(a) Successful-task delay", "Average delay (ms)", 1000),
    Panel("deadline_violation_rate", "(b) Deadline violation rate", "Deadline violation rate (%)", 100),
    Panel("energy_per_successful_task", "(c) Energy per successful task", "Energy per successful task (J)"),
)
BAR_PANELS = (
    Panel("blocked_user_seconds", "(a) Blocked user-time", "Blocked user-time (user-s/episode)", exclude=("full_local",)),
    Panel("total_energy", "(b) Total terminal energy", "Total terminal energy (kJ/episode)", 0.001),
    Panel("handovers_per_user_minute", "(c) Handover frequency", "Handovers per user-minute"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def values(rows: list[dict], panel: Panel, method: str) -> tuple[np.ndarray, ...]:
    group = sorted((r for r in rows if r["method"] == method), key=lambda r: int(r["num_users"]))
    x = np.asarray([int(r["num_users"]) for r in group])
    if x.tolist() != USERS:
        raise ValueError(f"Incomplete user coverage: {panel.key}/{method}")
    y, low, high = (np.asarray([float(r[panel.key + suffix]) for r in group]) * panel.multiplier
                    for suffix in ("", "_ci_low", "_ci_high"))
    if not np.all(np.isfinite([y, low, high])) or np.any(low > y + 1e-8) or np.any(high < y - 1e-8):
        raise ValueError(f"Invalid values/CI: {panel.key}/{method}")
    return x, y, low, high


def style() -> dict:
    return {
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.65,
        "axes.labelpad": 3, "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.8, "ytick.major.size": 2.8,
        "axes.axisbelow": True, "axes.grid": True, "grid.alpha": 0.4,
        "grid.color": "#BFBFBF", "grid.linestyle": "--", "grid.linewidth": 0.45,
        "hatch.linewidth": 0.3, "savefig.bbox": None,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.formatter.useoffset": False,
    }


def draw(rows: list[dict], panels: tuple[Panel, ...], output: Path,
         filename: str, kind: str) -> dict:
    with plt.rc_context(style()):
        fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), dpi=150)
        # Preserve the physical plotting area while reclaiming the second legend row.
        axes = [fig.add_axes([left, 0.705 / HEIGHT_IN, 0.252, 1.695 / HEIGHT_IN])
                for left in (0.076, 0.406, 0.736)]
        audit = []
        for axis, panel in zip(axes, panels):
            methods = [(method, label) for method, label in METHODS if method not in panel.exclude]
            series = []
            for index, (method, label) in enumerate(methods):
                x, y, low, high = values(rows, panel, method)
                identity = method_visual_style(method)
                if kind == "line":
                    identity.update(linewidth=1.55 if method == "han_pdqn" else 1.05,
                                    markersize=5 if method == "han_pdqn" else 3.6,
                                    markeredgewidth=0.35)
                    axis.plot(x, y, **identity, clip_on=False,
                              zorder=5 if method == "han_pdqn" else 3)
                    axis.fill_between(x, low, high, color=identity["color"], alpha=0.14, linewidth=0)
                else:
                    width = 0.79 / len(methods)
                    positions = np.arange(len(USERS)) + (index - (len(methods) - 1) / 2) * width
                    axis.bar(positions, y, width=width, color=identity["color"],
                             edgecolor="white", linewidth=0.25, hatch=HATCHES[method],
                             yerr=np.maximum(0, np.stack([y - low, high - y])),
                             error_kw={"ecolor": "#333333", "elinewidth": 0.55,
                                        "capsize": 1.05, "capthick": 0.5}, zorder=3)
                series.append({"method": method, "users": x.tolist(), "mean": y.tolist(),
                               "ci_low": low.tolist(), "ci_high": high.tolist()})
            if kind == "line":
                axis.set_xlim(19, 41)
                axis.set_xticks(USERS)
                lower, upper = axis.get_ylim()
                if lower < 0 and min(min(s["ci_low"]) for s in series) >= 0:
                    axis.set_ylim(0, upper)
            else:
                axis.set_xticks(np.arange(5), labels=[str(n) for n in USERS])
                axis.set_xlim(-0.58, 4.58)
                axis.set_ylim(bottom=0)
                axis.grid(axis="x", visible=False)
            axis.set_xlabel("Number of users")
            axis.set_ylabel(panel.ylabel)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            axis.yaxis.set_major_formatter(formatter)
            center = axis.get_position().x0 + axis.get_position().width / 2
            fig.text(center, 0.216 / HEIGHT_IN, panel.label, ha="center", va="center", fontsize=8.5)
            audit.append({"metric": panel.key, "display_multiplier": panel.multiplier,
                          "ylabel": panel.ylabel, "excluded_methods": list(panel.exclude),
                          "series": series})
        handles, labels = [], []
        for method, label in METHODS:
            identity = method_visual_style(method)
            if kind == "line":
                identity.update(linewidth=1.5 if method == "han_pdqn" else 1.05,
                                markersize=5 if method == "han_pdqn" else 3.6,
                                markeredgewidth=0.35)
                handles.append(Line2D([], [], **identity))
            else:
                handles.append(Patch(facecolor=identity["color"], edgecolor="white",
                                     linewidth=0.25, hatch=HATCHES[method]))
            labels.append(label)
        legend_center = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
        legend = fig.legend(handles, labels, loc="lower center",
                            bbox_to_anchor=(legend_center, 2.47 / HEIGHT_IN), borderaxespad=0,
                            ncol=len(labels), frameon=True, fontsize=8, borderpad=0.3,
                            handlelength=1.45, handletextpad=0.4, columnspacing=0.9)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = fig.bbox
        legend_box = legend.get_window_extent(renderer)
        legend_gap_pt = (legend_box.y0 - max(ax.bbox.y1 for ax in axes)) * 72 / fig.dpi
        for text in fig.findobj(matplotlib.text.Text):
            if not text.get_visible() or not text.get_text():
                continue
            box = text.get_window_extent(renderer)
            if box.x0 < bounds.x0 - 1 or box.y0 < bounds.y0 - 1 or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1:
                raise RuntimeError(f"Text outside fixed IEEE canvas: {text.get_text()}")
        png, pdf = output / f"{filename}.png", output / f"{filename}.pdf"
        fig.savefig(png, dpi=DPI, bbox_inches=None, facecolor="white")
        fig.savefig(pdf, format="pdf", bbox_inches=None, facecolor="white",
                    metadata={"Title": filename, "Subject": "Five-training-seed means and 95% bootstrap intervals"})
        plt.close(fig)
    with Image.open(png) as image:
        if image.size != (round(WIDTH_IN * DPI), round(HEIGHT_IN * DPI)):
            raise RuntimeError(f"Unexpected PNG dimensions: {image.size}")
        pixels = image.size
        image.verify()
    return {"png": png.name, "pdf": pdf.name, "kind": kind,
            "size_inches": [WIDTH_IN, HEIGHT_IN], "size_mm": [WIDTH_IN * 25.4, HEIGHT_IN * 25.4],
            "png_dpi": DPI, "png_pixels": pixels,
            "font": "Times New Roman", "font_size_points": {"labels": 8.5, "ticks": 8, "legend": 8},
            "legend": {"rows": 1, "columns": len(labels), "center_x_fraction": legend_center,
                       "gap_above_axes_points": legend_gap_pt},
            "panels": audit}


def readme(output: Path) -> None:
    content = r"""# IEEE 双栏通栏候选图

两张图均为 **181.864 × 69.088 mm（7.16 × 2.72 in）**，每张横排三个子图；七个图例项单行排列，图例框距子图顶边约 5 pt。

| 图 | PNG 预览 | 矢量 PDF（论文优先使用） |
|---|---|---|
| 图 1：时延、违约率、每成功任务能耗 | [figure1_line_metrics.png](figure1_line_metrics.png) | [figure1_line_metrics.pdf](figure1_line_metrics.pdf) |
| 图 2：阻塞用户时间、总终端能耗、切换频率 | [figure2_bar_metrics.png](figure2_bar_metrics.png) | [figure2_bar_metrics.pdf](figure2_bar_metrics.pdf) |

- PNG：600 dpi，4296 × 1632 像素。PDF：直接矢量导出，嵌入 TrueType 字体，适合 Overleaf。
- Times New Roman；轴标签与子图标签 8.5 pt，刻度与共享图例 8 pt。
- 原有算法配色和标记；均值、95% CI 直接来自 `results/multiseed_0907/multiuser_summary.csv`。
- 五个训练种子 42–46；用户规模 20、25、30、35、40。保留原汇总中的两次复跑替换，不改变统计样本。
- 图 2(a) 沿用用户选择，排除 Full-Local；图 2(b)(c) 保留七种算法。所有柱状纵轴从零起。
- 时延为 ms，违约率为 %，每成功任务能耗为 J，累计阻塞时间为 user-s/episode，总终端能耗换算为 kJ/episode，切换频率为 handovers/user-minute。
- 每成功任务能耗的分子包含所有已记账终端能耗；五种子 CI 不表示逐任务分布。

## IEEEtran 插入示例

使用 `figure*` 与 `width=\textwidth`，不要另设 height 或使用 resizebox 强制改变纵横比。图号由 LaTeX 自动编号。

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figure1_line_metrics.pdf}
    \caption{Task-level performance versus the number of users:
    (a) average successful-task delay, (b) deadline violation rate,
    and (c) terminal energy per successful task.
    Shaded regions indicate 95\% bootstrap confidence intervals over five training seeds.}
    \label{fig:task_performance}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figure2_bar_metrics.pdf}
    \caption{Service and resource costs versus the number of users:
    (a) cumulative blocked user-time per episode, (b) total terminal energy per episode,
    and (c) handovers per user-minute. Full-Local is omitted from (a)
    and performs no handovers in (c).
    Error bars indicate 95\% bootstrap confidence intervals over five training seeds.}
    \label{fig:service_resource_costs}
\end{figure*}
```

## 复现

```powershell
python scripts/plot_paper_candidate_figures.py
```

原始数据、单指标图片和训练历史不变；本目录只保存候选成图与用途说明。
"""
    (output / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results" / "multiseed_0907")
    parser.add_argument("--output", type=Path, default=ROOT / "figure" / "paper_candidate")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = [source / "multiuser_summary.csv", source / "multiuser_seed_records.csv", source / "suite_manifest.json"]
    protected = sources + sorted((ROOT / "figure").glob("*.png")) + sorted((ROOT / "figure" / "training").glob("*.png"))
    before = {str(path): digest(path) for path in protected}
    with sources[0].open(encoding="utf-8-sig", newline="") as handle:
        summary = list(csv.DictReader(handle))
    with sources[1].open(encoding="utf-8-sig", newline="") as handle:
        seeds = list(csv.DictReader(handle))
    validate_source(summary, seeds)
    figures = [draw(summary, LINE_PANELS, output, "figure1_line_metrics", "line"),
               draw(summary, BAR_PANELS, output, "figure2_bar_metrics", "bar")]
    for path, value in before.items():
        if digest(Path(path)) != value:
            raise RuntimeError(f"Input/previous figure modified: {path}")
    manifest = {"source": str(source), "source_sha256": {str(path): before[str(path)] for path in sources},
                "training_seeds": [42, 43, 44, 45, 46], "figures": figures,
                "protected_files_verified_unchanged": len(protected)}
    (output / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme(output)
    print(json.dumps({"output": str(output), "figures": [{k: v for k, v in item.items() if k != "panels"} for item in figures],
                      "protected_files_unchanged": len(protected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
