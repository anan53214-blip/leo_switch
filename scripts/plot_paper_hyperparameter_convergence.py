"""Format the existing U30 parameter-sensitivity data for one IEEE column."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "dif_converge"
OUTPUT = ROOT / "figure" / "paper_candidate"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protected = [SOURCE / name for name in (
        "convergence_curves.csv", "convergence_seed_records.csv", "convergence_summary.csv",
        "experiment_manifest.json", "convergence_u30.pdf", "convergence_u30.png")]
    before = {str(path): digest(path) for path in protected}
    with protected[0].open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    histories = json.loads((SOURCE / "experiment_manifest.json").read_text(encoding="utf-8"))
    episode_length = int(histories["fixed_config"]["max_steps"])
    assert episode_length == 512
    OUTPUT.mkdir(parents=True, exist_ok=True)
    colors = ("#0072B2", "#009E73", "#D55E00", "#CC79A7")
    linestyles = ("--", "-.", "-", ":")
    audit = []
    with plt.rc_context({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.65, "axes.labelpad": 3,
        "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "grid.alpha": 0.4, "grid.linestyle": "--", "grid.linewidth": 0.45,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None,
    }):
        fig = plt.figure(figsize=(3.5, 4.8), dpi=150)
        axes = [fig.add_axes([0.17, 0.635, 0.80, 0.315]),
                fig.add_axes([0.17, 0.14, 0.80, 0.315])]
        cases = [
            [(rate, 512) for rate in (1e-4, 3e-4, 1e-3, 3e-3)],
            [(1e-3, batch) for batch in (128, 256, 512, 1024)],
        ]
        bounds = []
        for panel, (axis, settings) in enumerate(zip(axes, cases)):
            for index, (rate, batch) in enumerate(settings):
                setting = f"lr{rate:.0e}_bs{batch}"
                group = sorted((row for row in rows if row["setting"] == setting),
                               key=lambda row: int(row["total_steps"]))
                assert group and {int(row["seed_count"]) for row in group} == {5}
                assert {int(row["smoothing_window"]) for row in group} == {3}
                x = np.asarray([int(row["total_steps"]) / episode_length for row in group])
                mean, low, high = (np.asarray([float(row[key]) for row in group])
                                   for key in ("mean_reward", "ci_low", "ci_high"))
                assert np.array_equal(x, np.arange(1, 293))
                assert np.all(low <= mean + 1e-9) and np.all(high >= mean - 1e-9)
                default = rate == 1e-3 and batch == 512
                color = "#D55E00" if default else colors[index]
                label = rf"$\alpha={rate:.0e}$" if panel == 0 else f"B = {batch}"
                axis.plot(x, mean, color=color, linestyle=linestyles[index],
                          linewidth=1.5 if default else 1.1, label=label)
                axis.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
                bounds.extend([float(low.min()), float(high.max())])
                audit.append({"panel": panel + 1, "setting": setting,
                              "episode_count": len(x), "first_episode": float(x[0]),
                              "last_episode": float(x[-1]), "values_changed": False})
            axis.set_xlabel("Training episodes")
            axis.set_xlim(0, 300)
            axis.set_xticks([0, 50, 100, 150, 200, 250, 300])
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            axis.grid(True)
            axis.set_axisbelow(True)
            axis.legend(loc="lower right", frameon=True, ncol=2, fontsize=7.5,
                        borderpad=0.3, labelspacing=0.3, columnspacing=0.8,
                        handlelength=1.7, handletextpad=0.4)
        margin = (max(bounds) - min(bounds)) * 0.045
        for axis in axes:
            axis.set_ylim(min(bounds) - margin, max(bounds) + margin)
            axis.set_ylabel("Mean episode reward")
        fig.text(0.57, 0.52, "(a) Learning rate (batch size = 512)",
                 ha="center", va="center", fontsize=8.5)
        fig.text(0.57, 0.025, r"(b) Batch size ($\alpha=10^{-3}$)",
                 ha="center", va="center", fontsize=8.5)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for text in fig.findobj(matplotlib.text.Text):
            if not text.get_visible() or not text.get_text():
                continue
            bbox = text.get_window_extent(renderer)
            if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.x1 + 1 or bbox.y1 > fig.bbox.y1 + 1:
                raise RuntimeError(f"Clipped text: {text.get_text()}")
        for suffix in ("png", "pdf"):
            fig.savefig(OUTPUT / f"convergence_u30.{suffix}", dpi=600,
                        bbox_inches=None, facecolor="white")
        plt.close(fig)
    with Image.open(OUTPUT / "convergence_u30.png") as image:
        assert image.size == (2100, 2880)
        image.verify()
    for path, expected in before.items():
        assert digest(Path(path)) == expected, path
    shutil.copy2(OUTPUT / "convergence_u30.pdf", ROOT / "paper" / "convergence_u30.pdf")
    report = {"source": str(SOURCE), "source_sha256": before,
              "output_size_mm": [88.9, 121.92], "layout": "single column; two vertically stacked panels", "png_dpi": 600,
              "episode_conversion": "episodes = environment steps / 512; complete episodes 1–292",
              "smoothing": "original 3-episode trailing mean, unchanged",
              "confidence_interval": "original five-seed 95% bootstrap intervals, unchanged",
              "panels": audit}
    (OUTPUT / "convergence_u30_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"png": str(OUTPUT / "convergence_u30.png"),
                      "paper_pdf": str(ROOT / "paper" / "convergence_u30.pdf"),
                      "original_files_unchanged": len(before)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
