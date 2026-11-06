"""Draw the U30 algorithm learning curves at IEEE single-column print size."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WIDTH_IN = 3.5
HEIGHT_IN = 2.82
LEGEND_GAP_PT = 5.04
sys.path.insert(0, str(ROOT))
from scripts.baseline_plot_config import method_visual_style
from scripts.paper_metrics import bootstrap_mean_ci
from scripts.plot_multiseed_training_figures import METHOD_NAMES, METHOD_ORDER, _History, _training_curve


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = ROOT / "results" / "multiseed_0907"
    output = ROOT / "figure" / "paper_candidate"
    output.mkdir(parents=True, exist_ok=True)
    source_paths = [source / "multiuser_reward_convergence.png"]
    source_paths += [source / f"seed{seed}/u30/comparison_summary.json" for seed in range(42, 47)]
    for method in METHOD_ORDER:
        source_paths += [source / f"seed{seed}/u30/learned_baselines/{method}/training_history.json"
                         for seed in range(42, 47)]
    before = {str(path): digest(path) for path in source_paths}
    series = []
    with plt.rc_context({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.linewidth": 0.65, "axes.labelpad": 3,
        "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "grid.alpha": 0.4, "grid.linestyle": "--", "grid.linewidth": 0.45,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None,
    }):
        fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), dpi=150)
        axis = fig.add_axes([0.17, 0.57 / HEIGHT_IN, 0.80, 1.77 / HEIGHT_IN])
        for method in METHOD_ORDER:
            curves = {}
            for seed in range(42, 47):
                path = source / f"seed{seed}/u30/learned_baselines/{method}/training_history.json"
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
                config = payload["config"]
                saved_users = config.get("num_users")
                if saved_users is None:
                    environment = json.loads((source / f"seed{seed}/u30/comparison_summary.json").read_text(encoding="utf-8"))
                    saved_users = environment["env_config"]["num_users"]
                assert int(saved_users) == 30
                length, budget = float(config["max_steps"]), float(config["total_timesteps"])
                assert length == 512 and budget == 150000
                history = _History(path, seed, 30, method, payload, length, budget, int(budget // length))
                curves[seed] = _training_curve(history, "mean_reward")
            common = sorted(set.intersection(*(set(curve) for curve in curves.values())))
            expected = np.arange(2, 293, 2) if method in {"han_mappo", "mappo_no_han"} else np.arange(1, 293)
            assert np.array_equal(common, expected), method
            statistics = np.asarray([bootstrap_mean_ci([curves[seed][episode] for seed in range(42, 47)])
                                     for episode in common])
            mean, low, high = statistics.T
            identity = method_visual_style(method)
            identity.update(linewidth=1.5 if method == "han_pdqn" else 1.05,
                            markersize=4 if method == "han_pdqn" else 2.8,
                            markeredgewidth=0.3, markevery=max(len(common) // 8, 1))
            axis.plot(common, mean, label=METHOD_NAMES[method], **identity,
                      zorder=5 if method == "han_pdqn" else 3)
            axis.fill_between(common, low, high, color=identity["color"], alpha=0.14, linewidth=0)
            series.append({"method": method, "training_seeds": [42, 43, 44, 45, 46],
                           "episodes": common, "mean_reward": mean.tolist(),
                           "ci_low": low.tolist(), "ci_high": high.tolist()})
        axis.set_xlabel("Training episodes")
        axis.set_ylabel("Mean episode reward")
        axis.set_xlim(0, 300)
        axis.set_xticks([0, 50, 100, 150, 200, 250, 300])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.grid(True)
        axis.set_axisbelow(True)
        handles, labels = axis.get_legend_handles_labels()
        legend_center = (axis.get_position().x0 + axis.get_position().x1) / 2
        legend_bottom = axis.get_position().y1 + LEGEND_GAP_PT / (72 * HEIGHT_IN)
        legend = fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(legend_center, legend_bottom), borderaxespad=0,
                   ncol=3, frameon=True, fontsize=7.5, borderpad=0.3,
                   handlelength=1.5, handletextpad=0.35, columnspacing=0.8,
                   labelspacing=0.35)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_gap_pt = (legend.get_window_extent(renderer).y0 - axis.bbox.y1) * 72 / fig.dpi
        for text in fig.findobj(matplotlib.text.Text):
            if not text.get_visible() or not text.get_text():
                continue
            bbox = text.get_window_extent(renderer)
            if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.x1 + 1 or bbox.y1 > fig.bbox.y1 + 1:
                raise RuntimeError(f"Clipped text: {text.get_text()}")
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"algorithm_convergence_u30.{suffix}", dpi=600,
                        bbox_inches=None, facecolor="white")
        plt.close(fig)
    with Image.open(output / "algorithm_convergence_u30.png") as image:
        assert image.size == (round(WIDTH_IN * 600), round(HEIGHT_IN * 600))
        image.verify()
    for path, original_hash in before.items():
        assert digest(Path(path)) == original_hash, path
    shutil.copy2(output / "algorithm_convergence_u30.pdf", ROOT / "paper" / "algorithm_convergence_u30.pdf")
    report = {"source": str(source), "num_users": 30,
              "size_mm": [WIDTH_IN * 25.4, HEIGHT_IN * 25.4],
              "legend": {"rows": 2, "columns": 3, "center_x_fraction": legend_center,
                         "gap_above_axes_points": legend_gap_pt},
              "source_sha256": before, "smoothing": "none; same raw episode rewards as original multiuser figure",
              "episode_conversion": "validated cumulative episodes, 512 environment steps per episode; partial and over-budget records excluded",
              "ppo_sampling": "two-episode means at native even episode coordinates; no interpolation",
              "confidence_interval": "95% percentile bootstrap of 5 training seeds; 2000 resamples; same helper as original figure",
              "series": series}
    (output / "algorithm_convergence_u30_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"paper_pdf": str(ROOT / "paper" / "algorithm_convergence_u30.pdf"),
                      "png": str(output / "algorithm_convergence_u30.png"),
                      "protected_files_unchanged": len(before),
                      "samples_per_method": {s["method"]: len(s["episodes"]) for s in series}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
