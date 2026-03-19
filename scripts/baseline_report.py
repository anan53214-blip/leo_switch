"""
基线对比报告与绘图脚本

运行多个基线策略，输出汇总结果，并生成对比图。

示例：
    python scripts/baseline_report.py --episodes 5 --max_steps 200
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from baseline_eval import evaluate_baseline
from src.environment.gym_env import EnvConfig


def _plot_bar(ax, labels: List[str], values: List[float], title: str, ylabel: str):
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    colors = [palette[i % len(palette)] for i in range(len(values))]
    ax.bar(labels, values, color=colors, edgecolor="#2F2F2F", linewidth=0.5)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def run_baselines(
    baselines: List[str],
    episodes: int,
    offload_ratio: float,
    max_steps: int,
    seed: int
) -> List[Dict]:
    env_config = EnvConfig(
        num_users=5,
        max_steps=max_steps,
        seed=seed
    )

    results = []
    for baseline in baselines:
        result = evaluate_baseline(
            baseline=baseline,
            episodes=episodes,
            env_config=env_config,
            offload_ratio=offload_ratio,
            max_steps=max_steps
        )
        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="基线对比报告与绘图")
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=["random", "stay", "max_elev", "max_rvt", "min_distance"],
        help="基线策略列表"
    )
    parser.add_argument("--episodes", type=int, default=5, help="评估episode数")
    parser.add_argument("--offload_ratio", type=float, default=0.5, help="固定卸载比例")
    parser.add_argument("--max_steps", type=int, default=200, help="每个episode步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output_dir", type=str, default="results/baselines", help="结果保存目录")
    args = parser.parse_args()

    results = run_baselines(
        baselines=args.baselines,
        episodes=args.episodes,
        offload_ratio=args.offload_ratio,
        max_steps=args.max_steps,
        seed=args.seed
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"baseline_summary_{timestamp}.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    try:
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8")
    except ImportError as exc:
        print("未检测到 matplotlib，无法生成图表。")
        print(f"汇总结果已保存: {summary_path}")
        raise exc

    labels = [r["baseline"] for r in results]
    mean_rewards = [r["mean_reward"] for r in results]
    mean_delays = [r["mean_delay"] for r in results]
    mean_energy = [r["mean_energy"] for r in results]
    handover_rates = [r["handover_success_rate"] for r in results]
    completion_rates = [r["task_completion_rate"] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150)
    axes = axes.flatten()

    _plot_bar(axes[0], labels, mean_rewards, "Mean Reward", "Reward")
    _plot_bar(axes[1], labels, mean_delays, "Mean Delay", "Delay")
    _plot_bar(axes[2], labels, mean_energy, "Mean Energy", "Energy")
    _plot_bar(axes[3], labels, handover_rates, "Handover Success Rate", "Rate")
    _plot_bar(axes[4], labels, completion_rates, "Task Completion Rate", "Rate")
    axes[5].axis("off")

    fig.tight_layout(pad=1.2)

    plot_path = output_dir / f"baseline_comparison_{timestamp}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"汇总结果已保存: {summary_path}")
    print(f"对比图已保存: {plot_path}")


if __name__ == "__main__":
    main()
