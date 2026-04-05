#!/usr/bin/env python3
"""
Reward weight sensitivity sweep for HAN-MAPPO training.

This script reuses scripts/train.py as the single training entrypoint, runs a
small set of predefined reward-weight configurations, and writes a compact
summary table for later comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"


@dataclass(frozen=True)
class WeightPreset:
    name: str
    delay: float
    energy: float
    handover: float
    load_balance: float
    qos: float

    def as_flags(self) -> List[str]:
        return [
            "--reward_delay_weight", str(self.delay),
            "--reward_energy_weight", str(self.energy),
            "--reward_handover_weight", str(self.handover),
            "--reward_load_balance_weight", str(self.load_balance),
            "--reward_qos_weight", str(self.qos),
        ]


PRESETS: Dict[str, WeightPreset] = {
    "balanced": WeightPreset("balanced", 1.0, 0.8, 0.5, 0.2, 0.3),
    "delay_focus": WeightPreset("delay_focus", 1.4, 0.4, 0.3, 0.1, 0.4),
    "energy_focus": WeightPreset("energy_focus", 0.8, 1.2, 0.3, 0.1, 0.3),
    "handover_focus": WeightPreset("handover_focus", 0.9, 0.5, 1.0, 0.3, 0.3),
    "coordination_focus": WeightPreset("coordination_focus", 0.9, 0.5, 0.7, 0.8, 0.3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="奖励权重敏感性实验脚本")
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["balanced", "delay_focus", "handover_focus", "coordination_focus"],
        help="要运行的预设名称列表",
    )
    parser.add_argument("--list_presets", action="store_true", help="仅列出可用预设")
    parser.add_argument("--python_exec", type=str, default=sys.executable, help="训练使用的 Python 解释器")
    parser.add_argument("--save_root", type=str, default="results/reward_sensitivity", help="实验结果根目录")
    parser.add_argument("--run_name", type=str, default=None, help="本次扫描目录名")
    parser.add_argument("--device", type=str, default="auto", help="训练设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_users", type=int, default=5, help="用户数量")
    parser.add_argument("--max_steps", type=int, default=1000, help="每个 episode 步数")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="每组权重训练总步数")
    parser.add_argument("--n_steps", type=int, default=2048, help="每轮 rollout 步数")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--eval_interval", type=int, default=10000, help="评估间隔")
    parser.add_argument("--eval_episodes", type=int, default=2, help="每次评估 episode 数")
    parser.add_argument("--graph_update_interval", type=int, default=20, help="图更新间隔")
    parser.add_argument("--han_hidden_dim", type=int, default=64, help="HAN 隐藏维度")
    parser.add_argument("--han_num_heads", type=int, default=4, help="HAN 注意力头数")
    parser.add_argument("--han_num_layers", type=int, default=2, help="HAN 层数")
    parser.add_argument("--dry_run", action="store_true", help="仅打印命令，不执行训练")
    parser.add_argument("--continue_on_error", action="store_true", help="单组失败后继续后续实验")
    return parser.parse_args()


def list_presets() -> None:
    print("Available presets:")
    for preset in PRESETS.values():
        print(
            f"- {preset.name}: "
            f"delay={preset.delay}, energy={preset.energy}, "
            f"handover={preset.handover}, load_balance={preset.load_balance}, qos={preset.qos}"
        )


def build_train_command(
    python_exec: str,
    preset: WeightPreset,
    save_path: Path,
    args: argparse.Namespace,
) -> List[str]:
    return [
        python_exec,
        str(TRAIN_SCRIPT),
        "--exp_name", f"reward_sensitivity_{preset.name}",
        "--device", args.device,
        "--seed", str(args.seed),
        "--num_users", str(args.num_users),
        "--max_steps", str(args.max_steps),
        "--total_timesteps", str(args.total_timesteps),
        "--n_steps", str(args.n_steps),
        "--learning_rate", str(args.learning_rate),
        "--batch_size", str(args.batch_size),
        "--eval_interval", str(args.eval_interval),
        "--eval_episodes", str(args.eval_episodes),
        "--graph_update_interval", str(args.graph_update_interval),
        "--han_hidden_dim", str(args.han_hidden_dim),
        "--han_num_heads", str(args.han_num_heads),
        "--han_num_layers", str(args.han_num_layers),
        "--save_path", str(save_path),
        *preset.as_flags(),
    ]


def load_summary(run_dir: Path) -> Dict[str, float]:
    history_path = run_dir / "training_history.json"
    if not history_path.exists():
        return {}

    history = json.loads(history_path.read_text(encoding="utf-8"))
    summary = history.get("summary", {})
    training = history.get("training", [])
    evaluation = history.get("evaluation", [])
    last_train = training[-1] if training else {}
    last_eval = evaluation[-1] if evaluation else {}

    return {
        "best_reward": summary.get("best_reward"),
        "last_mean_reward": last_train.get("mean_reward"),
        "last_rollout_mean_reward": last_train.get("rollout_mean_reward"),
        "last_eval_mean_reward": last_eval.get("eval_mean_reward"),
        "last_eval_std_reward": last_eval.get("eval_std_reward"),
        "last_eval_avg_delay": last_eval.get("avg_delay"),
        "last_eval_total_energy": last_eval.get("total_energy"),
        "last_eval_task_completion_rate": last_eval.get("task_completion_rate"),
        "last_eval_handover_success_rate": last_eval.get("handover_success_rate"),
        "last_eval_avg_load_balance_score": last_eval.get("avg_load_balance_score"),
    }


def write_summary(output_dir: Path, rows: List[Dict[str, object]]) -> None:
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if args.list_presets:
        list_presets()
        return

    unknown = [name for name in args.presets if name not in PRESETS]
    if unknown:
        raise SystemExit(f"未知预设: {', '.join(unknown)}")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (PROJECT_ROOT / args.save_root / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []

    for preset_name in args.presets:
        preset = PRESETS[preset_name]
        run_dir = output_dir / preset.name
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_train_command(args.python_exec, preset, run_dir, args)
        print(f"\n===== Running preset: {preset.name} =====")
        print(" ".join(cmd))

        row: Dict[str, object] = {
            "preset": preset.name,
            "delay_weight": preset.delay,
            "energy_weight": preset.energy,
            "handover_weight": preset.handover,
            "load_balance_weight": preset.load_balance,
            "qos_weight": preset.qos,
            "run_dir": str(run_dir),
            "status": "dry_run" if args.dry_run else "pending",
        }

        if not args.dry_run:
            try:
                subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
                row["status"] = "ok"
                row.update(load_summary(run_dir))
            except subprocess.CalledProcessError as exc:
                row["status"] = f"failed({exc.returncode})"
                if not args.continue_on_error:
                    summary_rows.append(row)
                    write_summary(output_dir, summary_rows)
                    raise

        summary_rows.append(row)
        write_summary(output_dir, summary_rows)

    print(f"\nSummary saved to: {output_dir}")


if __name__ == "__main__":
    main()
