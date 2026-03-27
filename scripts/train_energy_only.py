#!/usr/bin/env python3
"""
仅以能耗最小化为目标的训练脚本
================================

优化目标:
    min E[sum_t E_energy]

实现方式:
- 使用原始环境动力学（切换/卸载/队列处理）
- 覆盖 step 奖励为: reward = - (本步新增 total_energy)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

try:
    from scripts.train import TrainConfig, HANMAPPOTrainer
except ModuleNotFoundError:
    # 兼容 `python scripts/train_energy_only.py` 直接执行场景
    from train import TrainConfig, HANMAPPOTrainer
from src.environment.gym_env import LEOSatelliteEnv, EnvConfig


def generate_plots(save_path: str, window: int = 10):
    """训练/评估结束后生成可视化图表。"""
    try:
        from scripts.plot_results import generate_all_plots, setup_plot_style
    except ModuleNotFoundError:
        from plot_results import generate_all_plots, setup_plot_style

    history_file = Path(save_path) / "training_history.json"
    output_dir = Path(save_path) / "figures"

    if not history_file.exists():
        print(f"[WARNING] 找不到训练历史: {history_file}")
        return

    setup_plot_style()
    generate_all_plots(str(history_file), str(output_dir), window=window)
    print(f"图表已保存至: {output_dir}")


class EnergyOnlyEnv(LEOSatelliteEnv):
    """能耗单目标环境：奖励仅由能耗增量决定。"""

    def step(self, actions):
        prev_energy = float(self.stats.get("total_energy", 0.0))
        observation, _raw_reward, terminated, truncated, info = super().step(actions)

        energy_increment = max(float(self.stats.get("total_energy", 0.0)) - prev_energy, 0.0)
        objective_reward = -energy_increment

        info = dict(info)
        info["objective"] = "energy_only"
        info["energy_increment"] = energy_increment
        info["objective_reward"] = objective_reward

        return observation, objective_reward, terminated, truncated, info


class EnergyOnlyTrainer(HANMAPPOTrainer):
    """复用 HAN+MAPPO 管线，仅替换环境为能耗单目标环境。"""

    def _init_environment(self):
        self.logger.info("初始化能耗单目标环境...")

        env_config = EnvConfig(
            num_planes=self.config.num_planes,
            sats_per_plane=self.config.sats_per_plane,
            altitude_km=self.config.altitude_km,
            inclination_deg=self.config.inclination_deg,
            num_users=self.config.num_users,
            max_steps=self.config.max_steps,
            time_step_sec=self.config.time_step_sec,
            seed=self.config.seed,
            reward_delay_weight=0.0,
            reward_energy_weight=1.0,
            reward_handover_weight=0.0,
            reward_qos_weight=0.0,
        )

        self.env = EnergyOnlyEnv(env_config)

        # 与父类保持一致
        self.num_agents = self.config.num_users
        self.max_candidates = self.config.max_visible_sats
        self.raw_obs_dim = self.env.user_obs_dim
        self.han_out_dim = self.config.han_out_dim
        self.obs_dim = self.han_out_dim + 5
        self.global_state_dim = self.num_agents * self.obs_dim

        self.logger.info(f"  - 原始观测维度: {self.raw_obs_dim}")
        self.logger.info(f"  - HAN嵌入维度: {self.han_out_dim}")
        self.logger.info(f"  - 拼接后观测维度: {self.obs_dim}")
        self.logger.info(f"  - 全局状态维度: {self.global_state_dim}")


def parse_args():
    parser = argparse.ArgumentParser(description="LEO 能耗单目标训练")

    parser.add_argument("--exp_name", type=str, default="han_mappo_energy_only", help="实验名称")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="auto", help="设备 (cuda/cpu/auto)")

    parser.add_argument("--num_users", type=int, default=5, help="用户数量")
    parser.add_argument("--max_steps", type=int, default=1000, help="每episode最大步数")

    parser.add_argument("--total_timesteps", type=int, default=500000, help="总训练步数")
    parser.add_argument("--n_steps", type=int, default=2048, help="每次更新收集步数")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")

    parser.add_argument("--han_hidden_dim", type=int, default=64, help="HAN隐藏维度")
    parser.add_argument("--han_num_heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--han_num_layers", type=int, default=2, help="HAN层数")

    parser.add_argument("--save_path", type=str, default=None,
                        help="模型保存路径（显式指定时将直接使用）")
    parser.add_argument("--save_root", type=str, default="results/energy_only_train",
                        help="未指定save_path时的结果根目录，将自动创建时间戳子目录")
    parser.add_argument("--run_name", type=str, default=None,
                        help="运行名（用于自动目录名后缀，默认使用时间戳）")
    parser.add_argument("--load_path", type=str, default=None, help="加载检查点路径")
    parser.add_argument("--eval_interval", type=int, default=10000, help="评估间隔")
    parser.add_argument("--eval_episodes", type=int, default=2, help="每次评估episode数（默认2以提速）")
    parser.add_argument("--graph_update_interval", type=int, default=20, help="图重建间隔（步），增大可提速")
    parser.add_argument("--eval_only", action="store_true", help="仅评估，不训练")
    parser.add_argument("--no_plot", action="store_true", help="结束后不自动生成可视化图表")
    parser.add_argument("--plot_window", type=int, default=10, help="绘图平滑窗口大小")

    return parser.parse_args()


def build_config(args) -> TrainConfig:
    config = TrainConfig()
    config.exp_name = args.exp_name
    config.seed = args.seed
    config.device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and not torch.cuda.is_available():
        config.device = "cpu"

    config.num_users = args.num_users
    config.max_steps = args.max_steps
    config.total_timesteps = args.total_timesteps
    config.n_steps = args.n_steps
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.han_hidden_dim = args.han_hidden_dim
    config.han_num_heads = args.han_num_heads
    config.han_num_layers = args.han_num_layers
    run_suffix = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_save_path = Path(args.save_root) / f"{args.exp_name}_{run_suffix}"
    config.save_path = args.save_path if args.save_path else str(auto_save_path)
    config.load_path = args.load_path
    config.eval_interval = args.eval_interval
    config.eval_episodes = args.eval_episodes
    config.graph_update_interval = args.graph_update_interval
    return config


def main():
    args = parse_args()
    config = build_config(args)
    print(f"[INFO] 本次结果路径: {config.save_path}")

    trainer = EnergyOnlyTrainer(config)

    if config.load_path:
        trainer.load_checkpoint(config.load_path)

    if args.eval_only:
        trainer._evaluate()
    else:
        trainer.train()

    if not args.no_plot:
        generate_plots(config.save_path, window=args.plot_window)


if __name__ == "__main__":
    main()
