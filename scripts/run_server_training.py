#!/usr/bin/env python3
"""
LEO HAN+MAPPO 服务器端完整训练脚本
====================================

适用于: 配备 NVIDIA RTX 3090 (24GB VRAM) 的 Linux 服务器

【使用流程】
步骤1: 将整个项目上传到服务器
步骤2: 安装依赖
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install numpy matplotlib scipy gymnasium

步骤3: 运行训练
    # 后台运行（推荐），输出日志到文件
    nohup python scripts/run_server_training.py > train_output.log 2>&1 &

    # 或用 tmux/screen
    tmux new -s leo_train
    python scripts/run_server_training.py

步骤4: 训练完成后生成图表
    python scripts/plot_results.py --input results/full_train_v3/training_history.json --output results/full_train_v3/figures

步骤5: 下载结果
    scp -r user@server:~/LEO_switch/results ./results_from_server/

【参数说明】
- 3090 (24GB) 上本模型很小（< 1M 参数），瓶颈在 CPU 端环境仿真
- total_timesteps = 1_000_000 预计训练 2-4 小时（取决于 CPU 速度）
- 多用户 (10~20) 会增加每步计算量，但对 GPU 影响不大
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

# ---- 确保路径正确 ----
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import numpy as np
import torch


# ============================================================
#  训练配置方案（根据需求选择）
# ============================================================

# 方案 A: 标准训练（推荐首次使用，~2小时）
STANDARD_CONFIG = {
    'exp_name':         'han_mappo_standard',
    'seed':             42,
    'device':           'cuda',          # 3090 自动使用 CUDA

    # 环境
    'num_users':        10,              # 10 个用户（增加竞争）
    'max_steps':        2000,            # 每 episode 2000 步（更长的决策序列）

    # HAN
    'han_hidden_dim':   64,
    'han_out_dim':      64,
    'han_num_heads':    4,
    'han_num_layers':   2,

    # Actor / Critic
    'actor_hidden_dims':  (256, 128),
    'critic_hidden_dims': (256, 256, 128),

    # MAPPO
    'learning_rate':    3e-4,            # v4: 提升学习率，增大策略更新步长
    'gamma':            0.99,
    'gae_lambda':       0.95,
    'clip_range':       0.2,
    'entropy_coef':     0.01,            # v4: 降低熵系数，Beta分布下0.05过大
    'n_epochs':         10,              # v4: 增加epoch数，充分利用数据
    'batch_size':       64,              # v4: 减小batch，增加mini-batch数量

    # 训练
    'total_timesteps':  1_000_000,       # 100万步
    'n_steps':          2048,            # 每轮收集 2048 步
    'eval_interval':    20_000,          # 每 2 万步评估一次
    'eval_episodes':    5,               # 评估 5 个 episode
    'save_interval':    100_000,         # 每 10 万步保存一次
    'log_interval':     1,               # 每次更新都打印日志
    'early_stop_patience': 50,           # v4: 放宽早停，给策略更多学习时间

    # 路径
    'save_path':        'results/full_train_v4',
    'log_path':         'results/logs',
}

# 方案 B: 大规模用户训练（~4-6小时）
LARGE_SCALE_CONFIG = {
    **STANDARD_CONFIG,
    'exp_name':         'han_mappo_large',
    'num_users':        20,              # 20 个用户
    'max_steps':        3000,            # 更长 episode
    'total_timesteps':  2_000_000,       # 200万步
    'n_steps':          4096,            # 更大的 rollout buffer
    'batch_size':       256,             # 更大的批大小
    'eval_interval':    50_000,
    'save_interval':    200_000,
    'early_stop_patience': 50,
    'save_path':        'results/large_train',
}

# 方案 C: 快速验证（~15-30分钟）
QUICK_TEST_CONFIG = {
    **STANDARD_CONFIG,
    'exp_name':         'han_mappo_quick',
    'num_users':        5,
    'max_steps':        1000,
    'total_timesteps':  100_000,         # 10万步
    'n_steps':          1024,
    'eval_interval':    10_000,
    'save_interval':    50_000,
    'early_stop_patience': 15,
    'save_path':        'results/quick_test',
}

# 方案 D: 多种子对比实验（用于论文）
MULTI_SEED_SEEDS = [42, 123, 456, 789, 2024]


# ============================================================
#  主训练函数
# ============================================================

def run_training(config_dict: dict):
    """执行单次训练"""
    from scripts.train import TrainConfig, HANMAPPOTrainer

    config = TrainConfig()

    # 更新配置
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # 设备检测
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA 不可用，回退到 CPU")
        config.device = 'cpu'

    # 打印系统信息
    print("=" * 70)
    print(f"  实验: {config.exp_name}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  设备: {config.device}")
    if config.device == 'cuda':
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  用户数: {config.num_users}")
    print(f"  总步数: {config.total_timesteps:,}")
    print(f"  每轮步数: {config.n_steps}")
    print(f"  预计更新数: {config.total_timesteps // config.n_steps}")
    print(f"  保存路径: {config.save_path}")
    print("=" * 70)

    # 创建训练器并开始训练
    trainer = HANMAPPOTrainer(config)
    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    print(f"\n训练完成! 耗时: {elapsed/3600:.2f} 小时")
    print(f"历史文件: {config.save_path}/training_history.json")

    return config.save_path


def run_multi_seed(base_config: dict, seeds: list):
    """多种子对比实验"""
    history_paths = []

    for i, seed in enumerate(seeds):
        cfg = base_config.copy()
        cfg['seed'] = seed
        cfg['exp_name'] = f"{base_config['exp_name']}_seed{seed}"
        cfg['save_path'] = f"results/multi_seed/seed_{seed}"

        print(f"\n{'#' * 70}")
        print(f"#  多种子实验 [{i+1}/{len(seeds)}]  seed = {seed}")
        print(f"{'#' * 70}")

        save_path = run_training(cfg)
        history_paths.append(f"{save_path}/training_history.json")

    # 保存路径列表供后续绘图
    meta = {
        'experiment': 'multi_seed',
        'seeds': seeds,
        'history_paths': history_paths,
    }
    meta_path = Path("results/multi_seed/experiment_meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n所有种子训练完成!")
    print(f"元数据: {meta_path}")
    return history_paths


# ============================================================
#  自动生成图表
# ============================================================

def generate_plots(save_path: str, window: int = 10):
    """训练完成后自动生成图表"""
    from scripts.plot_results import generate_all_plots, setup_plot_style

    history_file = f"{save_path}/training_history.json"
    output_dir = f"{save_path}/figures"

    if not Path(history_file).exists():
        print(f"[WARNING] 找不到训练历史: {history_file}")
        return

    setup_plot_style()
    generate_all_plots(history_file, output_dir, window=window)
    print(f"图表已保存至: {output_dir}")


def generate_comparison_plots(history_paths: list, window: int = 10):
    """多实验对比图表"""
    from scripts.plot_results import (
        setup_plot_style, plot_comparison, plot_comparison_metrics
    )

    output_dir = Path("results/multi_seed/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_plot_style()
    plot_comparison(history_paths, window, output_dir)
    plot_comparison_metrics(history_paths, window, output_dir)
    print(f"对比图表已保存至: {output_dir}")


# ============================================================
#  入口
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='LEO HAN+MAPPO 服务器端训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
配置方案:
  standard    标准训练 (10用户, 100万步, ~2h)      [推荐首次]
  large       大规模训练 (20用户, 200万步, ~4-6h)
  quick       快速验证 (5用户, 10万步, ~15-30min)
  multi_seed  多种子对比 (5个种子, ~10h)

示例:
  python scripts/run_server_training.py --plan standard
  python scripts/run_server_training.py --plan quick
  python scripts/run_server_training.py --plan multi_seed
  python scripts/run_server_training.py --plan standard --users 8 --steps 2000000
  python scripts/run_server_training.py --plot_only results/full_train
        """)

    parser.add_argument('--plan', type=str, default='standard',
                        choices=['standard', 'large', 'quick', 'multi_seed'],
                        help='训练方案 (默认: standard)')
    parser.add_argument('--users', type=int, default=None,
                        help='覆盖用户数')
    parser.add_argument('--steps', type=int, default=None,
                        help='覆盖总步数')
    parser.add_argument('--seed', type=int, default=None,
                        help='覆盖随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--window', type=int, default=10,
                        help='绘图滑动窗口大小')
    parser.add_argument('--no_plot', action='store_true',
                        help='训练后不自动生成图表')
    parser.add_argument('--plot_only', type=str, default=None,
                        help='仅生成图表（指定结果目录）')

    args = parser.parse_args()

    # ---------- 仅绘图模式 ----------
    if args.plot_only:
        generate_plots(args.plot_only, window=args.window)
        sys.exit(0)

    # ---------- 选择配置 ----------
    configs = {
        'standard':   STANDARD_CONFIG,
        'large':      LARGE_SCALE_CONFIG,
        'quick':      QUICK_TEST_CONFIG,
    }

    if args.plan == 'multi_seed':
        base = STANDARD_CONFIG.copy()
        if args.users:
            base['num_users'] = args.users
        if args.steps:
            base['total_timesteps'] = args.steps
        base['device'] = args.device

        paths = run_multi_seed(base, MULTI_SEED_SEEDS)

        if not args.no_plot:
            # 每个种子单独画图
            for path in paths:
                save_dir = str(Path(path).parent)
                generate_plots(save_dir, window=args.window)
            # 对比图
            generate_comparison_plots(paths, window=args.window)
    else:
        cfg = configs[args.plan].copy()
        if args.users:
            cfg['num_users'] = args.users
        if args.steps:
            cfg['total_timesteps'] = args.steps
        if args.seed:
            cfg['seed'] = args.seed
        cfg['device'] = args.device

        save_path = run_training(cfg)

        if not args.no_plot:
            generate_plots(save_path, window=args.window)

    print("\n" + "=" * 70)
    print("  全部任务完成!")
    print("=" * 70)
