"""
LEO卫星网络 HAN+MAPPO 训练结果可视化
=====================================

从 training_history.json 读取训练过程数据，生成论文级别的图表：

1. 奖励收敛曲线（含滑动平均 + 置信区间）
2. Actor / Critic Loss 曲线
3. 策略熵 & KL 散度曲线
4. 切换成功率 & 任务完成率曲线
5. 平均时延 & 累积能耗曲线
6. 评估奖励曲线（eval episodes）
7. 综合仪表盘（Dashboard）
8. 可选：多实验对比

【使用方法】
```bash
# 基本使用：可视化最近一次训练
python scripts/plot_results.py

# 指定历史文件
python scripts/plot_results.py --input results/models/training_history.json

# 指定输出目录
python scripts/plot_results.py --output results/figures

# 多实验对比
python scripts/plot_results.py --compare results/exp1/training_history.json results/exp2/training_history.json

# 自定义滑动窗口
python scripts/plot_results.py --window 50
```
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # 无头模式，支持服务器环境
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import FuncFormatter, MaxNLocator
except ImportError:
    print("错误: 需要安装 matplotlib。请运行: pip install matplotlib")
    sys.exit(1)

try:
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# 全局绘图样式（论文级别）
# ============================================================

# 颜色方案（色盲友好）
COLORS = {
    'primary':    '#2196F3',  # 蓝色
    'secondary':  '#FF5722',  # 橙红色
    'success':    '#4CAF50',  # 绿色
    'warning':    '#FF9800',  # 橙色
    'danger':     '#F44336',  # 红色
    'info':       '#00BCD4',  # 青色
    'purple':     '#9C27B0',  # 紫色
    'dark':       '#37474F',  # 深灰色
    'fill_alpha': 0.15,       # 置信区间填充透明度
}

# 多实验对比色板
PALETTE = ['#2196F3', '#FF5722', '#4CAF50', '#FF9800', '#9C27B0',
           '#00BCD4', '#795548', '#607D8B', '#E91E63', '#3F51B5']


def setup_plot_style():
    """配置论文级别的全局绘图样式"""
    plt.rcParams.update({
        # 字体
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        # 线条
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        # 图形
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        # 网格
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        # 边框
        'axes.spines.top': False,
        'axes.spines.right': False,
        # 图例
        'legend.framealpha': 0.8,
        'legend.edgecolor': '0.8',
    })


# ============================================================
# 数据处理工具
# ============================================================

def smooth(data: np.ndarray, window: int = 10) -> np.ndarray:
    """滑动平均平滑"""
    if len(data) < window:
        return data
    if HAS_SCIPY:
        return uniform_filter1d(data.astype(float), size=window)
    else:
        # 手动实现滑动平均
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode='same')


def compute_confidence_band(data: np.ndarray, window: int = 10
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算滑动平均及置信区间
    
    Returns:
        (mean, lower, upper)
    """
    mean = smooth(data, window)
    # 滑动标准差
    n = len(data)
    std = np.zeros(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        std[i] = np.std(data[lo:hi])
    
    return mean, mean - std, mean + std


def load_history(path: str) -> Dict:
    """加载训练历史 JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_series(records: List[Dict], key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从训练记录中提取某个指标的时间序列
    
    Returns:
        (steps数组, values数组)
    """
    steps = np.array([r['total_steps'] for r in records])
    values = np.array([r.get(key, 0) for r in records], dtype=float)
    return steps, values


def format_steps(x, _):
    """将步数格式化为 K/M 单位"""
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    else:
        return f'{x:.0f}'


# ============================================================
# 单独图表绘制函数
# ============================================================

def plot_reward_curve(records: List[Dict], window: int, save_dir: Path):
    """
    图1: 奖励收敛曲线
    - 原始 episode 奖励（浅色）
    - 滑动平均（深色）
    - 置信区间（阴影）
    """
    steps, rewards = extract_series(records, 'recent_mean_reward')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 原始数据（浅色散点）
    ax.plot(steps, rewards, alpha=0.25, color=COLORS['primary'], linewidth=0.8,
            label='Raw Reward')
    
    # 滑动平均 + 置信区间
    mean, lower, upper = compute_confidence_band(rewards, window)
    ax.fill_between(steps, lower, upper, alpha=COLORS['fill_alpha'],
                    color=COLORS['primary'])
    ax.plot(steps, mean, color=COLORS['primary'], linewidth=2.0,
            label=f'Moving Avg (w={window})')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Reward Convergence')
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax.legend(loc='lower right')
    
    fig.tight_layout()
    fig.savefig(save_dir / 'reward_curve.png')
    fig.savefig(save_dir / 'reward_curve.pdf')
    plt.close(fig)
    print("  ✓ reward_curve.png / .pdf")


def plot_loss_curves(records: List[Dict], window: int, save_dir: Path):
    """
    图2: Actor Loss & Critic Loss 双轴曲线
    """
    steps, actor_loss = extract_series(records, 'actor_loss')
    _, critic_loss = extract_series(records, 'critic_loss')
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Actor Loss (左轴)
    color_a = COLORS['primary']
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Actor Loss', color=color_a)
    ax1.plot(steps, smooth(actor_loss, window), color=color_a, linewidth=1.8,
             label='Actor Loss')
    ax1.tick_params(axis='y', labelcolor=color_a)
    ax1.xaxis.set_major_formatter(FuncFormatter(format_steps))
    
    # Critic Loss (右轴)
    ax2 = ax1.twinx()
    color_c = COLORS['secondary']
    ax2.set_ylabel('Critic Loss', color=color_c)
    ax2.plot(steps, smooth(critic_loss, window), color=color_c, linewidth=1.8,
             label='Critic Loss')
    ax2.tick_params(axis='y', labelcolor=color_c)
    ax2.spines['right'].set_visible(True)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.set_title('Actor & Critic Loss')
    fig.tight_layout()
    fig.savefig(save_dir / 'loss_curves.png')
    fig.savefig(save_dir / 'loss_curves.pdf')
    plt.close(fig)
    print("  ✓ loss_curves.png / .pdf")


def plot_entropy_kl(records: List[Dict], window: int, save_dir: Path):
    """
    图3: 策略熵 & KL 散度
    """
    steps, entropy = extract_series(records, 'entropy')
    _, kl_div = extract_series(records, 'kl_divergence')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # 策略熵
    ax1.plot(steps, smooth(entropy, window), color=COLORS['purple'], linewidth=1.8)
    ax1.fill_between(steps,
                     smooth(entropy, window) - 0.5 * np.std(entropy),
                     smooth(entropy, window) + 0.5 * np.std(entropy),
                     alpha=COLORS['fill_alpha'], color=COLORS['purple'])
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Policy Entropy')
    ax1.set_title('Policy Entropy')
    ax1.xaxis.set_major_formatter(FuncFormatter(format_steps))
    
    # KL 散度
    ax2.plot(steps, smooth(kl_div, window), color=COLORS['warning'], linewidth=1.8)
    ax2.axhline(y=0.01, color='gray', linestyle=':', alpha=0.6, label='target KL=0.01')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('KL Divergence')
    ax2.set_title('Approx. KL Divergence')
    ax2.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax2.legend()
    
    fig.tight_layout()
    fig.savefig(save_dir / 'entropy_kl.png')
    fig.savefig(save_dir / 'entropy_kl.pdf')
    plt.close(fig)
    print("  ✓ entropy_kl.png / .pdf")


def plot_handover_task(records: List[Dict], window: int, save_dir: Path):
    """
    图4: 切换成功率 & 任务完成率
    """
    steps, ho_rate = extract_series(records, 'handover_success_rate')
    _, task_rate = extract_series(records, 'task_completion_rate')
    _, violations = extract_series(records, 'deadline_violations')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # 左图：成功率
    ax1.plot(steps, smooth(ho_rate * 100, window), color=COLORS['primary'],
             linewidth=1.8, label='Handover Success')
    ax1.plot(steps, smooth(task_rate * 100, window), color=COLORS['success'],
             linewidth=1.8, label='Task Completion')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Rate (%)')
    ax1.set_title('Handover Success & Task Completion Rate')
    ax1.set_ylim(-5, 105)
    ax1.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax1.legend()
    
    # 右图：截止期违反次数
    ax2.plot(steps, smooth(violations, window), color=COLORS['danger'],
             linewidth=1.8)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Deadline Violations')
    ax2.set_title('Deadline Violations (per Episode)')
    ax2.xaxis.set_major_formatter(FuncFormatter(format_steps))
    
    fig.tight_layout()
    fig.savefig(save_dir / 'handover_task_rate.png')
    fig.savefig(save_dir / 'handover_task_rate.pdf')
    plt.close(fig)
    print("  ✓ handover_task_rate.png / .pdf")


def plot_delay_energy(records: List[Dict], window: int, save_dir: Path):
    """
    图5: 平均时延 & 能耗曲线
    """
    steps, avg_delay = extract_series(records, 'avg_delay')
    _, total_energy = extract_series(records, 'total_energy')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # 平均时延
    delay_ms = avg_delay * 1000  # 转毫秒
    ax1.plot(steps, smooth(delay_ms, window), color=COLORS['info'], linewidth=1.8)
    mean_d, lower_d, upper_d = compute_confidence_band(delay_ms, window)
    ax1.fill_between(steps, lower_d, upper_d, alpha=COLORS['fill_alpha'],
                     color=COLORS['info'])
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Avg Delay (ms)')
    ax1.set_title('Average Task Delay')
    ax1.xaxis.set_major_formatter(FuncFormatter(format_steps))
    
    # 能耗
    ax2.plot(steps, smooth(total_energy, window), color=COLORS['warning'], linewidth=1.8)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Total Energy (J)')
    ax2.set_title('Energy Consumption per Episode')
    ax2.xaxis.set_major_formatter(FuncFormatter(format_steps))
    
    fig.tight_layout()
    fig.savefig(save_dir / 'delay_energy.png')
    fig.savefig(save_dir / 'delay_energy.pdf')
    plt.close(fig)
    print("  ✓ delay_energy.png / .pdf")


def plot_eval_curve(eval_records: List[Dict], save_dir: Path):
    """
    图6: 评估奖励曲线（带误差棒）
    """
    if not eval_records:
        print("  ⚠ 无评估数据，跳过 eval_curve")
        return
    
    steps = np.array([r['total_steps'] for r in eval_records])
    means = np.array([r['eval_mean_reward'] for r in eval_records])
    stds = np.array([r['eval_std_reward'] for r in eval_records])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.errorbar(steps, means, yerr=stds, fmt='o-', color=COLORS['success'],
                capsize=3, capthick=1.2, linewidth=1.8, markersize=5,
                label='Eval Reward (mean +/- std)')
    ax.fill_between(steps, means - stds, means + stds,
                    alpha=COLORS['fill_alpha'], color=COLORS['success'])
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Eval Reward')
    ax.set_title('Evaluation Reward')
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(save_dir / 'eval_curve.png')
    fig.savefig(save_dir / 'eval_curve.pdf')
    plt.close(fig)
    print("  ✓ eval_curve.png / .pdf")


def plot_clip_fraction(records: List[Dict], window: int, save_dir: Path):
    """
    图7: PPO Clip Fraction
    """
    steps, clip_frac = extract_series(records, 'clip_fraction')
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(steps, smooth(clip_frac, window), color=COLORS['dark'], linewidth=1.8)
    ax.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5, label='clip_range=0.2')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Clip Fraction')
    ax.set_title('PPO Clip Fraction')
    ax.set_ylim(-0.02, max(0.5, np.max(clip_frac) * 1.1))
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(save_dir / 'clip_fraction.png')
    fig.savefig(save_dir / 'clip_fraction.pdf')
    plt.close(fig)
    print("  ✓ clip_fraction.png / .pdf")


# ============================================================
# 综合仪表盘
# ============================================================

def plot_dashboard(records: List[Dict], eval_records: List[Dict],
                   window: int, save_dir: Path, summary: Dict):
    """
    图8: 6合1综合仪表盘
    
    布局:
    ┌──────────┬──────────┬──────────┐
    │  奖励曲线 │ Loss曲线 │  熵 & KL  │
    ├──────────┼──────────┼──────────┤
    │ 切换/任务 │ 延迟/能耗 │  评估曲线  │
    └──────────┴──────────┴──────────┘
    """
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)
    
    steps, rewards = extract_series(records, 'recent_mean_reward')
    _, actor_loss = extract_series(records, 'actor_loss')
    _, critic_loss = extract_series(records, 'critic_loss')
    _, entropy = extract_series(records, 'entropy')
    _, kl_div = extract_series(records, 'kl_divergence')
    _, ho_rate = extract_series(records, 'handover_success_rate')
    _, task_rate = extract_series(records, 'task_completion_rate')
    _, avg_delay = extract_series(records, 'avg_delay')
    _, total_energy = extract_series(records, 'total_energy')
    
    # ---- (0,0) 奖励曲线 ----
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(steps, rewards, alpha=0.2, color=COLORS['primary'], linewidth=0.6)
    mean_r = smooth(rewards, window)
    ax.plot(steps, mean_r, color=COLORS['primary'], linewidth=2)
    ax.set_title('Reward Convergence')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Reward')
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    
    # ---- (0,1) Loss 曲线 ----
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(steps, smooth(actor_loss, window), color=COLORS['primary'],
            linewidth=1.5, label='Actor')
    ax.plot(steps, smooth(critic_loss, window), color=COLORS['secondary'],
            linewidth=1.5, label='Critic')
    ax.set_title('Loss')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax.legend(fontsize=9)
    
    # ---- (0,2) 熵 & KL ----
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(steps, smooth(entropy, window), color=COLORS['purple'],
            linewidth=1.5, label='Entropy')
    ax2 = ax.twinx()
    ax2.plot(steps, smooth(kl_div, window), color=COLORS['warning'],
             linewidth=1.5, label='KL Div')
    ax2.spines['right'].set_visible(True)
    ax.set_title('Entropy & KL')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Entropy', color=COLORS['purple'])
    ax2.set_ylabel('KL', color=COLORS['warning'])
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    
    # ---- (1,0) 切换 & 任务率 ----
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(steps, smooth(ho_rate * 100, window), color=COLORS['primary'],
            linewidth=1.5, label='Handover Success')
    ax.plot(steps, smooth(task_rate * 100, window), color=COLORS['success'],
            linewidth=1.5, label='Task Completion')
    ax.set_title('Handover & Task Rate')
    ax.set_xlabel('Steps')
    ax.set_ylabel('%')
    ax.set_ylim(-5, 105)
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax.legend(fontsize=9)
    
    # ---- (1,1) 延迟 & 能耗 ----
    ax = fig.add_subplot(gs[1, 1])
    delay_ms = avg_delay * 1000
    ax.plot(steps, smooth(delay_ms, window), color=COLORS['info'],
            linewidth=1.5, label='Delay (ms)')
    ax_e = ax.twinx()
    ax_e.plot(steps, smooth(total_energy, window), color=COLORS['warning'],
              linewidth=1.5, label='Energy (J)')
    ax_e.spines['right'].set_visible(True)
    ax.set_title('Delay & Energy')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Delay (ms)', color=COLORS['info'])
    ax_e.set_ylabel('Energy (J)', color=COLORS['warning'])
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_e.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    
    # ---- (1,2) 评估曲线 / 训练摘要 ----
    ax = fig.add_subplot(gs[1, 2])
    if eval_records:
        eval_steps = np.array([r['total_steps'] for r in eval_records])
        eval_means = np.array([r['eval_mean_reward'] for r in eval_records])
        eval_stds = np.array([r['eval_std_reward'] for r in eval_records])
        ax.errorbar(eval_steps, eval_means, yerr=eval_stds, fmt='o-',
                    color=COLORS['success'], capsize=3, linewidth=1.5, markersize=4)
        ax.set_title('Eval Reward')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Eval Reward')
        ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    else:
        # 无评估数据时显示训练摘要
        ax.axis('off')
        summary_text = (
            f"Training Summary\n"
            f"{'─' * 30}\n"
            f"Total Steps:    {summary.get('total_steps', 0):,}\n"
            f"Total Episodes: {summary.get('total_episodes', 0):,}\n"
            f"Best Reward:    {summary.get('best_reward', 0):.2f}\n"
            f"Training Time:  {summary.get('training_time_sec', 0)/3600:.2f} h\n"
            f"Final Reward:   {float(rewards[-1]) if len(rewards) > 0 else 0:.2f}"
        )
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#f5f5f5', alpha=0.8))
    
    fig.suptitle('HAN+MAPPO LEO Satellite Network Training Dashboard', fontsize=16, fontweight='bold', y=0.98)
    fig.savefig(save_dir / 'dashboard.png')
    fig.savefig(save_dir / 'dashboard.pdf')
    plt.close(fig)
    print("  ✓ dashboard.png / .pdf")


# ============================================================
# 多实验对比
# ============================================================

def plot_comparison(history_paths: List[str], window: int, save_dir: Path):
    """
    多实验奖励收敛曲线对比
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, path in enumerate(history_paths):
        data = load_history(path)
        records = data.get('training', [])
        if not records:
            print(f"  ⚠ 无训练数据: {path}")
            continue
        
        steps, rewards = extract_series(records, 'recent_mean_reward')
        color = PALETTE[i % len(PALETTE)]
        
        # 提取实验名
        config = data.get('config', {})
        label = config.get('exp_name', Path(path).parent.name)
        
        ax.plot(steps, rewards, alpha=0.15, color=color, linewidth=0.5)
        mean_r = smooth(rewards, window)
        ax.plot(steps, mean_r, color=color, linewidth=2.0, label=label)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Multi-Experiment Reward Comparison')
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax.legend(loc='lower right')
    
    fig.tight_layout()
    fig.savefig(save_dir / 'comparison.png')
    fig.savefig(save_dir / 'comparison.pdf')
    plt.close(fig)
    print("  ✓ comparison.png / .pdf")


def plot_comparison_metrics(history_paths: List[str], window: int, save_dir: Path):
    """
    多实验关键指标对比（切换成功率、任务完成率、时延、能耗）
    """
    metrics = [
        ('handover_success_rate', 'Handover Success Rate (%)', 100),
        ('task_completion_rate',  'Task Completion Rate (%)', 100),
        ('avg_delay',             'Avg Delay (ms)',  1000),
        ('total_energy',          'Total Energy (J)',      1),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    
    for ax, (key, ylabel, scale) in zip(axes, metrics):
        for i, path in enumerate(history_paths):
            data = load_history(path)
            records = data.get('training', [])
            if not records:
                continue
            
            steps, values = extract_series(records, key)
            color = PALETTE[i % len(PALETTE)]
            label = data.get('config', {}).get('exp_name', Path(path).parent.name)
            
            ax.plot(steps, smooth(values * scale, window), color=color,
                    linewidth=1.8, label=label)
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
        ax.legend(fontsize=8)
    
    fig.suptitle('Multi-Experiment Metrics Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_dir / 'comparison_metrics.png')
    fig.savefig(save_dir / 'comparison_metrics.pdf')
    plt.close(fig)
    print("  ✓ comparison_metrics.png / .pdf")


# ============================================================
# 主入口
# ============================================================

def generate_all_plots(history_path: str, output_dir: str, window: int = 10):
    """生成所有单实验图表"""
    print(f"\n{'='*60}")
    print(f"LEO HAN+MAPPO 训练结果可视化")
    print(f"{'='*60}")
    print(f"  输入: {history_path}")
    print(f"  输出: {output_dir}")
    print(f"  滑动窗口: {window}")
    print()
    
    # 加载数据
    data = load_history(history_path)
    records = data.get('training', [])
    eval_records = data.get('evaluation', [])
    summary = data.get('summary', {})
    config = data.get('config', {})
    
    if not records:
        print("错误: 训练历史为空，无法生成图表。")
        return
    
    print(f"  训练记录: {len(records)} 条")
    print(f"  评估记录: {len(eval_records)} 条")
    print(f"  总步数:   {summary.get('total_steps', 'N/A')}")
    print(f"  最佳奖励: {summary.get('best_reward', 'N/A')}")
    print()
    
    # 创建输出目录
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成各图
    print("生成图表:")
    plot_reward_curve(records, window, save_dir)
    plot_loss_curves(records, window, save_dir)
    plot_entropy_kl(records, window, save_dir)
    plot_handover_task(records, window, save_dir)
    plot_delay_energy(records, window, save_dir)
    plot_eval_curve(eval_records, save_dir)
    plot_clip_fraction(records, window, save_dir)
    plot_dashboard(records, eval_records, window, save_dir, summary)
    
    # 保存配置信息
    info_path = save_dir / 'plot_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump({
            'source': str(history_path),
            'window': window,
            'num_training_records': len(records),
            'num_eval_records': len(eval_records),
            'config': config,
            'summary': summary,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n所有图表已保存至: {save_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='LEO HAN+MAPPO 训练结果可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/plot_results.py
  python scripts/plot_results.py --input results/models/training_history.json
  python scripts/plot_results.py --output results/figures --window 20
  python scripts/plot_results.py --compare results/exp1/training_history.json results/exp2/training_history.json
        """
    )
    
    parser.add_argument('--input', '-i', type=str,
                        default='results/models/training_history.json',
                        help='训练历史 JSON 文件路径 (默认: results/models/training_history.json)')
    parser.add_argument('--output', '-o', type=str,
                        default='results/figures',
                        help='图表输出目录 (默认: results/figures)')
    parser.add_argument('--window', '-w', type=int, default=10,
                        help='滑动平均窗口大小 (默认: 10)')
    parser.add_argument('--compare', nargs='+', type=str, default=None,
                        help='多实验对比模式：提供多个 training_history.json 路径')
    parser.add_argument('--no-show', action='store_true',
                        help='不自动打开图片（默认不打开）')
    
    args = parser.parse_args()
    
    # 配置全局样式
    setup_plot_style()
    
    if args.compare:
        # 多实验对比模式
        save_dir = Path(args.output)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"多实验对比模式")
        print(f"  实验数量: {len(args.compare)}")
        for p in args.compare:
            print(f"    - {p}")
        print(f"{'='*60}\n")
        
        plot_comparison(args.compare, args.window, save_dir)
        plot_comparison_metrics(args.compare, args.window, save_dir)
        
        print(f"\n对比图表已保存至: {save_dir}\n")
    else:
        # 单实验模式
        if not Path(args.input).exists():
            print(f"错误: 找不到训练历史文件: {args.input}")
            print("请先运行训练: python scripts/train.py")
            print("或指定文件路径: python scripts/plot_results.py --input <path>")
            sys.exit(1)
        
        generate_all_plots(args.input, args.output, args.window)


if __name__ == '__main__':
    main()
