# LEO HAN+MAPPO 训练使用方法

> **文档版本**: v4.0
> **更新日期**: 2026-04-15
> **适用范围**: 环境安装、训练运行、参数调优、结果可视化、常见问题
> **当前默认实验**: `results/full_train_delay_focus`

---

## 一、环境准备

### 1.1 硬件要求

| 级别 | GPU | CPU | 内存 | 预计速度 |
|------|-----|-----|------|----------|
| **推荐** | RTX 3090 (24GB) | 8核+ | 32GB | ~400 步/秒 |
| 可用 | RTX 2080Ti / A100 | 4核+ | 16GB | 200-600 步/秒 |
| 最低 | 无 GPU (CPU only) | 4核 | 8GB | ~40 步/秒 |

> **注意**: 模型仅 822K 参数，GPU 显存几乎不是瓶颈。训练速度主要受 **CPU 端环境仿真**限制。

### 1.2 软件安装

```bash
# 1. 创建 conda 环境
conda create -n satellite python=3.10 -y
conda activate satellite

# 2. 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其他依赖
pip install numpy scipy matplotlib gymnasium pyyaml pytest

# 4. 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "No GPU"
```

### 1.3 验证项目完整性

```bash
cd LEO_switch

# 运行单元测试
python -m pytest tests/test_graph.py -v
# 预期: 12 passed, 4 skipped

# 快速端到端检查
python -c "
from scripts.train import TrainConfig, HANMAPPOTrainer
c = TrainConfig()
c.total_timesteps = 100
c.n_steps = 100
c.device = 'cuda'
c.num_users = 3
c.save_path = '/tmp/test_leo'
t = HANMAPPOTrainer(c)
t.train()
print('端到端验证通过!')
"
```

---

## 二、训练运行

### 2.1 方式一：使用服务器训练脚本（推荐）

`scripts/run_server_training.py` 预设了 4 种训练方案：

| 方案 | 命令 | 用户数 | 总步数 | Early Stop | 预计耗时 |
|------|------|--------|--------|------------|----------|
| `quick` | `--plan quick` | 5 | 10 万 | 15 轮 | 15-30 分钟 |
| `standard` | `--plan standard` | 10 | 100 万 | 30 轮 | 3-5 小时 |
| `large` | `--plan large` | 20 | 200 万 | 50 轮 | 6-10 小时 |
| `multi_seed` | `--plan multi_seed` | 10 | 5×100 万 | 30 轮 | 15-25 小时 |

#### 推荐操作流程

```bash
# 第一步：快速验证（确保环境无误）
python scripts/run_server_training.py --plan quick

# 第二步：正式训练（后台运行）
nohup python scripts/run_server_training.py --plan standard > train.log 2>&1 &

# 监控进度
tail -f train.log

# 查看 GPU 使用情况
watch -n 2 nvidia-smi
```

#### 自定义参数

```bash
# 修改用户数和总步数
python scripts/run_server_training.py --plan standard --users 8 --steps 2000000

# 使用 CPU
python scripts/run_server_training.py --plan quick --device cpu

# 指定随机种子
python scripts/run_server_training.py --plan standard --seed 123
```

#### 多种子对比实验（用于论文）

```bash
# 自动用 5 个种子 [42, 123, 456, 789, 2024] 训练
python scripts/run_server_training.py --plan multi_seed

# 输出:
#   results/multi_seed/seed_42/training_history.json
#   results/multi_seed/seed_123/training_history.json
#   ...
#   results/multi_seed/figures/comparison.pdf      ← 对比图
```

### 2.2 方式二：使用 train.py 直接训练

`train.py` 支持通过命令行参数直接控制所有训练配置，语法为 `python scripts/train.py --参数名 值`。

#### 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--exp_name` | str | `han_mappo_delay_focus_fast` | 实验名称 |
| `--seed` | int | `42` | 随机种子 |
| `--device` | str | `auto` | 设备（`cuda`/`cpu`/`auto`） |
| `--num_users` | int | `10` | 用户数量（智能体数） |
| `--max_steps` | int | `2000` | 每 episode 最大步数 |
| `--total_timesteps` | int | `1000000` | 总训练步数 |
| `--n_steps` | int | `2048` | 每次更新收集的步数 |
| `--learning_rate` | float | `3e-4` | 学习率 |
| `--batch_size` | int | `256` | 小批量大小 |
| `--n_epochs` | int | `4` | 每次 PPO 更新 epoch 数 |
| `--han_hidden_dim` | int | `64` | HAN 隐藏维度 |
| `--han_num_heads` | int | `4` | 注意力头数 |
| `--han_num_layers` | int | `2` | HAN 层数 |
| `--eval_interval` | int | `100000` | 评估间隔（步数） |
| `--eval_episodes` | int | `3` | 每次评估 episode 数 |
| `--graph_update_interval` | int | `100` | 图重建/重编码间隔 |
| `--save_interval` | int | `200000` | 检查点保存间隔 |
| `--save_path` | str | `results/full_train_delay_focus` | 模型保存路径 |
| `--load_path` | str | `None` | 加载检查点路径 |
| `--eval_only` | flag | - | 仅评估不训练 |

#### 常用命令示例

```bash
# 基本训练（使用默认参数）
python scripts/train.py

# 显式指定当前默认实验参数
python scripts/train.py --num_users 10 --max_steps 2000 --total_timesteps 1000000

# 完整自定义训练
python scripts/train.py \
  --num_users 10 \
  --max_steps 2000 \
  --total_timesteps 1000000 \
  --n_steps 2048 \
  --learning_rate 3e-4 \
  --batch_size 256 \
  --n_epochs 4 \
  --eval_interval 100000 \
  --eval_episodes 3 \
  --graph_update_interval 100 \
  --save_path results/my_experiment

# 调整 HAN 网络结构
python scripts/train.py --han_hidden_dim 128 --han_num_heads 8 --han_num_layers 3

# 从检查点恢复训练
python scripts/train.py --load_path results/full_train_delay_focus/checkpoint_200704.pt

# 仅评估已训练模型
python scripts/train.py --load_path results/full_train_delay_focus/best_model.pt --eval_only

# 使用 CPU 训练
python scripts/train.py --device cpu --num_users 3 --total_timesteps 100000
```

> **提示**: 参数可以任意组合。未指定的参数使用 `TrainConfig` 中的默认值。

### 2.3 使用 tmux/screen 保持会话

```bash
# 创建 tmux 会话
tmux new -s leo_train

# 在会话中运行训练
python scripts/run_server_training.py --plan standard

# 分离会话: Ctrl+B, 然后按 D
# 重新连接: tmux attach -t leo_train
```

---

## 三、训练输出说明

### 3.1 输出文件结构

训练完成后 `results/` 目录结构如下：

```
results/
├── full_train_delay_focus/         # --plan standard 的输出
│   ├── best_model.pt               # 评估奖励最高的模型
│   ├── final_model.pt              # 训练结束时的模型
│   ├── checkpoint_200704.pt        # 中间检查点
│   ├── checkpoint_401408.pt
│   ├── training_history.json       # 完整训练历史（供可视化）
│   └── figures/                    # 自动生成的图表
│       ├── reward_curve.png/pdf
│       ├── loss_curves.png/pdf
│       ├── entropy_kl.png/pdf
│       ├── handover_task_rate.png/pdf
│       ├── delay_energy.png/pdf
│       ├── eval_curve.png/pdf
│       ├── clip_fraction.png/pdf
│       ├── dashboard.png/pdf       # 综合仪表盘
│       └── plot_info.json
└── logs/                           # 文本日志
```

### 3.2 training_history.json 格式

```json
{
  "config": { /* 完整训练配置 */ },
  "training": [
    {
      "update": 1,
      "total_steps": 2048,
      "episodes": 2,
      "mean_reward": 12.5,
      "actor_loss": 0.31,
      "critic_loss": 0.05,
      "entropy": 1.82,
      "kl_divergence": 0.003,
      "clip_fraction": 0.08,
      "handover_success_rate": 0.75,
      "task_completion_rate": 0.60,
      "avg_delay": 0.035,
      "total_energy": 4.2,
      "deadline_violations": 3,
      "recent_mean_reward": 12.5
    },
    ...
  ],
  "evaluation": [
    {
      "total_steps": 20000,
      "eval_mean_reward": 85.2,
      "eval_std_reward": 4.3,
      ...
    },
    ...
  ],
  "summary": {
    "total_steps": 1000000,
    "total_episodes": 976,
    "best_reward": 92.5,
    "training_time_sec": 7200
  }
}
```

### 3.3 检查点文件内容 (.pt)

```python
checkpoint = {
    'total_steps': int,
    'episodes': int,
    'best_reward': float,
    'config': dict,
    'actor_state_dict': OrderedDict,
    'critic_state_dict': OrderedDict,
    'actor_optimizer_state_dict': OrderedDict,
    'critic_optimizer_state_dict': OrderedDict,
    'han_state_dict': OrderedDict,
}
```

---

## 四、结果可视化

### 4.1 自动生成（训练结束自动执行）

使用 `run_server_training.py` 时，训练结束后图表自动生成在 `figures/` 子目录下。

### 4.2 手动生成图表

```bash
# 基本使用
python scripts/plot_results.py --input results/full_train_delay_focus/training_history.json

# 自定义输出目录和滑动窗口
python scripts/plot_results.py \
  --input results/full_train_delay_focus/training_history.json \
  --output results/my_figures \
  --window 20

# 仅重新绘图（不重新训练）
python scripts/run_server_training.py --plot_only results/full_train_delay_focus --window 15
```

### 4.3 多实验对比

```bash
python scripts/plot_results.py --compare \
  results/multi_seed/seed_42/training_history.json \
  results/multi_seed/seed_123/training_history.json \
  results/multi_seed/seed_456/training_history.json
```

### 4.4 图表说明

| # | 文件名 | 内容 | 论文用途 |
|---|--------|------|----------|
| 1 | `reward_curve` | 奖励收敛曲线（原始 + 滑动平均 + 置信区间） | 主收敛图 |
| 2 | `loss_curves` | Actor Loss & Critic Loss 双轴曲线 | 附录/训练稳定性分析 |
| 3 | `entropy_kl` | 策略熵变化 & 近似 KL 散度 | 探索-利用分析 |
| 4 | `handover_task_rate` | 切换成功率 & 任务完成率 & Deadline 违反数 | **核心性能指标** |
| 5 | `delay_energy` | 平均时延 (ms) & 每轮能耗 (J) | **核心性能指标** |
| 6 | `eval_curve` | 评估奖励（带误差棒） | 泛化性能 |
| 7 | `clip_fraction` | PPO Clip Fraction | 超参数分析 |
| 8 | `dashboard` | **6 合 1 综合仪表盘** | 一图总览 |
| 9 | `comparison` | 多实验奖励对比（多种子模式） | 稳定性验证 |
| 10 | `comparison_metrics` | 多实验指标对比（4 指标） | 消融实验 |

---

## 五、参数调优指南

### 5.1 关键超参数

| 参数 | 默认值 | 调优建议 |
|------|--------|----------|
| `learning_rate` | 3e-4 | 奖励震荡大 → 减小至 1e-4；收敛太慢可小幅增大 |
| `n_steps` | 2048 | 增大可提高梯度估计质量，但增大内存和每轮时间 |
| `batch_size` | 256 | 3090 当前默认；显存不足或更新过慢 → 减至 128/64 |
| `entropy_coef` | 0.01 | 探索不足可增大；策略太随机则继续保持低值 |
| `clip_range` | 0.2 | 一般不需要调；更新过激进 → 减小至 0.1 |
| `n_epochs` | 4 | KL 散度过大 → 减少；学习不足 → 小幅增大 |
| `gamma` | 0.99 | 任务时间尺度短 → 减小至 0.95 |
| `num_users` | 10 | 简单场景 3-5，复杂场景 10-20 |
| `total_timesteps` | 1M | 配合 early stopping 自动判断收敛 |
| `early_stop_patience` | 30 | 连续 N 次更新无改善则停止；0=禁用 |

### 5.2 调优流程

```
1. quick 方案跑通 (10万步)
     ↓
2. 观察 reward_curve 趋势
   - 上升 → 继续 standard 方案
   - 平坦/震荡 → 调整 lr, entropy_coef
     ↓
3. standard 方案跑完 (100万步)
     ↓
4. 观察关键指标:
   - 切换成功率 < 70% → 增大 handover 奖励权重
   - 任务完成率 < 50% → 检查 MEC 队列是否拥堵
   - 时延过高 → 调整 reward_delay_weight
     ↓
5. 调参后重跑，用 multi_seed 验证稳定性
     ↓
6. 与基线对比 (baseline_eval.py)
```

### 5.3 训练日志解读

训练过程中日志输出格式：

```
Update    1 | Steps:   2,048 | Episodes:     2 | EpReward:    12.50 | FPS:   400 | Actor Loss: 0.3148 | Critic Loss: 0.0521
```

| 字段 | 含义 | 理想趋势 |
|------|------|----------|
| `EpReward` | Episode 平均奖励 | 持续上升 → 稳定 |
| `FPS` | 每秒仿真步数 | 稳定在 200-500 |
| `Actor Loss` | 策略损失 | 先升后降或波动 |
| `Critic Loss` | 值函数损失 | 持续下降 |

---

## 六、进阶用法

### 6.1 加载已训练模型进行评估

```python
from scripts.train import TrainConfig, HANMAPPOTrainer

config = TrainConfig()
config.device = 'cuda'
config.num_users = 10
trainer = HANMAPPOTrainer(config)
trainer.load_checkpoint('results/full_train_delay_focus/best_model.pt')
trainer._evaluate()  # 运行评估
```

### 6.2 提取训练数据做自定义分析

```python
import json
import numpy as np

with open('results/full_train_delay_focus/training_history.json') as f:
    data = json.load(f)

# 提取奖励曲线
rewards = [r['recent_mean_reward'] for r in data['training']]
steps = [r['total_steps'] for r in data['training']]

# 提取切换成功率
ho_rate = [r['handover_success_rate'] for r in data['training']]

# 提取评估结果
eval_rewards = [r['eval_mean_reward'] for r in data['evaluation']]

print(f"最终奖励: {rewards[-1]:.2f}")
print(f"最终切换成功率: {ho_rate[-1]*100:.1f}%")
print(f"最佳评估奖励: {max(eval_rewards):.2f}")
```

### 6.3 下载服务器结果到本地

```bash
# 下载所有结果
scp -r user@server:~/LEO_switch/results/ ./results_from_server/

# 仅下载图表和历史
scp -r user@server:~/LEO_switch/results/full_train_delay_focus/figures/ ./
scp user@server:~/LEO_switch/results/full_train_delay_focus/training_history.json ./

# 在本地重新生成图表（自定义样式）
python scripts/plot_results.py --input training_history.json --output ./my_figures --window 30
```

---

## 七、常见问题

### Q1: CUDA 不兼容 / "no kernel image is available"

**原因**: PyTorch 版本与 GPU 计算能力不匹配  
**解决**: 安装匹配 CUDA 版本的 PyTorch

```bash
# RTX 3090 (sm_86) 需要 CUDA 11.1+
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Q2: 训练开始后 EpReward 一直为 0 或 "RollReward"

**原因**: `n_steps` < `max_steps` 时，一个 rollout 中没有完成任何 episode  
**不影响训练**: rollout reward 仍在累积，当 episode 完成后会显示 EpReward  
**解决**: 可减小 `max_steps`（如 500）使 episode 更短

### Q3: 内存不足 (OOM)

**解决方案**:
- 减小 `n_steps`（2048 → 1024）
- 减小 `batch_size`（256 → 128 或 64）
- 减少 `num_users`

### Q4: 训练极慢

**原因**: 环境仿真在 CPU 端，是主要瓶颈  
**解决**:
- 确保使用多核 CPU
- 减少 `num_users`
- 减少 `max_steps`

### Q5: 奖励不收敛 / 持续震荡

**可能原因与对策**:
- 学习率太大 → 减至 `5e-5`
- 探索不足 → 增大 `entropy_coef` 至 `0.05`
- 环境太难 → 先用 3 用户、500 步 episode 验证
- 训练步数不够 → 至少 50 万步

### Q7: 训练过早收敛 / reward 很快不再上升

**可能原因与对策**:
- 环境太简单 → 增加 `num_users`（10-20），提高 `task_arrival_prob`（0.6+）
- MEC 资源过剩 → 降低 `max_queue_size`、`satellite_cpu_freq_ghz`
- 探索不足 → 增大 `entropy_coef` 至 0.03-0.05
- Early stopping 会自动检测并停止浪费算力

### Q6: 如何对比不同算法?

```bash
# 运行基线评估
python scripts/baseline_eval.py

# 生成基线报告
python scripts/baseline_report.py
```

---

## 八、完整命令速查

```bash
# ==================== 安装 ====================
conda create -n satellite python=3.10 -y
conda activate satellite
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy matplotlib gymnasium pyyaml pytest

# ==================== 验证 ====================
python -m pytest tests/test_graph.py -v

# ==================== 训练（方式一：run_server_training.py） ====================
# 快速验证 (~15分钟)
python scripts/run_server_training.py --plan quick

# 标准训练 (~3-5小时，含 early stopping)
nohup python scripts/run_server_training.py --plan standard > train.log 2>&1 &

# 大规模训练 (~6-10小时)
nohup python scripts/run_server_training.py --plan large > train_large.log 2>&1 &

# 多种子对比 (~15-25小时)
nohup python scripts/run_server_training.py --plan multi_seed > train_multi.log 2>&1 &

# 自定义参数
python scripts/run_server_training.py --plan standard --users 15 --steps 2000000

# ==================== 训练（方式二：train.py 直接指定参数） ====================
# 10用户 + 100万步（等价于当前默认配置）
python scripts/train.py --num_users 10 --max_steps 2000 --total_timesteps 1000000

# 完整自定义
python scripts/train.py --num_users 10 --max_steps 2000 --total_timesteps 1000000 \
  --learning_rate 3e-4 --batch_size 256 --n_epochs 4 --save_path results/custom

# ==================== 可视化 ====================
# 自动（训练结束后已生成）
ls results/full_train_delay_focus/figures/

# 手动重新生成
python scripts/plot_results.py -i results/full_train_delay_focus/training_history.json -w 20

# 多实验对比
python scripts/plot_results.py --compare results/multi_seed/seed_*/training_history.json

# ==================== 评估 ====================
python scripts/train.py --load_path results/full_train_delay_focus/best_model.pt --eval_only

# ==================== 下载结果 ====================
scp -r user@server:~/LEO_switch/results/ ./
```
