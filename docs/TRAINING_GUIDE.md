# LEO HAN+MAPPO 训练指南

本文档说明当前仓库的训练入口、常用参数、输出结构、绘图方式和基线对比流程。

默认建议使用项目 Conda 环境中的 Python：

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe
```

如无特殊说明，下面命令中的 `python` 可以替换为上述完整解释器路径。

## 1. 系统目标

本项目研究 LEO 卫星网络中的联合优化问题：

- 用户在可见卫星之间进行切换。
- 用户任务可以在本地计算，也可以卸载到卫星 MEC。
- 策略需要同时考虑时延、能耗、服务连续性、任务完成率、队列压力和负载均衡。

主方法为 `HAN+MAPPO`：

- HAN 编码用户、卫星、MEC 之间的异构图关系。
- MAPPO 训练多用户联合策略。
- 每个用户输出混合动作：

```text
action_i = (handover_action_i, offload_ratio_i)
```

## 2. 环境与安装

推荐环境：

| 项目 | 建议 |
| --- | --- |
| Python | 3.10 |
| GPU | CUDA 可用 GPU，CPU 也可用于小规模 smoke |
| 内存 | 16GB 以上 |
| 主要依赖 | `torch`、`numpy`、`scipy`、`matplotlib`、`gymnasium`、`pyyaml`、`pytest` |

示例安装：

```bash
conda create -n satellite python=3.10 -y
conda activate satellite
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy matplotlib gymnasium pyyaml pytest
```

验证 PyTorch：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 3. 训练入口

主要训练脚本：

```text
scripts/train.py
```

默认训练：

```powershell
python scripts\train.py
```

常用训练命令：

```powershell
python scripts\train.py `
  --num_users 10 `
  --max_steps 2000 `
  --total_timesteps 1000000 `
  --learning_rate 3e-4 `
  --batch_size 256 `
  --n_epochs 4 `
  --save_path results\my_experiment
```

仅评估已有模型：

```powershell
python scripts\train.py `
  --load_path results\full_train_delay_focus\best_model.pt `
  --eval_only
```

CPU smoke：

```powershell
python scripts\train.py `
  --device cpu `
  --num_users 3 `
  --total_timesteps 100000
```

## 4. 关键训练参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--num_users` | `10` | 用户数量 |
| `--max_steps` | `2000` | 每个 episode 最大步数 |
| `--total_timesteps` | `1000000` | 总训练步数 |
| `--n_steps` | `2048` | 每次 MAPPO rollout 步数 |
| `--learning_rate` | `3e-4` | 学习率 |
| `--batch_size` | `256` | PPO mini-batch 大小 |
| `--n_epochs` | `4` | 每轮 PPO 更新 epoch 数 |
| `--han_hidden_dim` | `64` | HAN 隐藏维度 |
| `--han_num_heads` | `4` | HAN 注意力头数 |
| `--han_num_layers` | `2` | HAN 层数 |
| `--eval_interval` | `100000` | 评估间隔 |
| `--eval_episodes` | `3` | 每次评估 episode 数 |
| `--save_interval` | `200000` | checkpoint 保存间隔 |
| `--save_path` | `results/full_train_delay_focus` | 模型与日志保存目录 |
| `--best-model-metric` | `reward` | `best_model.pt` 的选择指标 |
| `--early_stop_patience` | `30` | 连续多少次无提升后 early stop |

可用的 `--best-model-metric` 包括：

- `reward`
- `avg_delay`
- `total_energy`
- `service_continuity_rate`
- `avg_load_balance_score`
- `task_completion_rate`
- `latency_priority_score`

## 5. 默认环境参数

| 参数 | 当前默认值 |
| --- | --- |
| `num_planes` | `6` |
| `sats_per_plane` | `11` |
| `altitude_km` | `550.0` |
| `inclination_deg` | `53.0` |
| `time_step_sec` | `1.0` |
| `max_visible_sats` | `10` |
| `task_arrival_prob` | `0.45` |

Reward 权重：

| 参数 | 默认值 |
| --- | --- |
| `reward_delay_weight` | `1.4` |
| `reward_energy_weight` | `0.4` |
| `reward_handover_weight` | `0.3` |
| `reward_load_balance_weight` | `0.1` |
| `reward_qos_weight` | `0.4` |
| `reward_enqueue_bonus` | `0.02` |
| `reward_queue_full_penalty` | `0.3` |

## 6. 输出结构

典型训练目录：

```text
results/full_train_delay_focus/
  best_model.pt
  final_model.pt
  checkpoint_*.pt
  training_history.json
  figures/
    reward_curve.png
    reward_curve.pdf
    loss_curves.png
    entropy_kl.png
    handover_task_rate.png
    delay_energy.png
    eval_curve.png
    dashboard.png
```

`training_history.json` 主要包含：

- `config`：训练配置。
- `training`：每次训练更新的指标记录。
- `evaluation`：评估记录。
- `summary`：总步数、总 episode、最佳分数、训练时间等摘要。

常见训练指标：

- `mean_reward`
- `recent_mean_reward`
- `actor_loss`
- `critic_loss`
- `entropy`
- `kl_divergence`
- `handover_success_rate`
- `task_completion_rate`
- `avg_delay`
- `total_energy`
- `service_continuity_rate`

## 7. 绘图

单次训练绘图：

```powershell
python scripts\plot_results.py `
  --input results\full_train_delay_focus\training_history.json
```

指定输出目录与平滑窗口：

```powershell
python scripts\plot_results.py `
  --input results\full_train_delay_focus\training_history.json `
  --output results\my_figures `
  --window 20
```

多实验对比：

```powershell
python scripts\plot_results.py --compare `
  results\multi_seed\seed_42\training_history.json `
  results\multi_seed\seed_123\training_history.json `
  results\multi_seed\seed_456\training_history.json
```

当前 reward 图样式：

- 原始 `mean_reward` 画为半透明阴影或波动背景。
- 轻度平滑后的 reward 画为实线。
- 当旧历史缺少 `mean_reward` 时，回退使用 `recent_mean_reward`。
- 平滑窗口不会过大，曲线保留适当震荡。
- 横纵轴沿用项目原有格式。

## 8. 基线对比

统一对比脚本：

```text
scripts/compare_system_baselines.py
```

当前默认基线集合：

- `random`
- `min_distance`
- `full_local`
- `joint_greedy`
- `dqn`
- `mappo_no_han`

旧基线 `stay`、`max_elev`、`max_rvt`、`threshold_rvt` 已移除。

对比已有系统训练结果：

```powershell
python scripts\compare_system_baselines.py `
  --run-mode compare_only `
  --system-run-dir results\full_train_latency_priority
```

快速 smoke：

```powershell
python scripts\compare_system_baselines.py `
  --run-mode compare_only `
  --system-run-dir results\full_train_latency_priority `
  --device cpu `
  --episodes 1 `
  --max-steps 50 `
  --dqn-timesteps 500 `
  --no-han-total-timesteps 2048
```

正式对比建议使用：

```powershell
python scripts\compare_system_baselines.py `
  --run-mode compare_only `
  --system-run-dir results\full_train_latency_priority `
  --episodes 5 `
  --best-model-metric latency_priority_score `
  --compare-ranking-metric latency_priority_score
```

## 9. 调参建议

Reward 大幅震荡或不收敛时：

- 降低 `learning_rate`，例如尝试 `1e-4` 或 `5e-5`。
- 适当提高 `entropy_coef`，例如 `0.03` 到 `0.05`。
- 先用 `num_users=3` 或 `num_users=5` 做小规模验证。

任务完成率偏低时：

- 检查 MEC 队列是否长期满载。
- 适当降低 `task_arrival_prob` 或增大训练步数。
- 观察 `pending_task_rate`、`task_resolution_rate` 和 `service_availability_rate`。

切换成功率偏低时：

- 检查候选卫星可见性、RVT 和距离分布。
- 对比 `min_distance` 与 `joint_greedy`，确认规则基线是否也存在类似问题。

## 10. 常见问题

### Q1: CUDA 报 `no kernel image is available`

通常是当前 PyTorch CUDA wheel 不支持显卡架构。可以改用 CPU 跑 smoke，或安装支持当前 GPU 的 PyTorch 版本。

```powershell
python scripts\compare_system_baselines.py --device cpu --episodes 1 --max-steps 50
```

### Q2: Reward 曲线过于平滑

当前代码已经改为“原始 reward 阴影 + 轻度平滑实线”。如果仍然过平滑，优先调小绘图窗口 `--window`。

### Q3: 想只跑部分基线

使用 `--baselines`：

```powershell
python scripts\compare_system_baselines.py `
  --run-mode compare_only `
  --system-run-dir results\full_train_latency_priority `
  --baselines full-local joint-greedy dqn mappo-no-han
```
