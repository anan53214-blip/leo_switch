# 系统架构说明

本文档概述 LEO_switch 的系统结构、训练流程和关键源码位置。

## 1. 研究问题

系统面向 LEO 卫星网络中的用户服务连续性与任务计算卸载问题。每个地面用户在动态可见卫星集合中选择是否切换服务卫星，同时决定任务在本地计算还是卸载到卫星 MEC。

优化目标包括：

- 降低端到端任务时延。
- 控制本地与卫星侧能耗。
- 提高切换成功率与服务连续性。
- 提高任务完成率与任务解析率。
- 避免 MEC 队列过载并改善负载均衡。

## 2. 总体结构

代码目录分工：

| 目录 | 作用 |
| --- | --- |
| `src/environment/` | LEO 环境、可见性、信道、MEC、任务和用户建模 |
| `src/graph/` | 异构图构建与特征提取 |
| `src/model/` | HAN 编码器、Actor、Critic |
| `src/algorithm/` | MAPPO、rollout buffer、训练逻辑 |
| `scripts/` | 训练、绘图、基线对比和批量运行入口 |
| `docs/` | 系统说明、训练说明、基线说明 |
| `results/` | 模型、训练历史、图表和对比结果 |

## 3. 环境层

核心文件：

- `src/environment/gym_env.py`
- `src/environment/constellation.py`
- `src/environment/visibility.py`
- `src/environment/channel.py`
- `src/environment/mec.py`
- `src/environment/task.py`
- `src/environment/user.py`

环境负责：

- 构建 Walker LEO 星座。
- 计算用户与卫星之间的可见性、距离、仰角、SNR 和剩余可见时间 RVT。
- 维护用户位置、任务到达、任务 deadline 和服务卫星。
- 维护卫星 MEC 队列、CPU 资源、任务入队和处理结果。
- 根据联合动作返回 reward、观测和统计指标。

每个用户的动作格式为：

```text
(handover_action, offload_ratio)
```

其中 `handover_action = 0` 表示不主动切换，其它值对应当前候选可见卫星；`offload_ratio = 0` 表示全本地计算，`1` 表示全部卸载。

## 4. 图建模层

核心文件：

- `src/graph/builder.py`
- `src/graph/features.py`
- `src/model/hetero_gnn.py`

异构图包含用户节点、卫星节点以及与任务/MEC 状态相关的特征。图特征会编码：

- 用户位置与当前服务关系。
- 卫星轨道状态与可见性。
- MEC 队列长度、CPU 利用率、可用频率。
- 用户到候选卫星的距离、仰角、SNR、RVT、负载等。

HAN 编码器将这些异构信息转为用户嵌入和卫星嵌入，供 MAPPO 的 Actor 和 Critic 使用。

## 5. 学习层

核心文件：

- `scripts/train.py`
- `src/algorithm/mappo.py`
- `src/algorithm/buffer.py`
- `src/model/actor.py`
- `src/model/critic.py`

训练采用 CTDE 思路：

- 执行时每个用户根据自身观测和图编码输出动作。
- 训练时 Critic 可以使用更全局的信息估计价值。
- Actor 同时处理离散切换动作与连续卸载比例。

训练循环大致为：

1. 环境 reset。
2. 构建异构图并计算 HAN 编码。
3. Actor 输出每个用户的切换动作和卸载比例。
4. 环境执行联合动作，返回 reward 和指标。
5. Rollout buffer 收集轨迹。
6. MAPPO 使用 GAE 和 PPO clipping 更新 Actor/Critic。
7. 周期性评估并保存 checkpoint。

## 6. Reward 与指标

Reward 主要由以下部分组成：

- 时延奖励或惩罚。
- 能耗奖励或惩罚。
- QoS 奖励。
- 切换收益与切换代价。
- 负载均衡奖励。
- 任务成功入队奖励。
- 队列满、deadline 违约、非法动作等惩罚。

训练与评估会记录：

- `mean_reward`
- `avg_delay`
- `total_energy`
- `handover_success_rate`
- `service_continuity_rate`
- `service_availability_rate`
- `task_completion_rate`
- `task_resolution_rate`
- `pending_task_rate`
- `energy_per_successful_task`
- `mec_load_fairness`

## 7. 运行入口

训练主方法：

```powershell
python scripts\train.py
```

绘制训练图：

```powershell
python scripts\plot_training_artifacts.py `
  --comparison-summary results\baseline_compare\<run_id> `
  --output-dir results\baseline_compare\<run_id>\replot
```

系统与基线统一对比：

```powershell
python scripts\compare_system_baselines.py `
  --run-mode compare_only `
  --system-run-dir results\full_train_latency_priority
```

## 8. 当前对比体系

系统方法：

- `HAN+MAPPO`

当前基线：

- `random`
- `min_distance`
- `full_local`
- `joint_greedy`
- `dqn`
- `mappo_no_han`

其中 `mappo_no_han` 是结构消融，用于验证 HAN 图编码器对最终性能的贡献。
