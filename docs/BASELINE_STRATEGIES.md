# 基线算法策略说明

本文档说明 `scripts/compare_system_baselines.py` 中实现的基线算法，用于与当前系统方法 `HAN+MAPPO` 在 `results/full_train_delay_focus` 场景下进行对比。

## 1. 对比对象

当前系统方法为已经训练完成的 `full_train_delay_focus` 模型，默认读取：

- `results/full_train_delay_focus/best_model.pt`
- `results/full_train_delay_focus/training_history.json`

基线方法均不训练神经网络，直接根据当前环境状态生成动作。每个用户在每个时间步输出联合动作：

```text
action_i = (handover_action_i, offload_ratio_i)
```

其中：

- `handover_action = 0` 表示保持当前连接。
- `handover_action = k` 表示切换到当前可见卫星列表中的第 `k` 个候选卫星。
- `offload_ratio in [0, 1]` 表示任务卸载到卫星 MEC 的比例。

评价指标包括：

- `mean_reward`
- `avg_delay`
- `total_energy`
- `handover_success_rate`
- `service_continuity_rate`
- `task_completion_rate`
- `pending_task_rate`
- `avg_load_balance_score`

## 2. 简单启发式基线

简单启发式基线只改变切换策略，卸载比例使用固定值。脚本会默认在 `0.0`、`0.5`、`1.0` 中搜索最优固定卸载比例，因此最终报告的是该启发式在固定卸载网格中的最好结果。

### 2.1 Random

`Random` 是随机策略基线。

切换策略：

- 对每个用户，在当前可用动作集合中随机选择。
- 可选动作包括 `0`，即不切换，以及所有当前可见候选卫星。

卸载策略：

- 使用固定卸载比例。
- 默认在 `0.0`、`0.5`、`1.0` 中搜索最优值。

对照意义：

- 作为性能下界。
- 用于验证系统方法至少应明显优于随机选择。

### 2.2 Stay

`Stay` 是被动保持连接策略。

切换策略：

- 主动决策时始终输出 `handover_action = 0`。
- 如果当前服务卫星不可见，环境内部仍可能触发被动重连或阻塞状态。

卸载策略：

- 使用固定卸载比例。
- 默认在 `0.0`、`0.5`、`1.0` 中搜索最优值。

对照意义：

- 反映完全不做主动切换规划时的性能。
- 用于衡量主动切换机制的收益。

### 2.3 Max-Elev

`Max-Elev` 是最大仰角策略。

切换策略：

- 在当前可见卫星中选择仰角 `elevation_deg` 最大的卫星。
- 如果该卫星已经是当前服务卫星，则保持不切换。

卸载策略：

- 使用固定卸载比例。
- 默认在 `0.0`、`0.5`、`1.0` 中搜索最优值。

对照意义：

- 代表只关注瞬时链路质量的策略。
- 仰角越高，一般传播距离更短，信道质量更好，但它不考虑剩余可见时间和 MEC 队列负载。

### 2.4 Max-RVT

`Max-RVT` 是最大剩余可见时间策略。

切换策略：

- 在当前可见卫星中选择 `rvt_seconds` 最大的卫星。
- 如果该卫星已经是当前服务卫星，则保持不切换。

卸载策略：

- 使用固定卸载比例。
- 默认在 `0.0`、`0.5`、`1.0` 中搜索最优值。

对照意义：

- 代表只关注服务连续性的策略。
- 适合 LEO 场景，因为卫星高速移动导致链路可用时间有限。
- 该策略通常比只看距离或仰角更稳定，但不直接考虑任务时延、能耗和 MEC 队列。

### 2.5 Min-Distance

`Min-Distance` 是最小距离策略。

切换策略：

- 在当前可见卫星中选择 `distance_km` 最小的卫星。
- 如果该卫星已经是当前服务卫星，则保持不切换。

卸载策略：

- 使用固定卸载比例。
- 默认在 `0.0`、`0.5`、`1.0` 中搜索最优值。

对照意义：

- 代表只关注传播距离的策略。
- 距离越小，传播时延和链路损耗通常越低。
- 但该策略不考虑目标卫星未来是否很快不可见，也不考虑 MEC 负载。

## 3. Threshold-RVT Adaptive

`Threshold-RVT Adaptive` 是带阈值的主动切换策略，同时使用任务信息自适应选择卸载比例。

### 3.1 切换策略

该策略优先保持当前连接，只有在当前连接质量不足时才主动切换。

保持当前连接的条件：

```text
current_rvt >= rvt_threshold
and current_queue_ratio <= queue_threshold
```

默认含义：

- 当前服务卫星剩余可见时间足够长。
- 当前服务卫星 MEC 队列没有明显拥塞。

如果不满足保持条件，则在可见卫星中计算综合分数：

```text
score =
    0.45 * RVT_score
  + 0.20 * elevation_score
  + 0.20 * SNR_score
  + 0.15 * queue_headroom
```

其中：

- `RVT_score` 表示剩余可见时间得分。
- `elevation_score` 表示仰角得分。
- `SNR_score` 表示信噪比得分。
- `queue_headroom` 表示目标 MEC 队列剩余空间。

选择综合分数最高的卫星作为切换目标。

### 3.2 卸载策略

如果当前用户没有任务，则卸载比例为 `0`。

如果当前用户有任务，则调用环境中的解析卸载估计器，在当前链路与目标卫星 MEC 能力下搜索卸载比例：

```text
offload_ratio = argmin objective(task, link, MEC)
```

然后根据目标 MEC 队列剩余空间做保守缩放：

```text
offload_ratio = offload_ratio * max(queue_headroom, 0.25)
```

这样可以避免在目标队列拥塞时继续大量卸载。

### 3.3 对照意义

该策略比简单启发式更强，因为它同时考虑：

- 链路连续性
- 当前链路质量
- MEC 队列拥塞
- 当前任务的计算量、数据量和 deadline

但它仍然是一步规则策略，不学习长期回报。

## 4. Joint Greedy

`Joint Greedy` 是最强的规则型基线。它同时枚举切换目标和卸载比例，并根据当前状态估计每个候选联合动作的价值。

### 4.1 候选动作集合

对每个用户，候选切换动作包括：

```text
handover_candidates = {0} union {1, 2, ..., num_visible_sats}
```

候选卸载比例默认包括：

```text
offload_candidates = {0.0, 0.25, 0.5, 0.75, 1.0}
```

因此每个用户会枚举：

```text
(handover_action, offload_ratio)
```

然后选择估计分数最高的联合动作。

### 4.2 用户决策顺序

同一时间步内，用户不是完全独立决策，而是按任务紧迫程度排序：

```text
deadline 较小的任务优先决策
```

每个用户确定动作后，策略会虚拟更新目标卫星的队列增量，用于后续用户评分。这样可以粗略模拟多用户竞争 MEC 资源的影响。

### 4.3 切换价值估计

如果候选目标卫星与当前服务卫星相同，则切换价值为 `0`。

如果需要切换，则估计切换成功概率：

```text
success_prob = f(elevation, RVT, SNR, utilization, queue_ratio, migration_load)
```

该函数与环境中的切换成功概率模型保持一致。

切换价值由期望收益组成：

```text
handover_value =
    success_prob * success_value
  + (1 - success_prob) * failure_value
```

其中：

- `success_value` 包含目标卫星质量收益、RVT 收益、切换成本、任务迁移成本和负载均衡收益。
- `failure_value` 是切换失败惩罚。

### 4.4 任务处理价值估计

对每个候选卸载比例，策略估计：

- 本地计算时延
- 本地计算能耗
- 上传时延
- 下载时延
- 卫星计算时延
- 当前 MEC 队列等待时间
- 同一步内前序用户虚拟加入队列后的额外等待

如果 `offload_ratio = 0`，则任务完全本地执行：

```text
total_delay = local_delay
total_energy = local_energy
```

如果 `offload_ratio > 0`，则任务被拆分为本地部分和卸载部分：

```text
offload_delay = upload_delay + queue_wait + satellite_compute_delay + download_delay
total_delay = max(local_delay, offload_delay)
total_energy = local_energy + upload_energy
```

如果目标 MEC 队列已满，则按环境逻辑退化为本地执行，并加入队列满惩罚。

### 4.5 最终评分

在多目标 `full_train_delay_focus` 场景下，任务评分复用环境中的 reward 逻辑：

```text
task_score = reward_delay + reward_energy + reward_qos + deadline_penalty
```

最终联合动作分数为：

```text
joint_score = handover_value + task_score + enqueue_bonus
```

策略选择分数最高的 `(handover_action, offload_ratio)`。

### 4.6 对照意义

`Joint Greedy` 是一个强规则基线，能够说明：

- 如果只做当前时刻最优的一步贪心，系统能达到什么水平。
- `HAN+MAPPO` 是否能通过长期学习超过手工规则。
- 图表示和多智能体强化学习是否带来了超越局部规则的收益。

它的局限是：

- 只做一步前瞻，不优化长期累计回报。
- 不学习复杂的未来切换时机。
- 对未来任务到达、未来队列变化和未来卫星运动只做近似估计。

## 5. 推荐使用方式

默认运行：

```powershell
python scripts\compare_system_baselines.py
```

快速测试：

```powershell
python scripts\compare_system_baselines.py --episodes 1 --max-steps 50
```

正式对比：

```powershell
python scripts\compare_system_baselines.py --episodes 5
```

输出目录默认为：

```text
results/baseline_compare/<timestamp>/
```

其中包括：

- `comparison_summary.json`
- `comparison_summary.csv`
- `episode_metrics.csv`
- `method_comparison.png`
- `reward_episode_comparison.png`
- `delay_episode_comparison.png`
- `energy_episode_comparison.png`
- `additional_metrics_episode_comparison.png`
- `reward_curve_vs_baselines.png`
