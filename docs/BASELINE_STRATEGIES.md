# 基线策略说明

本文档说明 `scripts/compare_system_baselines.py` 当前使用的系统方法与基线算法。

统一对比入口会将训练好的系统方法 `HAN+MAPPO` 与启发式策略、值函数强化学习基线、结构消融实验放在同一套评估流程中比较。当前 `--baselines all` 默认展开为：

- `random`
- `min_distance`
- `full_local`
- `joint_greedy`
- `dqn`
- `mappo_no_han`

旧的单因素切换规则 `stay`、`max_elev`、`max_rvt`、`threshold_rvt` 已从当前主动对比集合中移除。它们适合早期 sanity check，但彼此重叠较多，也会让论文图显得拥挤。当前集合保留了随机下界、几何规则、本地计算下界、强贪心规则、经典值函数 RL 基线，以及去掉 HAN 的结构消融。

## 1. 系统方法

`HAN+MAPPO` 是本文的主方法：

- HAN 负责编码卫星、用户、MEC 构成的异构图。
- MAPPO 负责学习多用户联合切换与卸载策略。
- 每个用户的动作是：

```text
action_i = (handover_action_i, offload_ratio_i)
```

其中 `handover_action = 0` 表示保持当前服务卫星不主动切换，`offload_ratio in [0, 1]` 表示卸载到卫星 MEC 的计算比例。

## 2. 当前基线

### 2.1 Random

`random` 从当前可用动作集合中均匀随机选择合法切换动作。卸载比例通过 `--fixed-offload-grid` 做小网格搜索，默认网格为：

```text
0.0 0.5 1.0
```

该基线主要作为策略质量的宽松下界。

### 2.2 Min-Distance

`min_distance` 选择传播距离最短的可见卫星。卸载比例同样使用固定网格搜索。

保留这个基线是为了保留一个简单、直观的几何驱动规则，同时避免同时放入多个高度相似的单因素卫星选择规则。

### 2.3 Full-Local

`full_local` 始终输出：

```text
handover_action = 0
offload_ratio = 0.0
```

它表示所有计算都在本地完成，并且不主动触发切换。这个基线用于说明卫星 MEC 卸载是否真的改善了时延、任务完成率和能耗权衡。

### 2.4 Joint Greedy

`joint_greedy` 会对所有候选组合打分：

```text
(handover_action, offload_ratio)
```

评分依据包括当前环境状态、队列压力、切换收益、任务时延、任务能耗、QoS 奖励和入队奖励。卸载比例网格由 `--joint-offload-grid` 控制，默认值为：

```text
0.0 0.25 0.5 0.75 1.0
```

这是当前最强的手写规则基线，用来回答学习式长时域协同策略是否能超过一步贪心优化。

### 2.5 DQN

`dqn` 是值函数强化学习基线。由于环境动作是混合动作，也就是“离散切换 + 连续卸载比例”，DQN 基线会先使用 `--dqn-offload-grid` 将卸载比例离散化，再把每个组合视为一个离散 Q 动作：

```text
q_action = (handover_action, discrete_offload_ratio)
```

默认 DQN 卸载比例网格为：

```text
0.0 0.5 1.0
```

训练步数默认沿用 `--total-timesteps`，也可以单独指定：

```powershell
--dqn-timesteps <steps>
```

训练后的 DQN checkpoint 保存到：

```text
<output_dir>/learned_baselines/dqn/dqn_model.pt
```

### 2.6 MAPPO 无 HAN 消融

`mappo_no_han` 是系统方法的结构消融。它保留 MAPPO 算法，但移除 HAN 图编码器，策略直接使用每个用户的原始环境观测。

训练步数默认沿用 `--total-timesteps`，也可以单独指定：

```powershell
--no-han-total-timesteps <steps>
```

实验产物保存到：

```text
<output_dir>/learned_baselines/mappo_no_han/
```

这个消融用于直接验证异构图表示是否带来了超过 MAPPO 学习器本身的收益。

## 3. 已移除基线

以下基线不再属于当前主动对比集合：

- `stay`
- `max_elev`
- `max_rvt`
- `threshold_rvt`

移除理由：

- `stay` 的作用基本由 `full_local` 覆盖，而 `full_local` 对“完全本地计算下界”的含义更清楚。
- `max_elev`、`max_rvt`、`min_distance` 都是单因素卫星选择规则。保留 `min_distance` 即可代表简单几何规则，减少重复曲线。
- `threshold_rvt` 是手调复合规则，它的角色更适合由 `joint_greedy` 替代，因为后者显式评估切换和卸载的联合决策。

## 4. 推荐命令

默认训练并对比：

```powershell
python scripts\compare_system_baselines.py
```

只对比已有系统训练结果：

```powershell
python scripts\compare_system_baselines.py `
  --run-mode compare_only `
  --system-run-dir results\full_train_latency_priority
```

快速 smoke 对比：

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

正式对比：

```powershell
python scripts\compare_system_baselines.py `
  --run-mode compare_only `
  --system-run-dir results\full_train_latency_priority `
  --episodes 5 `
  --best-model-metric latency_priority_score `
  --compare-ranking-metric latency_priority_score
```

自定义基线子集：

```powershell
python scripts\compare_system_baselines.py `
  --run-mode compare_only `
  --system-run-dir results\full_train_latency_priority `
  --baselines full-local joint-greedy dqn mappo-no-han
```

命令行中可以使用 `full-local`、`mappo-no-han` 这样的连字符写法，脚本内部会规范化为 `full_local`、`mappo_no_han`。

## 5. 输出文件

默认输出目录：

```text
results/baseline_compare/<timestamp>/
```

关键文件包括：

- `comparison_summary.json`
- `comparison_summary.csv`
- `episode_metrics.csv`
- `method_comparison.pdf`
- `reward_curve_vs_baselines.pdf`
- `baseline_reward_episode_comparison.pdf`
- `delay_energy_tradeoff.pdf`
- `reward_distribution.pdf`
- `paper_baseline_dashboard.pdf`

学习式基线的产物保存在：

```text
results/baseline_compare/<timestamp>/learned_baselines/
```

## 6. 对比指标

对比摘要包含：

- `mean_reward`
- `avg_delay`
- `total_energy`
- `handover_success_rate`
- `handover_failure_rate`
- `service_continuity_rate`
- `service_availability_rate`
- `task_completion_rate`
- `task_resolution_rate`
- `pending_task_rate`
- `avg_load_balance_score`
- `selection_metric`
- `selection_score`
- `energy_per_resolved_task`
- `primary_metric_wins`
- `primary_metric_win_count`

默认排序指标为：

```text
latency_priority_score
```

主要对比维度包括平均时延、服务连续性、服务可用性和任务完成率。
