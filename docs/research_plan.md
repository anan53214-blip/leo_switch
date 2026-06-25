# 研究计划

本文档记录 LEO_switch 当前实验工作的研究目标、技术路线和对比设计。

## 1. 研究目标

面向 LEO 卫星 MEC 网络，研究多用户联合卫星切换与任务卸载策略。目标是在动态可见性、链路变化、MEC 队列竞争和任务 deadline 约束下，提高系统整体服务质量。

核心目标：

- 降低任务平均时延。
- 提高服务连续性与服务可用性。
- 提高任务完成率。
- 控制总能耗。
- 减少无效切换和队列阻塞。

## 2. 方法路线

主方法为 `HAN+MAPPO`：

1. 构建用户、卫星和 MEC 的异构图。
2. 使用 HAN 提取用户侧和卫星侧嵌入。
3. 每个用户作为一个智能体输出切换动作与卸载比例。
4. 使用 MAPPO 进行多智能体协同训练。
5. 用 latency-priority 指标和多项系统指标进行综合比较。

## 3. 关键创新点

### 3.1 异构图状态表达

相比只使用扁平观测，HAN 可以显式建模：

- 用户与可见卫星关系。
- 卫星之间的拓扑与轨道结构。
- MEC 队列与计算资源状态。
- 链路质量、RVT 和负载信息。

### 3.2 联合切换与卸载

系统不是单独优化切换或卸载，而是将二者作为联合动作：

```text
action_i = (handover_action_i, offload_ratio_i)
```

这能同时处理“切到哪里”和“卸载多少”的问题。

### 3.3 多智能体协同

多个用户共享卫星 MEC 队列和无线资源，单用户最优不一定带来系统最优。MAPPO 用集中训练、分散执行的方式处理多用户协同。

## 4. 对比实验设计

当前对比集合：

| 方法 | 作用 |
| --- | --- |
| `HAN+MAPPO` | 系统主方法 |
| `random` | 随机策略下界 |
| `min_distance` | 简单几何规则 |
| `full_local` | 全本地计算下界 |
| `joint_greedy` | 一步联合贪心强规则 |
| `dqn` | 经典值函数 RL 基线 |
| `mappo_no_han` | 去掉 HAN 的结构消融 |

旧基线 `stay`、`max_elev`、`max_rvt`、`threshold_rvt` 已移除，原因是它们与当前基线集合存在明显重叠。

## 5. 评价指标

主要指标按论文常见 KPI 分组报告，不再使用自定义综合排序分：

- 延迟 QoS：`avg_delay`
- 任务可靠性：`task_success_rate`, `deadline_violation_rate`, `task_failure_rate`
- 服务连续性/切换：`service_continuity_rate`,
  `service_availability_rate`, `handover_success_rate`,
  `handover_failure_rate`, `handover_frequency`
- 能耗代价：`energy_per_successful_task`, `total_energy`
- 资源诊断：`mec_load_fairness`

辅助指标：

- `task_completion_rate`
- `task_settlement_rate`
- `task_resolution_rate`
- `pending_task_rate`

## 6. 实验流程

1. 训练主方法 `HAN+MAPPO`。
2. 使用 `best_model_metric` 选择最优 checkpoint。
3. 在统一环境和随机种子下评估主方法。
4. 运行启发式基线、DQN 和 no-HAN 消融。
5. 生成 `comparison_summary.csv/json` 和论文图。
6. 检查 reward 曲线、delay-energy tradeoff、基线排名和统计稳定性。

推荐命令：

```powershell
python scripts\compare_system_baselines.py `
  --run-mode compare_only `
  --system-run-dir results\full_train_latency_priority `
  --episodes 5 `
  --compare-ranking-metric avg_delay
```

## 7. 论文图建议

建议保留以下图：

- 系统方法与基线的综合柱状图。
- reward 收敛曲线。
- delay-energy tradeoff。
- reward 分布图。
- baseline dashboard。

Reward 曲线建议使用当前实现：

- 原始数据作为半透明阴影。
- 轻度平滑后的数据作为实线。
- 保留适当震荡，不做过度平滑。

## 8. 后续工作

可继续扩展：

- 多随机种子统计显著性。
- 不同用户数量下的泛化测试。
- 不同任务到达率下的拥塞敏感性。
- 不同星座规模下的可扩展性。
- 对 DQN 和 no-HAN MAPPO 进行更长训练，保证对比充分。
