# 系统场景与 Reward 细节

本文档说明当前环境建模、任务卸载流程和 reward 组成。源码以 `src/environment/gym_env.py`、`src/environment/mec.py` 和 `scripts/train.py` 为准。

## 1. 系统场景

系统模拟一个 LEO 卫星 MEC 网络，用于研究动态 LEO 拓扑下的多用户
时延敏感任务卸载与服务切换问题。工业场景中，偏远地区、海上、
灾害救援和空天地一体网络常面临地面基础设施不足、终端算力有限、
回传云端时延高的问题；LEO-MEC 能把计算能力推近用户，但卫星高速
移动、可见窗口短、链路质量快变、星上 MEC 资源有限，会使服务切换、
任务卸载和队列拥塞耦合在一起。

因此，本系统的目标不是让更多 MEC 服务器保持活跃，而是在多用户竞争
和频繁切换风险下联合决定服务卫星、切换时机和卸载比例，使任务尽量
按 deadline 成功完成、服务尽量不中断，并控制能耗和 MEC 热点拥塞。

基础星座参数如下：

- 共有 `6` 个轨道平面。
- 每个轨道平面 `11` 颗卫星。
- 总卫星数为 `66`。
- 轨道高度为 `550 km`。
- 轨道倾角为 `53 deg`。
- 默认用户数为 `20`。
- 每步时间为 `1 s`。
- 每个用户最多保留 `10` 个可见卫星候选。

用户在地面区域内移动，随着卫星轨道传播，可见卫星集合会随时间变化。每个用户可能产生计算任务，任务可本地处理，也可按一定比例卸载到当前或新切换的卫星 MEC。

## 2. 决策变量

每个用户在每一步输出一个混合动作：

```text
action_i = (handover_action_i, offload_ratio_i)
```

含义：

- `handover_action_i = 0`：保持当前服务卫星。
- `handover_action_i > 0`：选择候选可见卫星中的一个作为目标服务卫星。
- `offload_ratio_i = 0.0`：全部本地计算。
- `offload_ratio_i = 1.0`：全部卸载到卫星 MEC。
- `0.0 < offload_ratio_i < 1.0`：本地与 MEC 分割计算。

## 3. 可见性与切换

环境会根据卫星位置和用户位置计算候选卫星特征：

- 卫星 ID。
- 距离。
- 仰角。
- SNR。
- RVT，即剩余可见时间。
- 当前负载或队列压力。

切换决策需要考虑：

- 目标卫星是否可见。
- 目标卫星是否有足够好的链路条件。
- 当前服务卫星剩余可见时间是否不足。
- 切换是否会造成服务中断或任务迁移开销。
- 目标 MEC 队列是否过载。

## 4. MEC 与任务处理

卫星 MEC 参数：

| 参数 | 当前默认值 |
| --- | --- |
| `satellite_cpu_freq_ghz` | `5.0` |
| `satellite_max_cpu_freq_ghz` | `8.0` |
| `satellite_num_cores` | `4` |
| `max_queue_size` | `6` |

用户本地计算参数：

| 参数 | 当前默认值 |
| --- | --- |
| `user_cpu_freq_ghz` | `0.5` |
| `user_max_cpu_freq_ghz` | `1.5` |
| `bandwidth_mhz` | `10.0` |
| `user_tx_power_dbm` | `24.0` |

任务进入系统后，环境会计算：

- 上传时延。
- MEC 排队等待时间。
- MEC 处理时间。
- 下载时延。
- 本地计算时延与能耗。
- 任务是否完成、超时或仍在队列中。

## 5. Reward 组成

当前 reward 使用“QoS 门控任务收益 + 服务连接惩罚”：

- deadline 内完成的任务：
  `1 - 0.60 × clip(时延/deadline) - 0.10 × clip(能耗/10J)`；
- 超时或最终失败的任务：固定为 `-1`；
- 尚未结算的任务：暂不计分，完成或失败后通过 `pending_rewards` 发放；
- 服务中断：按用户在当前时隙内的中断比例计罚，完整中断最多 `-0.30`；
- 切换失败：固定 `-0.20`，旧链路仍有效时不产生服务中断惩罚；
- 全局 reward：所有用户 reward 的算术平均值。

成功切换不再获得额外正奖励，只承担实际切换时延对应的服务中断惩罚。负载均衡、
队列压力、切换次数和动作合法性不进入 reward，分别由评价指标和 action mask 处理。
完整公式、事件语义、默认参数和论文依据见
[Reward 函数设计（方案二）](REWARD_WEIGHT_CONFIG.md)。

## 6. 关键统计指标

训练、评估和基线对比中常用指标：

| 指标 | 含义 |
| --- | --- |
| `mean_reward` | 平均 reward |
| `avg_delay` | 平均任务时延 |
| `total_energy` | 总能耗 |
| `handover_success_rate` | 切换成功率 |
| `handover_failure_rate` | 切换失败率 |
| `handover_frequency` | 单位用户服务时间内的切换频率 |
| `service_continuity_rate` | 服务连续性 |
| `service_availability_rate` | 服务可用性 |
| `task_completion_rate` | 任务完成率 |
| `task_success_rate` | 任务成功率 |
| `task_failure_rate` | 任务失败率 |
| `task_settlement_rate` | 任务结算率 |
| `task_resolution_rate` | 任务解析率 |
| `pending_task_rate` | 未完成或排队任务比例 |
| `energy_per_successful_task` | 每个成功任务的平均能耗 |
| `energy_per_resolved_task` | 每个已解决任务的平均能耗，保留为历史兼容指标 |
| `mec_load_fairness` | 活跃 MEC 服务器之间的 queue/CPU utilization Jain 公平性 |
| `active_load_balance_score` | `mec_load_fairness` 的兼容别名 |
| `avg_load_balance_score` | `mec_load_fairness` 的兼容别名 |

`task_success_rate` is the primary task outcome metric and is computed as `completed_tasks / total_tasks`.
`task_failure_rate` is computed as `deadline_violations / total_tasks`.
`task_settlement_rate` is computed as `(completed_tasks + deadline_violations) / total_tasks`.
The legacy `task_resolution_rate` is kept as an alias of `task_settlement_rate` for backward compatibility, but it should not be used as the main competitiveness metric because deadline failures are counted as settled tasks.
`energy_per_successful_task` is computed as `total_energy / completed_tasks` and
is preferred for paper claims because it avoids rewarding policies that save
energy by completing few tasks.

The custom composite scores `effective_latency_score` and
`latency_priority_score` have been removed from the current evaluation
protocol. Comparisons should report the single paper-style KPIs directly:
delay QoS, task reliability, service continuity/handover behavior, and
resource cost. Checkpoint selection defaults to `avg_delay`; final claims
should use the full metric group and trade-off plots rather than a single
aggregate score.

`mec_load_fairness` is computed from MEC queue pressure and CPU utilization on
active MEC servers only using Jain's fairness index. Idle satellites are excluded
from the fairness denominator so a small number of active MEC servers can still
show meaningful balance differences. It is a resource-side diagnostic for
explaining hotspot avoidance, not a primary ranking KPI. The previous
`mec_activity_score` has been removed from the evaluation protocol because MEC
activity does not directly imply better task success, service continuity, delay,
or energy efficiency.

## 7. 与基线的关系

当前基线设计覆盖不同对照维度：

- `full_local`：验证完全不卸载时的性能下界。
- `min_distance`：验证简单几何切换规则。
- `joint_greedy`：验证一步显式优化能达到的强规则性能。
- `dqn`：验证离散化值函数 RL 方法。
- `mappo_no_han`：验证没有 HAN 图编码时 MAPPO 的性能。

因此，对比主方法时应重点观察：

- HAN+MAPPO 是否降低 `avg_delay`。
- 是否提升 `task_success_rate` 和 `service_continuity_rate`。
- 是否降低 `deadline_violation_rate`、`handover_failure_rate` 和
  `handover_frequency`。
- 是否降低 `energy_per_successful_task`。
- 是否在资源诊断中保持合理的 `mec_load_fairness`，避免 MEC 热点拥塞。
- 是否优于 `mappo_no_han`，从而证明 HAN 结构有效。
