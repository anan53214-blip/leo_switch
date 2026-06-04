# 系统场景与 Reward 细节

本文档说明当前环境建模、任务卸载流程和 reward 组成。源码以 `src/environment/gym_env.py`、`src/environment/mec.py` 和 `scripts/train.py` 为准。

## 1. 系统场景

系统模拟一个 LEO 卫星 MEC 网络：

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

当前 reward 同时考虑收益与惩罚：

- 时延项：鼓励低时延任务处理。
- 能耗项：鼓励更低能耗。
- QoS 项：鼓励满足任务 deadline 和服务质量。
- 切换项：奖励有效切换，惩罚失败切换和不必要切换成本。
- 负载均衡项：鼓励选择队列压力较小的卫星。
- 入队奖励：任务成功进入 MEC 队列时给予小奖励。
- 队列满惩罚：目标 MEC 队列已满时惩罚。

主要 reward 权重：

| 参数 | 当前默认值 |
| --- | --- |
| `reward_delay_weight` | `0.35` |
| `reward_energy_weight` | `0.05` |
| `reward_handover_weight` | `0.10` |
| `reward_load_balance_weight` | `0.05` |
| `reward_qos_weight` | `0.40` |
| `reward_service_continuity_weight` | `0.15` |
| `reward_deadline_slack_weight` | `0.25` |
| `reward_deadline_penalty` | `1.00` |
| `reward_failed_task_penalty` | `0.80` |
| `reward_enqueue_bonus` | `0.0` |
| `reward_queue_full_penalty` | `0.3` |

`reward_service_continuity` is kept as the legacy breakdown key, but it is now
a signed service-interruption penalty: no interruption contributes `0`, and
interrupted time contributes `-reward_service_continuity_weight *
interruption_seconds / step_user_seconds`.

## 6. 关键统计指标

训练、评估和基线对比中常用指标：

| 指标 | 含义 |
| --- | --- |
| `mean_reward` | 平均 reward |
| `avg_delay` | 平均任务时延 |
| `total_energy` | 总能耗 |
| `handover_success_rate` | 切换成功率 |
| `handover_failure_rate` | 切换失败率 |
| `service_continuity_rate` | 服务连续性 |
| `service_availability_rate` | 服务可用性 |
| `task_completion_rate` | 任务完成率 |
| `task_success_rate` | 任务成功率 |
| `task_failure_rate` | 任务失败率 |
| `task_settlement_rate` | 任务结算率 |
| `task_resolution_rate` | 任务解析率 |
| `pending_task_rate` | 未完成或排队任务比例 |
| `avg_load_balance_score` | MEC queue/CPU utilization load-balance score |

`task_success_rate` is the primary task outcome metric and is computed as `completed_tasks / total_tasks`.
`task_failure_rate` is computed as `deadline_violations / total_tasks`.
`task_settlement_rate` is computed as `(completed_tasks + deadline_violations) / total_tasks`.
The legacy `task_resolution_rate` is kept as an alias of `task_settlement_rate` for backward compatibility, but it should not be used as the main competitiveness metric because deadline failures are counted as settled tasks.

The default `effective_latency_score` is now:

`1 / (1 + avg_delay) * service_continuity_rate * task_success_rate`

This makes deadline failures reduce the primary latency-oriented score directly.
The g1 300k/600s/u10 suite uses `latency_priority_score` by default for
checkpoint selection and comparison ranking so energy remains a secondary
evaluation term instead of being completely ignored.

`avg_load_balance_score` is now computed from MEC queue pressure and CPU
utilization only. It no longer includes the number of users connected to each
satellite, so local-compute policies cannot receive a high load-balance score
merely because serving-satellite connections are geographically spread out.

## 7. 与基线的关系

当前基线设计覆盖不同对照维度：

- `full_local`：验证完全不卸载时的性能下界。
- `min_distance`：验证简单几何切换规则。
- `joint_greedy`：验证一步显式优化能达到的强规则性能。
- `dqn`：验证离散化值函数 RL 方法。
- `mappo_no_han`：验证没有 HAN 图编码时 MAPPO 的性能。

因此，对比主方法时应重点观察：

- HAN+MAPPO 是否降低 `avg_delay`。
- 是否提升 `task_completion_rate` 和 `service_continuity_rate`。
- 是否保持可接受的 `total_energy`。
- 是否优于 `mappo_no_han`，从而证明 HAN 结构有效。
