# Reward 权重配置记录

记录时间：2026-07-06

## 背景

本次调整面向 `scripts/train.py` 的默认训练配置，以及 `scripts/compare_system_baselines.py` 的默认系统-基线对比配置。目标从原来的时延优先，调整为更适合论文叙述的多目标优化：最小化任务时延与任务能耗，同时保留 deadline、QoS、服务连续性、切换与负载均衡作为约束或辅助评价因素。

需要注意：环境 reward 公式结构本身没有修改；本记录对应 `scripts/train.py` 中 `TrainConfig` 与 CLI 默认值，以及 `compare_system_baselines.py` 中 `build_default_train_config()` 在未提供已有系统运行目录时生成的默认训练配置。

## 修改前默认参数

| 参数 | 修改前 | 含义 |
| --- | ---: | --- |
| `reward_delay_weight` | 0.35 | 任务时延惩罚权重 |
| `reward_energy_weight` | 0.05 | 任务能耗惩罚权重 |
| `reward_handover_weight` | 0.10 | 切换相关奖励/惩罚权重 |
| `reward_load_balance_weight` | 0.05 | 负载均衡奖励权重 |
| `reward_qos_weight` | 0.40 | 任务按 deadline 成功完成的 QoS 奖励 |
| `reward_service_continuity_weight` | 0.15 | 服务连续性相关权重 |
| `reward_deadline_slack_weight` | 0.25 | deadline 剩余裕量奖励 |
| `reward_enqueue_bonus` | 0.00 | 入队奖励 |
| `reward_deadline_penalty` | 1.00 | deadline 超时惩罚 |
| `reward_failed_task_penalty` | 0.80 | 任务失败固定惩罚 |

该配置的实际倾向是时延优先：除了 `reward_delay_weight` 明显高于 `reward_energy_weight`，`reward_qos_weight`、`reward_deadline_slack_weight`、`reward_deadline_penalty` 和 `reward_failed_task_penalty` 也会间接强化 deadline/时延目标。

## 本次修改后参数

| 参数 | 修改后 | 调整说明 |
| --- | ---: | --- |
| `reward_delay_weight` | 0.25 | 保持时延为核心目标，但降低原先的时延主导性 |
| `reward_energy_weight` | 0.30 | 提高能耗权重，使能耗与时延处于接近地位 |
| `reward_handover_weight` | 0.10 | 保持不变，继续约束不必要切换 |
| `reward_load_balance_weight` | 0.05 | 保持不变，作为辅助负载均衡指标 |
| `reward_qos_weight` | 0.25 | 降低成功完成奖励，避免 QoS 项过度放大 deadline 目标 |
| `reward_service_continuity_weight` | 0.15 | 保持不变，维持服务连续性要求 |
| `reward_deadline_slack_weight` | 0.10 | 降低 deadline 裕量奖励，减少对极低时延的额外偏置 |
| `reward_enqueue_bonus` | 0.00 | 保持不变 |
| `reward_deadline_penalty` | 0.70 | 保留超时约束，但降低其对 reward 的压倒性影响 |
| `reward_failed_task_penalty` | 0.60 | 保留失败惩罚，但与多目标优化保持平衡 |

## 推荐论文表述

可以将系统优化目标表述为：

> 在满足任务 deadline、服务连续性与星载 MEC 资源约束的条件下，联合优化用户切换与任务卸载决策，以最小化归一化任务时延和归一化任务能耗，并提升负载均衡程度。

这比“只最小化时延与能耗”更稳妥，因为当前环境 reward 中仍然保留了 QoS、deadline、服务连续性、切换和负载均衡项。

## 影响范围

- 已修改：`scripts/train.py` 的 `TrainConfig` 默认 reward 权重。
- 已修改：`scripts/train.py` 的 CLI 默认 reward 权重。
- 已修改：`scripts/compare_system_baselines.py` 的 `build_default_train_config()` 默认 reward 权重。
- 未修改：环境 reward 公式本身，即 `src/environment/gym_env.py` 中 `_compute_task_reward()` 的结构。
- 对已有训练结果无影响；只有新运行 `scripts/train.py`，或新运行 `compare_system_baselines.py` 且未通过已有 `--system-run-dir` / `--system-checkpoint` 读取旧配置时，才会使用本次新默认值。
