# multiseed_0907 指标独立图

来源：`results/multiseed_0907` 当前结果；5 个训练种子 42–46、用户规模 20/25/30/35/40。
结果图默认包含 7 种方法；按用户要求，`blocked_user_seconds.png` 仅比较除 Full-Local 以外的 6 种方法。训练图只包含实际保存了该项轨迹的方法。

- 根目录：评估结果随用户规模变化，一指标一图，文件名使用 CSV 指标名。
- `training/`：训练过程随 episode 变化，同一指标的五个用户规模放在一张图中。
- 沿用原图 Times New Roman/serif、算法颜色与标记；红色星形为 HAN+PDQN。
- 结果图直接使用原表均值及训练种子层面的 95% bootstrap CI。
- 比例转百分比，时延转 ms，吞吐量保持 tasks/(user·min)，奖励惩罚保留负值。
- 柱状图从零起；有序规模趋势采用折线，吞吐量、计数和奖励贡献采用分组柱状，稀疏事件采用点线。
- 阻塞比例及中断相关图保留完整主轴，并附靠近零的局部放大；`blocked_user_seconds.png` 已移除 Full-Local 及放大窗，直接展示其余 6 种方法。
- 原始结果、历史记录及原始图片均未修改。

## 统计口径

当前汇总包含 U20/seed43、U30/seed45 两次 HAN+PDQN 修复后的结果。用户于 2026-10-29 确认，复跑原因是断电导致训练中断及数据异常；图表使用修复后的有效结果，不将这两次复跑视为择优替换。
成功率以生成任务数为分母；完成率以已结算任务数为分母，两者不能互换。
每成功任务能耗包含失败及已启动未完成任务已记账的能量。P95 是各评估回合 P95 的均值。
Full-Local 没有切换或 MEC 负载，其零失败率及公平性极值不表示切换或负载管理更优。

## 结果图清单

| 指标文件 | 标题 | 图型 | 单位/纵轴 |
|---|---|---|---|
| [avg_delay](avg_delay.png) | Resolved-Task Delay | 折线＋95% CI | Average Delay (ms) |
| [avg_success_delay](avg_success_delay.png) | Successful-Task Delay | 折线＋95% CI | Average Delay (ms) |
| [p95_success_delay](p95_success_delay.png) | P95 Successful-Task Delay | 折线＋95% CI | P95 Delay (ms) |
| [energy_per_successful_task](energy_per_successful_task.png) | Energy per Successful Task | 折线＋95% CI | Energy / Successful Task (J) |
| [energy_per_resolved_task](energy_per_resolved_task.png) | Energy per Resolved Task | 折线＋95% CI | Energy / Resolved Task (J) |
| [task_success_rate](task_success_rate.png) | Task Success Rate | 折线＋95% CI | Task Success Rate (%) |
| [deadline_violation_rate](deadline_violation_rate.png) | Deadline Violation Rate | 折线＋95% CI | Deadline Violation Rate (%) |
| [task_failure_rate](task_failure_rate.png) | Task Failure Rate | 折线＋95% CI | Task Failure Rate (%) |
| [task_completion_rate](task_completion_rate.png) | Resolved-Task Completion Rate | 折线＋95% CI | Task Completion Rate (%) |
| [pending_task_rate](pending_task_rate.png) | Pending-Task Rate | 折线＋95% CI | Pending-Task Rate (%) |
| [task_settlement_rate](task_settlement_rate.png) | Task Settlement Rate | 折线＋95% CI | Task Settlement Rate (%) |
| [task_resolution_rate](task_resolution_rate.png) | Task Resolution Rate | 折线＋95% CI | Task Resolution Rate (%) |
| [blocked_time_ratio](blocked_time_ratio.png) | Blocked User-Time Ratio | 折线＋95% CI | Blocked User-Time Ratio (%) |
| [service_availability_rate](service_availability_rate.png) | Service Availability | 折线＋95% CI | Service Availability (%) |
| [service_continuity_rate](service_continuity_rate.png) | Service Continuity | 折线＋95% CI | Uninterrupted User-Time (%) |
| [forced_termination_rate](forced_termination_rate.png) | Forced Disconnection Event Ratio | 折线＋95% CI | Forced Disconnection Events (%) |
| [handover_success_rate](handover_success_rate.png) | Handover Success Rate | 折线＋95% CI | Handover Success Rate (%) |
| [handover_failure_rate](handover_failure_rate.png) | Handover Failure Rate | 折线＋95% CI | Handover Failure Rate (%) |
| [handover_frequency](handover_frequency.png) | Handover Frequency | 折线＋95% CI | Handovers / User-Second |
| [handovers_per_user_minute](handovers_per_user_minute.png) | Handover Frequency | 折线＋95% CI | Handovers / User-Minute |
| [handover_action_rate](handover_action_rate.png) | Handover Action Rate | 折线＋95% CI | Handover Actions (%) |
| [local_compute_rate](local_compute_rate.png) | Local-Mode Action Rate | 折线＋95% CI | Local-Mode Actions (%) |
| [mean_offload_ratio](mean_offload_ratio.png) | Mean Offloading Ratio | 折线＋95% CI | Offloading Ratio (%) |
| [jain_mec_load_fairness](jain_mec_load_fairness.png) | MEC Load Jain Fairness | 折线＋95% CI | Jain Fairness Index |
| [reachable_jain_mec_load_fairness](reachable_jain_mec_load_fairness.png) | Reachable-MEC Load Jain Fairness | 折线＋95% CI | Jain Fairness Index |
| [load_balance_variance](load_balance_variance.png) | MEC Load Variance | 折线＋95% CI | Normalized Load Variance |
| [load_balance_coefficient](load_balance_coefficient.png) | MEC Load-Balance Coefficient | 折线＋95% CI | Load-Balance Coefficient |
| [mec_load_fairness](mec_load_fairness.png) | MEC Load-Balance Coefficient | 折线＋95% CI | Load-Balance Coefficient |
| [active_load_balance_score](active_load_balance_score.png) | Active Load-Balance Score | 折线＋95% CI | Load-Balance Coefficient |
| [avg_load_balance_score](avg_load_balance_score.png) | Average Load-Balance Score | 折线＋95% CI | Load-Balance Coefficient |
| [active_mec_load_fairness](active_mec_load_fairness.png) | Legacy Active-MEC Load Fairness | 折线＋95% CI | Legacy Load-Fairness Score |
| [mean_reward](mean_reward.png) | Evaluation Reward | 折线＋95% CI | Mean Episode Reward |
| [std_reward](std_reward.png) | Evaluation Reward Variability | 折线＋95% CI | Within-Seed Reward SD |
| [selection_score](selection_score.png) | Model Selection Score | 折线＋95% CI | Selection Score (Reward) |
| [successful_task_throughput](successful_task_throughput.png) | Per-User Successful-Task Throughput | 分组柱状＋95% CI | Tasks / User-Minute |
| [total_energy](total_energy.png) | Total Terminal Energy | 分组柱状＋95% CI | Energy / Episode (J) |
| [completed_tasks](completed_tasks.png) | Successful Tasks | 分组柱状＋95% CI | Tasks / Episode |
| [deadline_violations](deadline_violations.png) | Deadline-Violating Tasks | 分组柱状＋95% CI | Tasks / Episode |
| [failed_tasks](failed_tasks.png) | Other Failed Tasks | 分组柱状＋95% CI | Tasks / Episode |
| [pending_tasks](pending_tasks.png) | Pending Tasks | 分组柱状＋95% CI | Tasks / Episode |
| [resolved_tasks](resolved_tasks.png) | Resolved Tasks | 分组柱状＋95% CI | Tasks / Episode |
| [total_tasks](total_tasks.png) | Generated Tasks | 分组柱状＋95% CI | Tasks / Episode |
| [total_handovers](total_handovers.png) | Total Handover Attempts | 分组柱状＋95% CI | Attempts / Episode |
| [handover_attempts](handover_attempts.png) | Handover Attempts | 分组柱状＋95% CI | Attempts / Episode |
| [handover_committed](handover_committed.png) | Committed Handovers | 分组柱状＋95% CI | Handovers / Episode |
| [handover_aborted](handover_aborted.png) | Aborted Handovers | 折线＋95% CI | Handovers / Episode |
| [failed_handovers](failed_handovers.png) | Failed Handovers | 折线＋95% CI | Handovers / Episode |
| [handover_radio_failures](handover_radio_failures.png) | Handover Radio Failures | 折线＋95% CI | Failures / Episode |
| [migration_rejections](migration_rejections.png) | Migration Rejections | 折线＋95% CI | Rejections / Episode |
| [reconnection_attempts](reconnection_attempts.png) | Reconnection Attempts | 分组柱状＋95% CI | Attempts / Episode |
| [reconnections](reconnections.png) | Successful Reconnections | 分组柱状＋95% CI | Reconnections / Episode |
| [blocked_user_seconds](blocked_user_seconds.png) | Blocked User-Time | 分组柱状＋95% CI | Blocked Time / Episode (user-s) |
| [handover_interruption_seconds](handover_interruption_seconds.png) | Handover Interruption Time | 分组柱状＋95% CI | Interruption / Episode (user-s) |
| [service_interruption_seconds](service_interruption_seconds.png) | Service Interruption Time | 分组柱状＋95% CI | Interruption / Episode (user-s) |
| [total_user_seconds](total_user_seconds.png) | Total User-Time Exposure | 分组柱状＋95% CI | Observation Time / Episode (user-s) |
| [reward_task_success](reward_task_success.png) | Task-Success Reward | 分组柱状＋95% CI | Reward Contribution / Episode |
| [reward_load_balance](reward_load_balance.png) | Load-Balance Reward | 分组柱状＋95% CI | Reward Contribution / Episode |
| [penalty_delay](penalty_delay.png) | Delay Penalty | 分组柱状＋95% CI | Reward Contribution / Episode |
| [penalty_energy](penalty_energy.png) | Energy Penalty | 分组柱状＋95% CI | Reward Contribution / Episode |
| [penalty_task_failure](penalty_task_failure.png) | Task-Failure Penalty | 分组柱状＋95% CI | Reward Contribution / Episode |
| [penalty_service_interruption](penalty_service_interruption.png) | Service-Interruption Penalty | 分组柱状＋95% CI | Reward Contribution / Episode |
| [penalty_failed_handover](penalty_failed_handover.png) | Failed-Handover Penalty | 折线＋95% CI | Reward Contribution / Episode |
| [primary_metric_win_count](primary_metric_win_count.png) | Primary-Metric Wins | 分组柱状＋95% CI | Mean Wins per Seed (out of 6) |

## 重复字段

按要求保留每个有数据的指标文件；以下字段在本批结果中是重复表达，不应在论文中当作独立证据：

- `mec_load_fairness` 与 `load_balance_coefficient` 数值一致。
- `active_load_balance_score` 与 `load_balance_coefficient` 数值一致。
- `avg_load_balance_score` 与 `load_balance_coefficient` 数值一致。
- `reachable_jain_mec_load_fairness` 与 `jain_mec_load_fairness` 数值一致。
- `task_resolution_rate` 与 `task_settlement_rate` 数值一致。
- `selection_score` 与 `mean_reward` 数值一致。
- `total_handovers` 与 `handover_attempts` 数值一致。

`failed_handovers` 与 `handover_aborted` 在本批结果中也数值一致，但保留各自事件名称。

## 不单独绘图的元数据

- `num_users`：用户规模横轴，不是待比较的结果指标。
- `seed`：训练种子标识。
- `seed_count`：有效训练种子数量；用于说明统计覆盖。
- `episodes`：每次评估的回合数，当前恒为 5。
- `trained_timesteps`：训练预算元数据，存在缺失且有效值恒定。
- `best_training_eval_reward`：只在部分方法中保存的模型选择记录，不能替代统一测试结果。
- `selected_offload`：仅单一基线保存的固定参数，不能形成跨方法比较。
- `load_variance_sample_count`：负载统计采样数量，不作为性能指标。
- `*_ci_low`、`*_ci_high` 用于误差范围；`*_sample_count` 用于统计覆盖，不单独出图。

## 训练曲线

每个 episode 为 512 环境步；优先使用记录的真实累计 `episodes`，并排除 `partial_episode=True` 的未完整回合。
训练指标仅保留预算内的完整回合（最多 292）。PPO 每 1024 步记录一次，保留 2、4、… 的实际采样位置，不插造逐回合数据。
PPO 的累计计数、总能耗和奖励分项除以记录覆盖的回合数，统一为每回合；已归一化比例、时延和平均奖励不重复除。
训练曲线不平滑，置信区间按完整训练种子轨迹 bootstrap 2000 次得到。稀疏训练评估的横轴采用准确 steps/512，可出现小数 episode 等价位置。
缺失字段不补零。PDQN 的吞吐训练轨迹只有 4 个种子，图例明确标注；其余覆盖见下表及 plot_manifest.json。
不同算法的 loss 和 recent_mean_reward 窗口定义可能不同，这些图仅为各自优化过程的诊断。

| 指标文件 | 标题 | 实际支持的方法与训练种子数 |
|---|---|---|
| [active_load_balance_score](training/active_load_balance_score.png) | Active Load Balance Score | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [active_mec_load_fairness](training/active_mec_load_fairness.png) | Active MEC Load Fairness | HAN+PDQN (n=5) |
| [actor_grad_norm](training/actor_grad_norm.png) | Actor Grad Norm | MADDPG (n=5) |
| [actor_loss](training/actor_loss.png) | Actor Loss | HAN+MAPPO (n=5), MAPPO (n=5), MADDPG (n=5) |
| [avg_delay](training/avg_delay.png) | Resolved-Task Delay | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [avg_load_balance_score](training/avg_load_balance_score.png) | Avg Load Balance Score | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [avg_success_delay](training/avg_success_delay.png) | Average Successful-Task Delay | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [blocked_time_ratio](training/blocked_time_ratio.png) | Blocked Time Ratio | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [blocked_user_seconds](training/blocked_user_seconds.png) | Blocked User Seconds | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [clip_fraction](training/clip_fraction.png) | Clip Fraction | HAN+MAPPO (n=5), MAPPO (n=5) |
| [completed_tasks](training/completed_tasks.png) | Completed Tasks | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=4) |
| [critic_grad_norm](training/critic_grad_norm.png) | Critic Grad Norm | MADDPG (n=5) |
| [critic_loss](training/critic_loss.png) | Critic Loss | HAN+MAPPO (n=5), MAPPO (n=5), MADDPG (n=5) |
| [deadline_violation_rate](training/deadline_violation_rate.png) | Deadline Violation Rate | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [deadline_violations](training/deadline_violations.png) | Deadline Violations | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [energy_per_resolved_task](training/energy_per_resolved_task.png) | Energy per Resolved Task | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [energy_per_successful_task](training/energy_per_successful_task.png) | Energy per Successful Task | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [entropy](training/entropy.png) | Entropy | HAN+MAPPO (n=5), MAPPO (n=5) |
| [epsilon](training/epsilon.png) | Discrete Exploration Probability | HAN+PDQN (n=5), PDQN (n=5) |
| [exploration_noise](training/exploration_noise.png) | Continuous Exploration Noise | MADDPG (n=5) |
| [failed_handovers](training/failed_handovers.png) | Failed HANdovers | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [failed_tasks](training/failed_tasks.png) | Failed Tasks | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [forced_disconnects](training/forced_disconnects.png) | Forced Disconnects | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [forced_termination_rate](training/forced_termination_rate.png) | Forced Termination Rate | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [han_grad_norm](training/han_grad_norm.png) | HAN Grad Norm | HAN+MAPPO (n=5) |
| [han_metapaths_executed](training/han_metapaths_executed.png) | HAN Meta-Path Executions | HAN+MAPPO (n=5) |
| [han_parameter_delta](training/han_parameter_delta.png) | HAN Parameter Delta | HAN+MAPPO (n=5) |
| [handover_aborted](training/handover_aborted.png) | HANdover Aborted | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [handover_attempts](training/handover_attempts.png) | HANdover Attempts | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [handover_committed](training/handover_committed.png) | HANdover Committed | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [handover_failure_rate](training/handover_failure_rate.png) | HANdover Failure Rate | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [handover_frequency](training/handover_frequency.png) | Handover Frequency | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [handover_interruption_seconds](training/handover_interruption_seconds.png) | HANdover Interruption Seconds | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [handover_radio_failures](training/handover_radio_failures.png) | HANdover Radio Failures | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [handover_success_rate](training/handover_success_rate.png) | HANdover Success Rate | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [handovers_per_user_minute](training/handovers_per_user_minute.png) | Handover Frequency | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [jain_mec_load_fairness](training/jain_mec_load_fairness.png) | MEC Load Jain Fairness | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [kl_divergence](training/kl_divergence.png) | Kl Divergence | HAN+MAPPO (n=5), MAPPO (n=5) |
| [load_balance_coefficient](training/load_balance_coefficient.png) | MEC Load-Balance Coefficient | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [load_balance_variance](training/load_balance_variance.png) | MEC Load Variance | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [mean_reward](training/mean_reward.png) | Training Episode Return | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), MADDPG (n=5), PDQN (n=5) |
| [mec_load_fairness](training/mec_load_fairness.png) | MEC Load Fairness | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [migration_rejections](training/migration_rejections.png) | Migration Rejections | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [p95_success_delay](training/p95_success_delay.png) | P95 Successful-Task Delay | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [param_loss](training/param_loss.png) | Param Loss | HAN+PDQN (n=5), PDQN (n=5) |
| [penalty_delay](training/penalty_delay.png) | Penalty Delay | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [penalty_energy](training/penalty_energy.png) | Penalty Energy | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [penalty_failed_handover](training/penalty_failed_handover.png) | Penalty Failed HANdover | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [penalty_service_interruption](training/penalty_service_interruption.png) | Penalty Service Interruption | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [penalty_task_failure](training/penalty_task_failure.png) | Penalty Task Failure | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [pending_task_rate](training/pending_task_rate.png) | Pending Task Rate | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [pending_tasks](training/pending_tasks.png) | Pending Tasks | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [q_loss](training/q_loss.png) | Q Loss | HAN+PDQN (n=5), PDQN (n=5) |
| [q_mean](training/q_mean.png) | Q Mean | MADDPG (n=5) |
| [reachable_jain_mec_load_fairness](training/reachable_jain_mec_load_fairness.png) | Reachable MEC Load Jain Fairness | HAN+PDQN (n=5) |
| [recent_mean_reward](training/recent_mean_reward.png) | Recorded Recent Episode Return | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), MADDPG (n=5), PDQN (n=5) |
| [reconnection_attempts](training/reconnection_attempts.png) | Reconnection Attempts | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [reconnections](training/reconnections.png) | Reconnections | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [resolved_tasks](training/resolved_tasks.png) | Resolved Tasks | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [reward_load_balance](training/reward_load_balance.png) | Reward Load Balance | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [reward_task_success](training/reward_task_success.png) | Reward Task Success | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [rollout_mean_reward](training/rollout_mean_reward.png) | Mean Step Reward | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [rollout_total_reward](training/rollout_total_reward.png) | Rollout Return per Episode | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [service_availability_rate](training/service_availability_rate.png) | Service Availability Rate | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [service_continuity_rate](training/service_continuity_rate.png) | Service Continuity Rate | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [service_interruption_seconds](training/service_interruption_seconds.png) | Service Interruption Seconds | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [std_reward](training/std_reward.png) | Within-Rollout Episode Return Variation | HAN+MAPPO (n=5), MAPPO (n=5) |
| [successful_handovers](training/successful_handovers.png) | Successful HANdovers | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [successful_task_throughput](training/successful_task_throughput.png) | Per-User Successful-Task Throughput | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=4) |
| [target_q_mean](training/target_q_mean.png) | Target Q Mean | MADDPG (n=5) |
| [task_completion_rate](training/task_completion_rate.png) | Task Completion Rate | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [task_failure_rate](training/task_failure_rate.png) | Task Failure Rate | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [task_resolution_rate](training/task_resolution_rate.png) | Task Resolution Rate | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [task_settlement_rate](training/task_settlement_rate.png) | Task Settlement Rate | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [task_success_rate](training/task_success_rate.png) | Task Success Rate | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [td_abs_mean](training/td_abs_mean.png) | Td Abs Mean | MADDPG (n=5) |
| [total_energy](training/total_energy.png) | Terminal Energy per Episode | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=5) |
| [total_handovers](training/total_handovers.png) | Total HANdovers | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [total_tasks](training/total_tasks.png) | Total Tasks | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5) |
| [total_user_seconds](training/total_user_seconds.png) | Total User Seconds | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), PDQN (n=4) |
| [eval_mean_reward](training/eval_mean_reward.png) | Training-Checkpoint Evaluation Return | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), MADDPG (n=5), PDQN (n=5) |
| [eval_std_reward](training/eval_std_reward.png) | Training-Checkpoint Evaluation Return Variation | HAN+PDQN (n=5), HAN+MAPPO (n=5), MAPPO (n=5), MADDPG (n=5), PDQN (n=5) |

## 图型参考

- [Shen et al., IEEE TMC 2024](https://shen-hang.github.io/files/TMC2024.pdf)：Fig.9 完成率分组柱状；Fig.10–11 失败率与完成率随参数变化的折线。
- [Dai et al., IEEE TMC 2025](https://www.cs.newpaltz.edu/~lik/publications/Longbao-Dai-IEEE-TMC-2025.pdf)：Fig.7(b) 吞吐分组柱状与误差棒；Fig.6(c) 吞吐时序折线。
- [Huang et al., IEEE JSAC 2024](https://arxiv.org/pdf/2401.01140)：Fig.7 能耗、Fig.11 时延随用户数变化的带标记折线。

## 复现

```powershell
python scripts/plot_individual_multiseed.py
```
