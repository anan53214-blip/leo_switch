# Reward 函数设计（方案二）

更新时间：2026-07-28
适用环境版本：`environment_schema_version = 5`

## 1. 设计目标

当前 reward 采用“QoS 门控任务收益 + 服务连接惩罚”的简单结构：

- 任务在 deadline 内完成才获得正收益；
- 超时或最终失败统一记为固定负收益；
- 服务中断按实际中断时长处罚；
- 切换失败使用一个固定的小惩罚；
- 负载均衡、队列占用、切换次数等不再重复进入 reward，只作为评价指标。

该设计避免旧 reward 同时使用 QoS、deadline 裕量、时延、入队、队列满和负载均衡等
多个相关项对同一结果重复计分，也使 MAPPO 的训练信号更容易解释。

## 2. 数学定义

对用户 \(i\) 的一个任务，定义归一化时延和能耗：

\[
\hat d_i=\operatorname{clip}\left(\frac{d_i}{D_i},0,1\right),\qquad
\hat e_i=\operatorname{clip}\left(\frac{e_i}{E_{\mathrm{ref}}},0,1\right)
\]

其中：

- \(d_i\)：任务端到端时延；
- \(D_i\)：任务 deadline；
- \(e_i\)：任务总能耗；
- \(E_{\mathrm{ref}}=10\,\mathrm{J}\)：能耗归一化参考值。

单任务 reward 为：

\[
r_i^{task}=
\begin{cases}
1-w_d\hat d_i-w_e\hat e_i, & d_i\le D_i \\
-1, & d_i>D_i\ \text{或任务最终失败} \\
0, & 任务尚未结算
\end{cases}
\]

服务连接 reward 为：

\[
r_i^{link}
=-w_I\operatorname{clip}\left(\frac{t_i^{interrupt}}{\Delta t},0,1\right)
-w_F\mathbb{1}(\text{切换失败})
\]

其中 \(t_i^{interrupt}\) 是用户在当前时隙内的服务中断时间，\(\Delta t\) 是时隙长度。
成功切换或重新接入会计入实际切换时延；用户处于 `BLOCKED` 时按整个时隙中断计算。

用户级和环境级 reward 分别为：

\[
r_i=r_i^{task}+r_i^{link},\qquad
r=\frac{1}{N}\sum_{i=1}^{N}r_i
\]

同一用户在一个时隙内结算多个任务时，各任务 reward 累加后再计算用户级 reward。

## 3. 当前默认配置

| 配置项 | 默认值 | 含义 |
| --- | ---: | --- |
| `reward_delay_weight` | `0.60` | 成功任务的归一化时延惩罚 \(w_d\) |
| `reward_energy_weight` | `0.10` | 成功任务的归一化能耗惩罚 \(w_e\) |
| `reward_interruption_weight` | `0.30` | 整个时隙完全中断时的最大惩罚 \(w_I\) |
| `reward_failed_handover_penalty` | `0.20` | 每次切换失败的固定惩罚 \(w_F\) |
| `REWARD_ENERGY_REFERENCE_J` | `10.0 J` | 固定的能耗归一化参考值 |
| `TASK_SUCCESS_REWARD` | `1.0` | deadline 内完成任务的基础收益 |
| `TASK_FAILURE_PENALTY` | `1.0` | 超时或最终失败任务的固定惩罚绝对值 |

成功任务的单任务 reward 位于 `[0.30, 1.00]`，失败任务固定为 `-1.00`。
因此成功和失败之间存在明确间隔，同时成功任务内部仍可根据时延和能耗排序。

## 4. 各事件如何计分

| 事件 | Reward 处理 |
| --- | --- |
| 任务在 deadline 内完成 | `+1 - 0.60×时延比例 - 0.10×能耗比例` |
| 任务超时、断链后无法继续服务或最终失败 | 固定 `-1`，不再叠加时延、能耗和 deadline 惩罚 |
| 任务仍在本地或 MEC 队列中 | 暂不计分，结算后通过 `pending_rewards` 发放 |
| 成功切换或阻塞后重新接入 | 不额外奖励，只按实际切换时延计算服务中断惩罚 |
| 切换失败但旧链路仍有效 | 原卫星不变，计 `-0.20` |
| 切换失败且旧链路已失效 | 计 `-0.20`，用户进入 `BLOCKED`，并按中断时长计罚 |
| 非法候选动作 | 动作掩码负责排除；环境防御性地按 stay 处理 |
| MEC 队列满 | 不单独处罚，由任务最终是否成功和实际时延反映后果 |
| 负载不均衡 | 不进入 reward，仅通过负载公平性指标报告 |

切换失败惩罚表示“无效决策事件”，服务中断惩罚表示“实际不可服务时长”，二者含义不同。
旧链路仍可服务时不会产生中断惩罚。

## 5. Reward 分解日志

训练历史和评估结果只记录以下 6 个 reward 分量：

| 字段 | 含义 |
| --- | --- |
| `reward_task_success` | 成功任务基础收益 |
| `penalty_delay` | 成功任务时延惩罚 |
| `penalty_energy` | 成功任务能耗惩罚 |
| `penalty_task_failure` | 失败任务固定惩罚 |
| `penalty_service_interruption` | 用户级服务中断惩罚 |
| `penalty_failed_handover` | 切换失败固定惩罚 |

所有分量都按用户数换算为对环境全局平均 reward 的实际贡献，因此同一统计周期内
6 个分量之和可还原该周期的累计全局 reward。

## 6. 已删除的旧 Reward 项

以下字段已经从环境、训练配置、命令行和绘图入口中删除：

- `reward_handover_weight`
- `reward_load_balance_weight`
- `reward_qos_weight`
- `reward_service_continuity_weight`
- `reward_deadline_slack_weight`
- `reward_enqueue_bonus`
- `reward_invalid_action_penalty`
- `reward_blocked_penalty`
- `reward_queue_full_penalty`
- `reward_deadline_penalty`
- `reward_failed_task_penalty`

历史结果目录中的旧配置作为实验记录保留，不代表当前默认配置。

## 7. 论文依据

1. Zhu 等人在卫星—地面边缘计算卸载中，将优化代价写成时延与能耗的加权和。
   本系统据此保留两个最直接、可归一化的任务成本项。
   D. Zhu et al., *Deep Reinforcement Learning-based Task Offloading in
   Satellite-Terrestrial Edge Computing Networks*, 2021.
   <https://arxiv.org/abs/2102.01876>

2. Lee 等人将 LEO 切换 reward 设计为接入时延与碰撞代价的简单线性组合。
   本系统据此用“实际服务中断时长 + 切换失败事件”描述连接代价，不再奖励切换本身。
   J.-H. Lee et al., *Handover Protocol Learning for LEO Satellite Networks:
   Access Delay and Collision Minimization*, 2023.
   <https://arxiv.org/abs/2310.20215>

3. He、Wang 和 Wang 的负载感知卫星切换研究采用按可行性分段的 reward：
   不可用动作和过载状态直接处罚，可服务状态才获得收益。本系统据此使用成功/失败门控，
   但把负载从 reward 移到独立评价指标，减少目标耦合。
   S. He, T. Wang, and S. Wang, *Load-Aware Satellite Handover Strategy Based
   on Multi-Agent Reinforcement Learning*, IEEE GLOBECOM, 2020.
   <https://doi.org/10.1109/GLOBECOM42002.2020.9322449>

4. Huang 等人的 SAGIN 联合卸载研究使用时延或能耗的简单目标，并通过动作可用性辅助
   处理不可行动作。本系统据此继续使用 action mask，而不是在 reward 中加入较大的
   “非法动作惩罚”。
   C. Huang et al., *Joint Offloading and Resource Allocation for Hybrid Cloud
   and Edge Computing in SAGINs*, 2024.
   <https://arxiv.org/abs/2401.01140>

这些论文提供的是设计原则而非完全相同的公式；当前系数仍需通过本系统的消融实验验证。

## 8. 实验与兼容性要求

- Reward 语义变更后，环境版本已升级为 `5`。
- 版本 5 之前的 checkpoint 不允许直接恢复训练。
- 新旧 reward 的回报数值不能直接比较；正式基线对比必须在版本 5 下重新训练所有方法。
- 调参时建议每次只改变一个权重，并同时报告任务成功率、平均时延、服务连续率、
  切换失败率和每成功任务能耗，而不是只比较平均 reward。

实现位置：

- 环境公式与默认值：`src/environment/gym_env.py`
- 训练配置与命令行：`scripts/train.py`
- 基线 reward 估计：`scripts/compare_system_baselines.py`
- reward 分量绘图：`scripts/plot_training_artifacts.py`
