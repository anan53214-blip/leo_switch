# LEO_switch 实验日志

本文记录会影响训练或评估行为的实验结果、诊断和代码变更。每次实验都应新增记录，
包括失败或结论不明确的实验。

## 记录清单

- 实验目录和命令
- 代码版本或本地修改摘要
- 目标、模型选择指标、随机种子、训练步数、每回合步数和用户数
- 核心指标：reward、时延、服务连续率、任务成功率、deadline 违约率和能耗
- Reward 分量：任务成功、时延、能耗、任务失败、服务中断和切换失败
- 结果解释：改善项、退化项和可能原因
- 后续决定：保留、回退、调参或重跑

## 2026-07-28 - 论文评价指标与多种子统计协议

**代码变更：**

- 新增 `scripts/paper_metrics.py`，固定用户和多用户绘图共用同一组字段、单位、
  指标优劣方向和 bootstrap 95% 置信区间实现。
- 新增成功任务平均时延、成功任务 P95 时延、阻塞时间占比、单位用户分钟切换
  次数、单位成功任务能耗和全 MEC 节点 Jain 公平指数。
- 固定用户核心图改为成功任务时延、P95 时延、任务成功率、截止期违反率、
  服务连续率和单位成功任务能耗；柱状图可读取 `seed_metrics` 绘制置信区间。
- Reward 收敛图只绘制有真实训练历史的学习算法，并按共同训练步数对齐多个
  种子后绘制均值和 95% 置信区间。
- 多用户脚本增加 `--seeds` 和 `uXX/seed_XX` 目录，保存
  `comparison_seed_records.csv` 与种子聚合后的 `comparison_summary.csv`；
  扩展性图使用种子级均值和置信区间。
- 聚合阶段拒绝环境版本不是 5、指标版本不是 2、缺少统一 Reward 配置或
  Reward 权重不一致的结果。

**统计口径：**

- 评估回合先在各训练种子内部取均值，再对种子级均值做 bootstrap；
- 正式论文实验建议至少 5 个训练种子，每个种子至少 10 个独立评估回合；
- 历史版本结果必须重新评估或重训，不能与环境版本 5 的结果混合。

**验证：**

- 新增指标公式、Jain 公平性、schema 拒绝、多种子聚合、Reward 曲线步数对齐
  和绘图冒烟测试。
- Python 编译检查通过。
- 固定用户柱状图和多用户置信区间曲线的合成数据绘图冒烟通过。
- `compare_system_baselines.py --help` 与
  `run_multiuser_scaling_suite.py --help` 均可直接运行。
- 全量测试：128 项通过。

详细设计、指标依据和正式运行命令见 `docs/论文图评价指标修改方案.md`。

### 正式训练前复查修复

**发现与修复：**

- 训练器的跨 episode 统计容器原先没有累计
  `successful_task_delay_samples`、`jain_load_fairness_sum` 和
  `jain_load_fairness_samples`，导致训练历史中的成功任务平均/P95 时延和
  Jain 公平指数错误为 0；现已补齐。
- 使用成功任务时延或单位成功任务能耗选择 checkpoint 时，零成功任务记录
  原先可能凭借数值 0 被误选；现统一返回负无穷选优分数。
- 固定用户图不再使用同一模型的评估 episode 伪装训练种子计算置信区间；
  只有存在 `seed_metrics` 时才绘制种子级 bootstrap 区间。
- 多用户局部聚合现在会把 `output_suffix` 传递到固定用户图，`no_han`
  等局部图不会覆盖完整算法图。

**真实短训练验证：**

- 临时目录：
  `C:\Users\19704\AppData\Local\Temp\leo_switch_metrics_fix_54303a1d9edf4a40a407b056ab7a1c0a`
- 配置：2 用户、8 总步数、每次 rollout 4 步、CPU、seed 42、
  `graph_update_interval=1`、环境版本 5、最佳模型指标 `reward`。
- 第一条训练记录包含 3 个成功任务，修复后
  `avg_success_delay=1.008602849 s`、
  `p95_success_delay=1.669737293 s`，不再错误为 0。
- rollout、参数更新、独立评估、best/checkpoint/final 模型和
  `training_history.json` 均正常生成。
- 修复后专项测试：52 项通过；全量测试：134 项通过。

### 第二轮 P0 数据完整性修复

**训练完成判定：**

- 多用户套件不再因为目录中存在任意一个 checkpoint 就认定训练完成。
- 只有 `final_model.pt` 与可读取的 `training_history.json` 同时存在，
  环境版本为 5，且 seed、用户数、训练步数、episode 长度、rollout 长度、
  评估/保存间隔、图更新间隔、最佳模型指标和 Reward 配置与当前命令一致时，
  才允许复用。
- 禁用早停时，历史实际步数必须达到目标训练步数；启用早停且训练正常落盘时，
  允许实际步数小于目标值。
- 若发现部分产物或配置不一致，脚本直接报错，要求使用新 `run-id`，或显式使用
  `--force-system-train` 重新训练，避免静默比较未完成模型。

**零成功任务处理：**

- `comparison_summary.csv` 新增总任务数、成功任务数、已结算任务数、待处理任务数
  和 deadline 违反次数，保证多种子聚合可以判断指标分母。
- 当某个种子 `completed_tasks=0` 时，成功任务平均/P95 时延和单位成功任务能耗
  作为无有效样本处理，不参与均值、bootstrap 区间、最佳指标或 Pareto 图。
- 聚合 CSV 为三项成功依赖指标增加有效样本数，全部种子均无成功任务时图中显示
  `N/A`，不再把数值 0 解释为最优。

**验证：**

- P0 专项及相关配置测试：33 项通过。
- 完整训练、仅有 `best_model.pt`、训练步数不足、seed 不匹配和合法早停均有
  独立回归测试。
- 多用户两用户规模、两 seed 正式参数 dry-run 通过。
- 全量测试：139 项通过。

## 2026-07-28 - Reward 方案二（环境版本 5）

**代码变更：**

- 任务 reward 改为 QoS 门控结构：deadline 内完成时为
  `1 - 0.60×时延比例 - 0.10×能耗比例`，超时或最终失败固定为 `-1`。
- 服务连接只保留两个惩罚：实际中断比例最高 `-0.30`，切换失败固定 `-0.20`。
- 成功切换不再获得额外奖励；入队、队列满、负载均衡、deadline 裕量和非法动作等
  相关项从 reward 中删除。
- 负载公平性、切换行为、任务可靠性和队列状态继续作为独立指标报告。
- Reward 分解统一为任务成功、时延、能耗、任务失败、服务中断和切换失败 6 项。
- 6 个分量统一采用全局用户平均尺度，相加可还原同一周期的累计全局 reward。
- `environment_schema_version` 升级为 `5`。

**论文依据：**

- Zhu et al. (2021)：卫星边缘卸载中的时延—能耗加权代价。
- Lee et al. (2023)：LEO 切换中的接入时延与失败事件代价。
- He et al. (GLOBECOM 2020)：按服务可行性分段的切换 reward。
- Huang et al. (2024)：SAGIN 卸载的简单目标与不可行动作辅助。

详细公式和文献链接见 `docs/REWARD_WEIGHT_CONFIG.md`。

**兼容性：** 版本 5 之前的 checkpoint 不能直接恢复；新旧平均 reward 不可直接比较，
正式实验需要在版本 5 下重新训练所有方法。

**验证：**

- Reward 专项测试：14 项通过。
- 全量测试：119 项通过。
- 2 用户、8 step、CPU MAPPO 短训练通过。
- checkpoint 和训练历史均为环境版本 5，只包含 4 个当前 reward 配置字段和 6 个
  当前 reward 分量；各分量之和与累计全局 reward 的误差小于 `5×10^-8`。
- 临时训练产物在验证后清理。

## 2026-07-28 - 原子切换与阻塞语义修复

**代码变更：**

- 切换取消随机成功判定，改为目标卫星可见性、仰角、RVT、SNR 和 MEC
  容量的确定性准入检查。
- MEC 迁移改为“预检查 + 原子提交”。只有全部未完成任务都能进入目标队列，
  才迁移任务并修改用户服务卫星；任一任务无法迁移时两个队列均不变。
- 旧链路仍有效时，切换失败只回滚并处罚，用户继续连接原卫星；旧链路已经
  失效时才进入 `BLOCKED`，并将无法继续服务的旧卫星任务显式结算为失败。
- 阻塞用户重新接入单独记录，不再混入普通切换成功率。
- 新增 `handover_attempts`、`handover_committed`、`handover_aborted`、
  `handover_radio_failures`、`migration_rejections`、
  `reconnection_attempts`、`reconnections` 和 `failed_tasks`。
  切换频率只使用已提交切换计数。
- checkpoint 和训练历史的 `environment_schema_version` 升级为 4。

**验证：**

- 原子迁移、失败回滚、阻塞和重新接入专项测试：6 passed
- 受影响模块回归：63 passed
- 全量测试：117 passed
- 2 用户、8 step、CPU 训练与两次独立评估通过；checkpoint 和训练历史
  均为环境版本 4，且包含新增指标；临时结果已清理

**影响：** 切换成功、用户阻塞、任务失败和切换频率的定义均已改变。
环境版本 4 之前的 checkpoint 和实验结果不能与新结果直接比较，正式比较需
重新训练所有方法。

## 2026-07-28 - 系统冗余代码清理

**代码变更：**

- 新增统一的 `build_env_config`，训练与基线对比入口不再分别维护
  `EnvConfig` 字段清单；新增环境字段会自动从训练配置传递。
- 删除已废弃的 `han_attn_cpq` 算法入口及其重复训练/评估实现，当前候选注意力
  HAN 方法统一使用 `han_attn`。
- 删除固定参数的一次性实验脚本
  `run_latency_priority_g1_300k_600s_u20_suite.py`，保留通用训练、系统对比、
  多用户扩展和绘图入口。
- 删除未被系统调用的旧标量星座坐标路径、图格式转换与调试辅助函数、旧
  Critic、旧模型保存接口、非原子迁移包装器以及多用户信道占位类。
- 清理各模块未使用的导入、兼容属性和重复配置赋值；历史实验记录保留，
  但活动代码和当前使用文档不再引用废弃名称。

**验证：**

- Python 全量编译通过。
- 全量测试：118 passed。
- `train.py --help` 与 `compare_system_baselines.py --help` 正常加载。
- 2 用户、8 step、CPU 的 MAPPO 真实训练通过并生成最终 checkpoint；
  验证产物已清理。

**兼容性：** 本次只删除不可达或已明确废弃的接口，不改变环境行为和模型结构，
因此不提升 `environment_schema_version` 或模型版本。旧命令中的
`han_attn_cpq` 需要改为 `han_attn`。

## 2026-07-05 - Repository Slimming Cleanup

**Code change:** removed historical one-off suite wrappers, the old
`plot_results.py` plotting entry, stale artifact-regeneration/cleanup scripts,
plot-wrapper tests, the broken `delay_only`/`energy_only` comparison branches,
the unused legacy `han_attn_legacy` trainer/baseline, the unused generic
`Runner`, and the unused single-agent `RolloutBuffer`.

**Current active entry points:** `scripts/train.py`,
`scripts/compare_system_baselines.py`, `scripts/plot_training_artifacts.py`,
and `scripts/run_multiuser_scaling_suite.py`.

**Validation:** `C:\Users\19704\.conda\envs\satellite.env\python.exe -m
pytest -q` passed with 86 tests. Full Python compilation via
`python -m py_compile` over all `*.py` files also passed.

**Follow-up:** keep historical commands in this log as provenance, but use the
active entry points above for new runs and plotting.

## 2026-07-05 - Paper-Style Load Balance Metric

**Code change:** added `scripts/load_balance_metrics.py` and wired training,
comparison, and plot-only artifact summaries to use the paper-style load
balance variance metric. When `load_balance_variance` or
`load_variance_samples` is available, `mec_load_fairness`,
`active_load_balance_score`, and `avg_load_balance_score` are compatibility
aliases for the coefficient `(1 - 4B) / (1 + 4B)`. Summaries also preserve
`load_balance_variance`, `load_balance_coefficient`,
`load_variance_sample_count`, and JSON CDF points from time-point variance
samples.

**Validation:** `python -m pytest tests\test_load_balance_metrics.py
tests\test_plot_training_artifacts.py -q` passed, and `python -m py_compile`
passed for `scripts/load_balance_metrics.py`, `scripts/plot_training_artifacts.py`,
`scripts/compare_system_baselines.py`, and `scripts/train.py`.

**Follow-up:** when the environment source is available, emit
`load_variance_samples` from each evaluation episode so Figure-9-style CDF
plots use all per-time-step load variance samples rather than only per-episode
mean values.

## 2026-05-10 01:07:57 - Baseline Compare, Latency Priority

**Experiment directory:** `results/baseline_compare/20260510_010757`

**Command summary:** `scripts/compare_system_baselines.py --run-mode train_compare --objective multi_objective --total-timesteps 1200000 --episodes 10 --max-steps 2000 --seed 42 --num-users 20 --best-model-metric effective_latency_score --compare-ranking-metric effective_latency_score --baselines all`

**Code state:** pre-fix reward and training defaults. Reward did not include an
explicit service-continuity term. MAPPO entropy coefficient used a hard-coded
linear decay to 10% by about 600k steps. Training CLI defaulted to
`n_epochs=4`, `batch_size=256`.

**Main result:**

| Method | Mean Reward | Effective Latency | Avg Delay | Continuity | Task Success | Deadline Violation | Energy / Resolved Task |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MADDPG | 952.221 | 0.297095 | 2.113535 | 0.988585 | 0.935667 | 0.064211 | 2.073609 |
| Min-Distance | 951.742 | 0.296882 | 2.113866 | 0.987993 | 0.935659 | 0.064219 | 2.073682 |
| DQN | 907.117 | 0.279776 | 2.186189 | 0.974788 | 0.914433 | 0.085330 | 1.702983 |
| HAN+MAPPO | 795.113 | 0.273059 | 2.218435 | 0.987705 | 0.889758 | 0.110106 | 1.194593 |
| MAPPO (no HAN) | 792.147 | 0.272036 | 2.233144 | 0.988277 | 0.889959 | 0.109919 | 1.245349 |

**Diagnosis:**

HAN+MAPPO was energy-efficient but lost on task success and deadline
violations. The earlier suspicion that service continuity was around 0.488 was
incorrect; that value was load-balance score. Evaluation continuity was already
high at 0.987705. The reward/metric mismatch still matters because
`effective_latency_score = 1 / (1 + delay) * continuity * task_success`, while
the reward optimized additive delay, energy, QoS, handover, and load-balance
terms.

Training curves showed exploration collapse after about 600k steps and weak
handover success improvement. HAN provided almost no measured gain over MAPPO
without HAN in this run.

**Code changes started from this diagnosis on 2026-05-12:**

- Added step-level `reward_service_continuity` to align reward with the
  continuity factor used by `effective_latency_score`.
- Lowered default `reward_failed_handover_penalty` from 0.6 to 0.3.
- Changed training defaults to `n_epochs=10`, `batch_size=64`,
  `entropy_coef=0.005`, and `entropy_schedule=constant`.
- Added configurable MAPPO entropy schedules: `constant` and `linear`.
- Added reward breakdown fields to baseline comparison summaries and
  per-episode CSV output.

**Follow-up experiment:**

Rerun the same suite with a new output directory and compare reward components
for every method, not only HAN+MAPPO training history. Primary success criteria:
HAN+MAPPO should improve `task_success_rate`, lower `deadline_violation_rate`,
and raise `effective_latency_score` without destroying energy efficiency.

## 2026-05-16 - HAN+MADDPG / HAN+PDQN Implementation Smoke

**Code version:** `b43ecfe` plus local implementation changes for
`src/algorithm/replay_buffer.py`, `src/algorithm/maddpg.py`,
`src/algorithm/pdqn.py`, `scripts/train.py`, and
`scripts/compare_system_baselines.py`.

**Current setting:** post-2026-05-12 environment/reward code is canonical.
Old 2026-05-10 learned-method results are historical reference only and should
not be mixed into strict current-code comparison tables.

**Commands and artifacts:**

- `python scripts/train.py --algorithm maddpg --total_timesteps 2000 --max_steps 100 --eval_episodes 1 --save_path results/smoke_han_maddpg --log_path results/smoke_logs --device cpu`
  - wrote `results/smoke_han_maddpg/training_history.json`
  - wrote `results/smoke_han_maddpg/final_model.pt`
- `python scripts/train.py --algorithm pdqn --total_timesteps 2000 --max_steps 100 --eval_episodes 1 --save_path results/smoke_han_pdqn --log_path results/smoke_logs --device cpu`
  - wrote `results/smoke_han_pdqn/training_history.json`
  - wrote `results/smoke_han_pdqn/final_model.pt`
- `python scripts/compare_system_baselines.py --baselines random maddpg pdqn han_maddpg han_pdqn --total-timesteps 2000 --maddpg-timesteps 2000 --pdqn-timesteps 2000 --episodes 1 --max-steps 100 --device cpu`
  - wrote `results/baseline_compare/20260516_153817/comparison_summary.json`
  - wrote `results/baseline_compare/20260516_153817/comparison_summary.csv`

**Smoke comparison snapshot:** 2000 timesteps, 1 evaluation episode, max 100
steps, CPU, seed 42. This is only a wiring smoke, not a paper-quality ranking.

| Method | Mean Reward | Effective Latency | Avg Delay | Continuity | Task Success | Deadline Violation | Energy / Resolved Task |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MADDPG | 98.752 | 0.319489 | 1.955349 | 0.999400 | 0.944767 | 0.055233 | 1.955349 |
| HAN+PDQN | 89.720 | 0.283957 | 2.161456 | 0.999400 | 0.898256 | 0.101744 | 0.783904 |
| PDQN | 88.287 | 0.275211 | 2.235813 | 0.991400 | 0.898256 | 0.101744 | 1.130091 |
| HAN+MADDPG | 89.545 | 0.274444 | 2.260448 | 0.999400 | 0.895349 | 0.104651 | 0.058514 |
| Random | 58.879 | 0.095314 | 2.417965 | 0.396000 | 0.822674 | 0.171512 | 1.912117 |
| HAN+MAPPO smoke system | 37.526 | 0.035208 | 2.632047 | 0.166000 | 0.770349 | 0.223837 | 1.004960 |

**Verification:** `pytest tests/ -v` passed with 77 passed and 4 skipped.

**Follow-up decision:** keep the implementation and rerun strict comparisons
under current code for `han_mappo`, `mappo_no_han`, `maddpg`, `han_maddpg`,
`pdqn`, and `han_pdqn`. Heuristic methods can be re-evaluated cheaply.

## 2026-05-17 - All Methods 1200k Baseline Compare, Artifact Summary

**Experiment directory:** `results/baseline_compare/20260516_221726_all_methods_1200k`

**System run directory:** `results/full_train_latency_priority_20260516_221726`

**Summary generation:** `scripts/generate_comparison_from_artifacts.py` rebuilt
`comparison_summary.json`, `comparison_summary.csv`, `episode_metrics.csv`, and
the PDF figures from existing trained artifacts. The summary metadata records
`run_mode=artifact_plot_only`, `generated_at=from_existing_artifacts`,
`objective=multi_objective`, `best_model_metric=effective_latency_score`,
`compare_ranking_metric=effective_latency_score`, `episodes=5`,
`max_steps=2000`, `seed=42`, `num_users=20`.

**Code state:** post-HAN+MADDPG/HAN+PDQN implementation and post parameter
speed-up defaults. MAPPO-family training used `n_epochs=6`, `batch_size=256`,
constant `entropy_coef=0.005`, service-continuity reward weight `0.5`, failed
handover penalty `0.3`, and `best_model_metric=effective_latency_score`.
The 2026-05-16 `参数更新_加速` change reduced PPO update work from 10 epochs
with 64 batch size to 6 epochs with 256 batch size.

**Main result:**

| Method | Mean Reward | Effective Latency | Avg Delay | Continuity | Task Success | Deadline Violation | Energy / Resolved Task |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Min-Distance | 1942.579 | 0.297771 | 2.107855 | 0.988042 | 0.936735 | 0.063255 | 2.068334 |
| HAN+PDQN | 1940.673 | 0.296991 | 2.109846 | 0.987164 | 0.935717 | 0.064273 | 2.063540 |
| MADDPG | 1932.651 | 0.294253 | 2.116899 | 0.982811 | 0.933305 | 0.066684 | 2.061306 |
| MAPPO (no HAN) | 1787.248 | 0.273940 | 2.206264 | 0.986906 | 0.890150 | 0.109828 | 1.092343 |
| HAN+MAPPO | 1785.752 | 0.273814 | 2.214800 | 0.988140 | 0.890972 | 0.109010 | 1.190327 |
| HAN+MADDPG | 1774.701 | 0.244121 | 2.178774 | 0.841562 | 0.922027 | 0.077953 | 2.065456 |
| PDQN | 1762.304 | 0.241604 | 2.268838 | 0.895503 | 0.881884 | 0.118080 | 1.929118 |
| Full-Local | 1765.470 | 0.240937 | 2.294284 | 0.915235 | 0.867412 | 0.132556 | 1.917774 |
| Joint Greedy | 1716.369 | 0.226832 | 2.318859 | 0.874955 | 0.860306 | 0.139658 | 1.878545 |
| Random | 1231.728 | 0.104135 | 2.562058 | 0.451001 | 0.823164 | 0.176672 | 2.012683 |

**Primary metric leaders:** Min-Distance won average delay, service
availability, task success, and deadline violation. HAN+MAPPO only led service
continuity, by a very small margin over Min-Distance.

**Comparison with 2026-05-10 current-overlap methods:**

| Method | Effective Latency Change | Delay Change | Task Success Change | Deadline Change | Energy / Resolved Task Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| HAN+MAPPO | +0.28% | 2.218435 -> 2.214800 | 0.889879 -> 0.890972 | 0.110106 -> 0.109010 | 1.194593 -> 1.190327 |
| MAPPO (no HAN) | +0.70% | 2.233144 -> 2.206264 | 0.890067 -> 0.890150 | 0.109919 -> 0.109828 | 1.245349 -> 1.092343 |
| MADDPG | -0.96% | 2.113535 -> 2.116899 | 0.935781 -> 0.933305 | 0.064211 -> 0.066684 | 2.073609 -> 2.061306 |
| Min-Distance | +0.30% | 2.113866 -> 2.107855 | 0.935773 -> 0.936735 | 0.064219 -> 0.063255 | 2.073682 -> 2.068334 |

**Diagnosis:**

The service-continuity reward and effective-latency checkpoint selection did
not materially close the gap for HAN+MAPPO. HAN+MAPPO is still
energy-efficient, but it remains task-success and deadline limited, and it is
slightly behind MAPPO without HAN on the primary `effective_latency_score`.
This means the HAN encoder has not yet produced a measurable advantage for the
MAPPO path under this 20-user/1200k setup.

The newly added off-policy learned baselines changed the ranking picture.
HAN+PDQN is nearly tied with Min-Distance and beats MADDPG on the primary score,
while standalone PDQN is weak. This suggests the HAN representation is useful
for PDQN in this setting, but the same conclusion does not hold for MAPPO or
MADDPG. HAN+MADDPG degraded badly on service continuity, with large failed
handover and handover-cost penalties, so it should not be treated as a strong
baseline yet.

The speed-up change likely reduced MAPPO wall-clock cost without obvious
metric damage compared with 2026-05-10, but it also did not unlock a meaningful
quality gain. The best evidence is that MAPPO no-HAN improved effective latency
by 0.70%, HAN+MAPPO improved by only 0.28%, and both remain far below
Min-Distance/HAN+PDQN/MADDPG.

**Data-quality notes:**

The summary was generated after the fact from artifacts, not directly by the
original train-compare run. `HAN+PDQN` has `summary.total_steps=1200000`, but
its config reports `total_timesteps=400000` and its visible training records
start at 802000 steps, which looks like a resumed or partially merged history.
The final comparison metrics are checkpoint evaluations, but the training
curve for HAN+PDQN should be interpreted cautiously.

**Follow-up decision:** keep HAN+PDQN and the artifact-summary helper as useful
evaluation additions. Do not claim HAN+MAPPO superiority from this run. Next
work should focus on why MAPPO learns an energy-saving policy with low task
success, and why HAN+MADDPG loses continuity. For paper-style ranking, rerun
the top contenders with multiple seeds and ensure the summary is produced by a
single uninterrupted comparison workflow.

## 2026-05-17 - Reward Curve / Off-Policy Evaluation Bugfix

**Context:** Follow-up debugging of
`results/baseline_compare/20260516_221726_all_methods_1200k` after the reward
curves appeared non-convergent for most methods.

**Root causes found:**

- `scripts/compare_system_baselines.py` plotted `training[*].mean_reward`
  whenever training records existed, even if comparable
  `evaluation[*].eval_mean_reward` records also existed. This mixed different
  reward semantics across algorithms: rollout mean reward for MAPPO,
  per-episode training reward for off-policy baselines, and deterministic
  checkpoint evaluation reward in the final table.
- `scripts/train.py` used the same `self.env` for `HAN+MADDPG` and `HAN+PDQN`
  training and evaluation. `_evaluate()` reset and stepped that environment
  during training, then the training loop continued with cached observations
  from the pre-evaluation environment state. This could poison replay data and
  explains records such as near-zero `mean_reward` with strong QoS metrics.
- The apparent HAN+MAPPO reward jump from 795.113 in the 2026-05-10 run to
  1785.752 in this run is mostly reward-definition drift, not algorithmic
  improvement. The new run includes `reward_service_continuity=988.140`; adding
  that term to the old reward scale accounts for almost all of the difference.

**Code changes:**

- `load_training_curve_from_path()` now prefers evaluation reward curves when
  valid `eval_mean_reward` records exist, and falls back to training rewards
  only when no evaluation curve is available.
- `HANMADDPGTrainer._evaluate()` now evaluates in an isolated environment,
  restores the training environment afterward, clears cached HAN embeddings,
  and closes the temporary environment.

**Verification:**

- Added `tests/test_baseline_plotting.py` for evaluation-reward curve priority.
- Added `tests/test_offpolicy_evaluation.py` for off-policy eval isolation.
- Ran `pytest tests -q`: 73 passed, 4 skipped.

**Follow-up decision:** regenerate comparison plots after this fix before using
reward curves in analysis. Existing final table metrics from checkpoint
evaluation remain useful, but the old reward curve PDF should be treated as
misleading for convergence claims.

## 2026-05-17 - Reward Function Rebalance

**Context:** Follow-up to the abnormal reward scale diagnosis. The previous
service-continuity reward granted a positive per-step uptime bonus, so a
2000-step run could accumulate roughly `+1000` reward from normal continuity
alone. That made reward magnitude weakly comparable across runs and allowed
one component to dominate the learning signal.

**Design change:** default reward weights were rebalanced around normalized
components:

| Component | Default weight |
| --- | ---: |
| Delay | 0.25 |
| Energy | 0.15 |
| Handover | 0.10 |
| Load balance | 0.05 |
| QoS / task success | 0.30 |
| Service interruption | 0.15 |
| Deadline violation | 0.30 |

`reward_service_continuity` remains the backward-compatible breakdown key, but
its semantics changed from a positive continuity bonus to a signed interruption
penalty:

`-reward_service_continuity_weight * interruption_seconds / step_user_seconds`.

No-interruption steps now contribute `0`, so high service continuity is still
measured by `service_continuity_rate` but no longer inflates episode reward by
hundreds of points.

**Code changes:**

- Updated `EnvConfig`, `TrainConfig`, training CLI defaults, server-training
  defaults, and baseline-comparison generated configs to the balanced weights.
- Passed `reward_deadline_penalty` through all train/compare environment
  builders so deadline weighting is not silently stuck at the environment
  default.
- Renamed the plotted reward component label from "Service Continuity Reward"
  to "Service Interruption Penalty".
- Added regression tests for the interruption-only continuity term, long-run
  continuity bound, balanced defaults, and server-training default parity.

**Expected metric effect:** new training runs will have much lower absolute
mean reward than the 2026-05-16 run, especially because the old
`reward_service_continuity=988.140` style contribution disappears. This is an
intentional scale correction, not a performance regression by itself. Future
comparisons should prioritize `effective_latency_score`, `avg_delay`,
`task_success_rate`, `deadline_violation_rate`, `service_continuity_rate`, and
the now-bounded reward breakdown rather than raw reward across old and new
reward definitions.

## 2026-05-21 - 1200k Latency-Priority Result Review

**Run artifacts reviewed:**

- Training: `results/full_train_latency_priority_20260517_225631`
- Baseline comparison:
  `results/baseline_compare/20260517_225631_all_methods_1200k`

**Configuration / selection metric:**

- Objective: `multi_objective`
- Total timesteps: `1,200,000`
- Best-model and comparison ranking metric: `effective_latency_score`
- Primary comparison metrics: average delay, service continuity, service
  availability, task success, deadline violation.

**Key observed metrics:**

- HAN+MAPPO training history reached its best eval score at about 301k steps:
  `effective_latency_score=0.2729`, `avg_delay=2.2260`,
  `service_continuity_rate=0.9880`, `task_success_rate=0.8911`.
- The final HAN+MAPPO eval at 1.2M steps was lower but still usable:
  `effective_latency_score=0.2228`, `avg_delay=2.3858`,
  `service_continuity_rate=0.9148`, `task_success_rate=0.8247`.
- In the generated comparison summary, checkpoint-evaluated HAN+MAPPO fell to
  `effective_latency_score=0.1261`, `avg_delay=2.4903`,
  `service_continuity_rate=0.5415`, `task_success_rate=0.8128`.
- Comparison leaders were Min-Distance on latency score
  (`0.2978`), delay (`2.1079`), task success (`0.9366`), and deadline violation
  (`0.0633`); MAPPO without HAN led service continuity (`0.9883`) and service
  availability (`0.9909`).

**Diagnosis:**

- The training itself did learn useful policies early, but HAN+MAPPO was not
  stable across the full 1.2M schedule. Evaluation quality oscillated sharply
  after 300k steps, with weak eval windows near 401k, 700k, 901k, and 1101k.
- The comparison artifact is suspicious for final ranking because the system
  checkpoint evaluation is far worse than the best and final evaluations stored
  in the training history, while episode-to-episode variance inside the
  comparison is very small. This points to a checkpoint/evaluation-path issue
  or a policy loading/eval-mode mismatch that should be investigated before
  using the table as a publication result.
- `Full-Local`, `MADDPG`, and `HAN+MADDPG` have identical comparison rows,
  which is another warning that at least part of the comparison table may be
  reusing fallback behavior or not evaluating distinct trained policies.

**Follow-up decision:**

- Do not present this comparison as a clean win/loss result yet. First verify
  the HAN+MAPPO checkpoint chosen by `best_model.pt`, evaluate `best_model.pt`
  and `final_model.pt` through the same comparison path, and inspect why
  MADDPG/HAN+MADDPG collapse to the same metrics as Full-Local.

**Plotting correction made during review:**

- `paper_baseline_dashboard.pdf` and `reward_curve_vs_baselines.pdf` were
  regenerated from existing artifacts after fixing the reward-curve loader.
  When dense `training` records exist, the plot now uses raw training rewards
  as the translucent shadow and the window-5 moving average as the solid line;
  sparse `evaluation` records are retained only as checkpoint markers.
- The previous dashboard used sparse evaluation rewards as the curve for
  methods that had evaluation records, which exaggerated isolated evaluation
  collapses into large triangular swings.

## 2026-05-21 - Off-Policy Baseline Repair Pass

**Problem diagnosed:** PDQN/HAN+PDQN were trained from a scalar mean reward even
though the environment has per-user actions. That turns heterogeneous user
outcomes into the same target for every agent and weakens credit assignment.
The PDQN exploration schedule also decayed across the full run, so a 1.2M-step
run was still heavily exploratory halfway through training. MADDPG/HAN+MADDPG
comparison rows collapsed to Full-Local behavior, so the comparison table needs
action-level diagnostics rather than relying on reward alone.

**Code changes made:**

- `LEOSatelliteEnv` now exposes `last_user_rewards` and `info["user_rewards"]`.
  The scalar environment reward remains the mean of this vector, preserving the
  existing Gym API while allowing off-policy algorithms to train from user-level
  targets.
- `MultiAgentReplayBuffer` now stores either scalar rewards or per-agent reward
  vectors. PDQN consumes per-agent rewards directly; MADDPG defensively averages
  vectors back to a centralized joint reward.
- PDQN target updates now use Double-DQN style action selection and evaluate
  the selected next action with the target networks. The parameter-network loss
  now optimizes the best valid discrete action instead of averaging all valid
  action Q-values.
- HAN+PDQN and raw PDQN replay insertion now use per-user rewards from the
  environment. PDQN epsilon decay defaults to the first 40% of training, with a
  lower bound past warmup, instead of stretching forced exploration across the
  entire run.
- Baseline/system evaluation summaries now include
  `handover_action_rate`, `local_compute_rate`, and `mean_offload_ratio` so
  Full-Local collapse and zero-offload policies are visible in the CSV.

**Verification:**

- `conda run -n satellite.env python -m pytest tests\test_baseline_plotting.py tests\test_offpolicy_evaluation.py tests\test_env_metrics.py::test_step_exposes_per_agent_rewards_matching_scalar_mean -q`
  passed: `9 passed`.

## 2026-05-22 - PDQN Fast Early-Convergence Patch

**Intent:** Apply a small, low-runtime-cost acceleration pass for PDQN and
HAN+PDQN before rerunning long comparisons. No long experiment was launched in
this code pass.

**Code changes made:**

- PDQN exploration now decays faster by default:
  `epsilon_decay_fraction=0.25` and `epsilon_final=0.02`.
- PDQN warmup and epsilon exploration use a `70%` safe heuristic plus `30%`
  random mix instead of pure random sampling. The safe heuristic keeps a stable
  serving satellite when RVT is acceptable, otherwise switches to a high
  elevation visible candidate, with a moderate offload ratio when a task exists.
- HAN+PDQN observations now concatenate `raw_obs + HAN_embed + rvt/task`
  features, so the policy is not forced to depend only on an untrained cached
  HAN embedding.
- PDQN parameter-network behavior-cloning loss was reduced to
  `bc_loss_coef=0.001`.
- PDQN Q and parameter networks now apply `LayerNorm(obs_dim)` to observation
  features before the MLPs. PDQN checkpoint loading tolerates older checkpoints
  that do not contain the new LayerNorm parameters.

**Expected metric effect:**

- Earlier training should spend less time on invalid/noisy handover-offload
  combinations, and HAN+PDQN should recover useful raw environment signals
  immediately. This should improve early reward and latency/task metrics
  without materially increasing per-step runtime.

**Verification:**

- `C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests\test_offpolicy_evaluation.py tests\test_han_integration.py tests\test_baseline_plotting.py tests\test_env_metrics.py::test_step_exposes_per_agent_rewards_matching_scalar_mean tests\test_mappo_entropy.py -q`
  passed: `24 passed`.

## 2026-05-28 - 20260525 Artifact-Only Full Baseline Plot Regeneration

**Experiment directory:** `results/baseline_compare/20260525_220324`

**System run directory:** `results/full_train_latency_priority_20260525_220324`

**Command/config:** Regenerated summaries and figures from existing artifacts,
without retraining:

`C:\Users\19704\.conda\envs\satellite.env\python.exe scripts\generate_comparison_from_artifacts.py --system-run-dir results\full_train_latency_priority_20260525_220324 --compare-dir results\baseline_compare\20260525_220324 --objective multi_objective --episodes 5 --max-steps 600 --seed 42 --num-users 20 --device cpu --metric effective_latency_score --plot-window 5`

The previous 4-method summary/plots were backed up under
`results/baseline_compare/20260525_220324/backup_before_artifact_full_20260528`.

**Code changes made:** `scripts/generate_comparison_from_artifacts.py` now
sanitizes artifact `save_path`/`log_path` values before evaluation, so Linux
paths embedded in old training histories do not make Windows try to write under
`/home`. The helper also loads MADDPG checkpoints into the current
`MADDPGAlgorithm` API instead of the old actor-only evaluation call.

**Main result:** The regenerated `comparison_summary.json` contains 10
methods: HAN+MAPPO, Random, Min-Distance, Full-Local, Joint Greedy, MADDPG,
PDQN, MAPPO(no-HAN), HAN+MADDPG, and HAN+PDQN. Current effective-latency
ranking is led by Min-Distance (`0.293305`), MADDPG (`0.292881`), PDQN
(`0.273594`), HAN+MAPPO (`0.271304`), and MAPPO(no-HAN) (`0.269998`).

**Diagnosis:** The earlier `paper_baseline_dashboard.pdf` was incomplete
because its input summary only contained four methods. The artifact directory
still had `maddpg` and `pdqn` checkpoints, but the overwritten
`comparison_summary.json` did not include those rows, so plotting could not
draw their lines. Rule-based methods are not stored under `learned_baselines`;
they are re-evaluated on demand and now appear in the regenerated summary and
figures.

**Verification:**

- `C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests\test_generate_comparison_from_artifacts.py -q`
  passed: `2 passed`.
- Regeneration completed and wrote `comparison_summary.json`,
  `comparison_summary.csv`, `episode_metrics.csv`, `paper_baseline_dashboard.pdf`,
  `reward_curve_vs_baselines.pdf`, and the other comparison PDFs.

## 2026-05-29 - Graph-Update-1 300k/600s/u10 Latency-Priority Run Analysis

**Experiment directories:**

- System training: `results/full_train_latency_priority_g1_300k_600s_u10_20260528_235135`
- Baseline comparison: `results/baseline_compare/g1_300k_600s_u10_20260528_235135`

**Command/config summary:** Diagnostic HAN+MAPPO latency-priority suite with
`graph_update_interval=1`, `total_timesteps=300000`, `max_steps=600`,
`num_users=10`, `eval_episodes=3`, `compare_episodes=3`, seed `42`, and
`best_model_metric=effective_latency_score`. Comparison ran in `compare_only`
mode against MAPPO(no-HAN), Random, Min-Distance, Full-Local, and Joint Greedy.

**Training result:** HAN+MAPPO completed `300032` steps / `293` episodes.
Best checkpoint by `effective_latency_score` occurred at `50176` steps with
`0.267513`; the final evaluation at `300032` steps had higher reward
(`95.4994`) but slightly lower effective-latency score (`0.266341`). Training
time was `12921.98` seconds.

**Comparison result:** On the 3-episode comparison, Min-Distance ranked first
by `effective_latency_score` (`0.285889`) and HAN+MAPPO ranked second
(`0.262544`), just ahead of MAPPO(no-HAN) (`0.261964`). HAN+MAPPO led service
continuity (`0.977033`), service availability (`0.979667`), total energy
(`1775.56`), and energy per resolved task (`0.834772`). Min-Distance led
average delay (`2.165214`), task completion (`0.926360`), and deadline
violation (`0.073630`).

**Diagnosis:** Updating the heterogeneous graph every step preserves fresh
graph context and yields excellent energy efficiency and service continuity,
but this 300k diagnostic run did not convert that extra graph freshness into a
clear latency/task-success lead. Relative to MAPPO(no-HAN), HAN+MAPPO improved
selection score by only `0.22%`, average delay by `0.31%`, and energy by
`13.55%`; task completion was effectively tied. Relative to Min-Distance,
HAN+MAPPO used about `59.6%` less energy but had worse delay, task completion,
deadline violation, reward, and primary ranking score.

**Follow-up decision:** Treat graph-update-1 as promising for energy/service
stability, but not yet as a latency-priority winner. Next ablations should keep
the same 10-user/600-step/300k setup and compare graph update intervals
directly, or adjust the reward/selection objective to put stronger pressure on
task success and deadline violations if the target is to beat Min-Distance on
effective latency.

## 2026-05-29 - HAN+MAPPO Raw-Plus-HAN Observation Path

**Intent:** Prepare a diagnostic rerun that tests whether HAN provides
incremental graph information to MAPPO instead of replacing the raw environment
state with a frozen cached embedding.

**Code changes made:**

- `HANMAPPOTrainer` now builds policy observations as
  `raw_observation + HAN_user_embedding + rvt_warning/task_features`.
- `HANPDQNTrainer` reuses the shared raw-plus-HAN observation layout so it does
  not prepend raw observations twice.
- Regression tests cover the HAN+MAPPO and HAN+PDQN observation dimensions and
  raw-observation prefix.

**Expected metric effect:** This should reduce the risk that HAN+MAPPO
underperforms MAPPO(no-HAN) because direct raw state features were removed. The
first success criterion is a measurable improvement over MAPPO(no-HAN) on
`effective_latency_score` without losing the existing energy advantage. This
does not yet train the HAN encoder end-to-end and should be treated as an
information-path ablation, not a final architecture claim.

## 2026-05-29 - Raw-Plus-HAN CPU 50k Eval Smoke

**Experiment directory:** `results/full_train_latency_priority_g1_300k_600s_u10_20260529_rawhan_cpu`

**Configuration:** CPU diagnostic launch with `graph_update_interval=1`,
`total_timesteps=300000`, `max_steps=600`, `num_users=10`,
`eval_interval=50000`, `eval_episodes=3`, and
`best_model_metric=effective_latency_score`. The run used the new raw-plus-HAN
observation path and did not load an old checkpoint.

**Observed result:** Training reached the first evaluation at `50176` steps
without observation-shape or CPU execution errors. The first eval reported
`effective_latency_score=0.2675`, `avg_delay=2.229s`, `task_completion=88.00%`,
`service_continuity=98.17%`, `avg_load_balance=0.658`, mean reward `93.60`,
and saved `best_model.pt`.

**Stop condition:** The run was intentionally stopped after the first eval as a
bug-smoke check, before final checkpoint/history export and before baseline
comparison. Therefore `training_history.json` and `comparison_summary.json` were
not produced for this partial run.

**Follow-up decision:** The raw-plus-HAN policy-input path passes the CPU
training/evaluation smoke gate. A full diagnostic comparison can reuse the same
configuration with a fresh run id when a complete 300k run is needed.

## 2026-06-01 - Raw-Plus-HAN g1 300k/600s/u10 Full Diagnostic Result

**Experiment directories:**

- System training: `results/full_train_latency_priority_g1_300k_600s_u10_20260529_130351`
- Baseline comparison: `results/baseline_compare/g1_300k_600s_u10_20260529_130351`

**Configuration:** Full raw-plus-HAN diagnostic run with
`graph_update_interval=1`, `total_timesteps=300000`, `max_steps=600`,
`num_users=10`, `eval_episodes=3`, `compare_episodes=3`,
`best_model_metric=effective_latency_score`, and comparison ranking by
`effective_latency_score`. Reward weights remained the balanced defaults:
delay `0.25`, energy `0.15`, QoS `0.30`, service interruption `0.15`, and
deadline penalty `0.30`.

**Training result:** HAN+MAPPO completed `300032` steps / `293` episodes.
Best eval checkpoint by `effective_latency_score` occurred at `50176` steps
with score `0.269787`, average delay `2.2080`, task completion `0.8831`,
deadline violation `0.1169`, service continuity `0.9800`, and energy per
resolved task `0.8494`. The final eval at `300032` steps dropped to
`effective_latency_score=0.259076`, `avg_delay=2.2663`, and
`task_completion=0.8699`. Training time was `6400.11` seconds.

**Comparison result:** On the 3-episode comparison, Min-Distance still ranked
first by `effective_latency_score` (`0.285889`). Raw-plus-HAN+MAPPO ranked
second (`0.262687`), ahead of MAPPO(no-HAN) (`0.261964`) but only by `0.28%`.
Compared with the previous g1 HAN+MAPPO run (`20260528_235135`), the system
score improved by only `0.05%`; average delay improved by `0.09%`, service
continuity by `0.01%`, and energy per resolved task regressed by `3.18%`.
Task completion was effectively unchanged and slightly lower
(`0.869742` vs `0.869900`).

**Diagnosis:** Raw-plus-HAN fixed the information-path concern but did not
unlock a meaningful HAN advantage for MAPPO. The comparison cleared neither
the planned `1%` improvement gate over MAPPO(no-HAN) nor the task-success gap
to Min-Distance. HAN+MAPPO still leads service continuity and availability and
is much more energy efficient than Min-Distance, but the decisive weakness is
unchanged: task completion and deadline violation remain near MAPPO(no-HAN)
and far behind Min-Distance.

**Follow-up decision:** Do not spend more runs only on raw-plus-HAN or
graph-update interval. The next useful ablation should change the learning
problem: either make HAN/HGT trainable or auxiliary-supervised, or introduce a
hierarchical heuristic-prior/residual-RL policy that starts from a
deadline-aware Min-Distance/greedy action and lets MAPPO learn corrections.

## 2026-06-01 - Attn+MAPPO Candidate-Attention Baseline Added

**Code change:** Added an independent `attn_mappo` algorithm path for direct
comparison with HAN+MAPPO. The new actor uses global satellite load
self-attention plus user-to-candidate cross-attention over the raw visible
satellite block. It does not call the heterogeneous graph builder or HAN
encoder, so it is a clean alternative to both HAN+MAPPO and MAPPO(no-HAN).

**Entry points:** `scripts/train.py --algorithm attn_mappo` trains the new
method. `scripts/compare_system_baselines.py --baselines attn_mappo ...`
trains/evaluates it as a learned baseline and labels it as `Attn+MAPPO`.
The g1 diagnostic suite now includes `attn_mappo` in its default comparison
set.

**Smoke experiment directory:** `results/attn_mappo_smoke`

**Smoke configuration:** CPU-only functional smoke with `num_users=2`,
`total_timesteps=16`, `max_steps=5`, `n_steps=8`, `batch_size=8`,
`n_epochs=1`, `eval_interval=8`, and
`best_model_metric=effective_latency_score`.

**Observed result:** The rollout, PPO update, evaluation, checkpoint save, and
`training_history.json` export all completed. The first smoke evaluation saved
`best_model.pt` with `effective_latency_score=0.3276`, `avg_delay=2.052s`,
task completion `100.00%`, service continuity `100.00%`, and load balance
`1.000`. These values are only a wiring smoke, not a performance claim.

**Expected metric effect:** This baseline should test whether explicit load
attention over visible candidate satellites can improve deadline-sensitive
handover/offloading without relying on the current non-end-to-end HAN path.
The success gate for a real 300k g1 run is outperforming both HAN+MAPPO and
MAPPO(no-HAN) on `effective_latency_score`, while keeping service continuity
and energy per resolved task near the current HAN+MAPPO level.

## 2026-06-03 - Latency-Priority Weight Update g1 300k/600s/u10 Result

**Experiment directories:**

- System training: `results/full_train_latency_priority_g1_300k_600s_u10_20260602_143937`
- Baseline comparison: `results/baseline_compare/g1_300k_600s_u10_20260602_143937`

**Configuration:** `graph_update_interval=1`, `total_timesteps=300000`,
`max_steps=600`, `num_users=10`, `n_steps=1024`, `eval_episodes=3`,
`compare_episodes=3`, and `best_model_metric=effective_latency_score`.
The reward update emphasized latency/deadline success: delay weight `0.35`,
energy weight `0.05`, QoS weight `0.40`, service continuity weight `0.15`,
deadline slack weight `0.25`, failed-task penalty `0.80`, and deadline
penalty `1.00`.

**Training result:** HAN+MAPPO completed `300032` steps / `293` episodes.
The best eval checkpoint occurred early at `50176` steps with
`effective_latency_score=0.267742`, `avg_delay=2.2161`,
`task_completion_rate=0.8798`, `service_continuity_rate=0.9787`,
`avg_load_balance_score=0.6890`, and energy per resolved task `0.7986`.
The final eval at `300032` steps was lower:
`effective_latency_score=0.261039`, `avg_delay=2.2733`,
`task_completion_rate=0.8744`, and deadline violation rate `0.1255`.

**Comparison result:** Min-Distance remained first with
`effective_latency_score=0.285889`, `avg_delay=2.1652`,
`task_completion_rate=0.9264`, and deadline violation rate `0.0736`, but with
high energy per resolved task `2.0674`. MAPPO(no-HAN) scored `0.262389`,
HAN+MAPPO scored `0.260886`, and Attn+MAPPO scored `0.257637`.

**Diagnosis:** The reward-weight update did not improve HAN+MAPPO relative to
the previous raw-plus-HAN run (`0.262687` -> `0.260886`). It mostly preserved
the existing ranking: Min-Distance wins the latency/deadline-oriented metric,
while learned methods remain much more energy efficient. Min-Distance is strong
because it uses full-local execution selected by offload-grid search
(`local_compute_rate=1.0`, `mean_offload_ratio=0.0`) and the current
`effective_latency_score` ignores energy, so a robust local-compute heuristic
gets rewarded for deadline success even with poor energy cost. This suggests
the evaluation metric and task/local-compute model are giving rule-based local
execution a structural advantage.

**Follow-up decision:** Do not keep tuning only reward weights. The next check
should validate whether Min-Distance is unrealistically advantaged by the
baseline offload search and metric definition. Useful ablations are:
compare under an energy-aware score, restrict Min-Distance to the same
offload/offloading action family as learned policies, and inspect whether local
CPU/task-size settings make full-local execution too reliable for this scenario.

## 2026-06-02 - Deadline-Priority Reward Redesign

**Context:** The 2026-06-01 Attn+MAPPO g1 comparison showed that candidate
attention learned the lowest energy per resolved task (`0.7492`) but ranked
behind HAN+MAPPO and MAPPO(no-HAN) on `effective_latency_score`. Diagnosis was
that the current additive reward allowed energy savings and enqueue bonuses to
partly offset deadline misses, while the primary score is directly limited by
delay, service continuity, and task success.

**Code change:** Reworked the task reward in `src/environment/gym_env.py` to
make deadline success the primary objective and energy a secondary objective:

- Default weights changed from delay `0.25`, energy `0.15`, QoS `0.30`,
  deadline penalty `0.30`, enqueue bonus `0.02` to delay `0.35`, energy
  `0.05`, QoS/success `0.40`, deadline-slack reward `0.25`, fixed failed-task
  penalty `0.80`, deadline-excess penalty `1.00`, and enqueue bonus `0.0`.
- Successful tasks now receive `reward_task_success` plus
  `reward_deadline_slack`; delay is a normalized negative cost based on
  `total_delay / max_delay`.
- Energy is no longer a positive reward and is only charged as a small
  successful-task penalty. Failed tasks cannot recover reward by being
  energy-efficient.
- Deadline failures now record both `penalty_task_failure` and
  `penalty_deadline`.
- The new reward fields are propagated through `scripts/train.py` and
  `scripts/compare_system_baselines.py` so training history, comparison CSV,
  and reward-component plots expose the new terms.

**Rationale:** This matches the common LEO/satellite MEC formulation where
deadline/QoS is treated as a hard or high-priority constraint and energy is
optimized after QoS is satisfied. It should push Attn+MAPPO and MAPPO-family
policies away from "cheap but late" offloading behavior and toward task
success/deadline improvement.

**Verification:**

- `pytest tests/test_reward_function.py -q`
- `pytest tests/test_env_metrics.py -q`
- `pytest tests/test_baseline_plotting.py tests/test_generate_comparison_from_artifacts.py -q`
- `pytest tests/test_candidate_attention.py tests/test_han_integration.py::test_attention_mappo_trainer_builds_attention_state_without_han -q`

**Follow-up decision:** Rerun the g1 300k/600s/u10 suite with the new reward
defaults, preferably with `compare_episodes >= 10` and multiple seeds. Success
criteria are lower `deadline_violation_rate`, higher `task_success_rate`, and
an `effective_latency_score` gain over both HAN+MAPPO and MAPPO(no-HAN),
while monitoring whether Attn+MAPPO keeps a meaningful energy advantage.

## 2026-06-03 - Constrained Local CPU and Energy-Aware Ranking Defaults

**Context:** The 2026-06-02 g1 result showed Min-Distance selecting
full-local execution (`local_compute_rate=1.0`, `mean_offload_ratio=0.0`) and
ranking first under `effective_latency_score` despite much higher energy per
resolved task. A local-compute sanity check indicated that the previous
`user_cpu_freq_ghz=1.0` setting made the current task/deadline distribution
too reliable for local execution.

**Code change:**

- Changed `src/environment/mec.py` default `user_cpu_freq_ghz` from `1.0` to
  `0.5` to model a resource-constrained ground/IoT terminal rather than a
  strong local device.
- Changed `scripts/run_latency_priority_g1_300k_600s_u10_suite.py` default
  `best_model_metric` and `compare_ranking_metric` from
  `effective_latency_score` to `latency_priority_score`.
- Updated scenario documentation so the CPU and reward-weight tables match the
  current code.

**Rationale:** This keeps deadline/QoS as the dominant objective but prevents
energy from being entirely free in the default g1 comparison. The existing
`latency_priority_score` already includes a small energy term and load-balance
term, so this is a conservative metric-alignment change rather than a new
score designed around a single baseline.

**Verification:**

- Added regression tests for the constrained local CPU default and the g1
  suite energy-aware ranking defaults.
- `pytest tests/test_reward_function.py::test_default_local_cpu_matches_constrained_terminal_scenario tests/test_reward_function.py::test_g1_latency_suite_defaults_use_energy_aware_ranking`

**Follow-up decision:** Rerun the g1 300k/600s/u10 suite with a fresh run ID.
Report both `latency_priority_score` and `effective_latency_score`, plus
`energy_per_resolved_task`, so the paper can distinguish true latency/QoS
improvement from energy-aware tradeoff changes.

## 2026-06-03 - g1 300k/600s/u10 Constrained-CPU Comparison Result

**Experiment directory:** `results/baseline_compare/g1_300k_600s_u10_20260603_105244`

**Configuration:** `num_users=10`, `max_steps=600`,
`total_timesteps=300000`, `task_arrival_prob=0.35`,
`best_model_metric=latency_priority_score`, and
`compare_ranking_metric=latency_priority_score`. This run used the constrained
local CPU setting (`user_cpu_freq_ghz=0.5`) and the deadline-priority reward
weights: delay `0.35`, energy `0.05`, QoS `0.40`, deadline slack `0.25`,
failed-task penalty `0.80`, and deadline penalty `1.00`.

**Main comparison:**

| Method | Selection Score | Effective Latency | Avg Delay | Task Success | Deadline Violation | Energy / Resolved | Mean Offload | Handover Action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Min-Distance | 0.610711 | 0.246736 | 2.397747 | 0.858626 | 0.141061 | 0.063004 | 1.000000 | 0.005000 |
| MAPPO(no-HAN) | 0.603661 | 0.251775 | 2.335683 | 0.860490 | 0.139354 | 0.148701 | 0.810078 | 0.001611 |
| Attn+MAPPO | 0.602867 | 0.251945 | 2.338451 | 0.861901 | 0.137943 | 0.169081 | 0.764465 | 0.000444 |
| HAN+MAPPO | 0.602833 | 0.249144 | 2.365555 | 0.859397 | 0.140448 | 0.127174 | 0.856019 | 0.001333 |
| Full-Local | 0.481106 | 0.087418 | 4.251546 | 0.523749 | 0.474841 | 0.463401 | 0.000000 | 0.000000 |

**Diagnosis:** Reducing local CPU to `0.5GHz` successfully removed the previous
full-local advantage: Full-Local dropped to `0.481106` selection score and
about `52.4%` task success. However, the three MAPPO variants still occupy the
same performance plateau. They all learn a near-static, offload-heavy policy
with almost no handover action. Their task success, service continuity, and
deadline violation rates differ only at the third decimal place, and the
three-episode comparison variance is similar to the observed method gaps.

The likely reason is that raw observations already expose strong local
decision features: current serving satellite information, up to ten visible
satellites with distance/elevation/SNR/RVT/load, and the current task features.
In a 10-user scenario with low queue-full and blocked penalties, global
coordination pressure is weak, so HAN and candidate-load attention add little
action-relevant information beyond the raw per-user observation. The learned
methods mostly differ in offload-ratio calibration: HAN+MAPPO offloads more
(`0.856`) and uses less energy than MAPPO(no-HAN)/Attn+MAPPO, while Attn+MAPPO
gets slightly better task success but higher energy. These tradeoffs nearly
cancel under `latency_priority_score`.

**Follow-up decision:** Treat this run as evidence that the current 10-user
g1 setting is not discriminative enough for graph/attention structure. The
next ablation should increase coordination pressure rather than only tune the
policy network: run more comparison episodes or seeds, test more users or
higher task-arrival probability, and report confidence intervals. If the paper
needs to demonstrate HAN's value, include a higher-contention scenario where
load distribution and global satellite state are actually decision-critical.

## 2026-06-04 - g1 300k/600s/u20 Load-Balance Metric Diagnosis

**Experiment directory:** `results/baseline_compare/g1_300k_600s_u20_20260603_201842`

**Observation:** Attn+MAPPO led the main latency/QoS metrics but ranked below
HAN+MAPPO, Min-Distance, and MAPPO(no-HAN) under `latency_priority_score`.
Attn+MAPPO achieved the best average delay (`2.5816s`), best task success
(`0.8377`), lowest deadline violation (`0.1622`), and highest
`effective_latency_score` (`0.2304`), but its `avg_load_balance_score` was only
`0.3521`. By contrast, Min-Distance scored `0.5818` on load balance, and
Full-Local scored `0.6038` despite doing no offloading.

**Root cause:** The current load-balance metric is not a pure MEC compute-load
metric. `LEOSatelliteEnv._compute_load_balance_score()` computes the spread of
`server.queue_length + len(server.connected_users)` over active satellites and
returns `1 / (1 + std(active_loads))`. This means methods can receive high
load-balance scores by spreading serving-satellite connections, even when tasks
are computed locally or when MEC queues are not the bottleneck. Conversely, a
policy that concentrates users on a few good satellites for latency/deadline
benefit can be penalized as "imbalanced" even if its task performance is better.

**Impact on ranking:** The `latency_priority_score` includes
`0.15 * avg_load_balance_score`. Attn+MAPPO gained on delay and task success
but lost roughly `0.033` selection-score points against HAN+MAPPO from the load
balance component alone (`0.0528` vs `0.0861`). This is large enough to reverse
the ranking even though Attn+MAPPO wins the primary latency/deadline metrics.

**Follow-up decision:** Treat the u20 result as evidence that Attn+MAPPO has a
latency/QoS advantage, but do not use the current `avg_load_balance_score` as a
paper claim about MEC load balancing. The next code change should separate
connection balance from MEC compute/queue balance, for example by reporting
both `connection_balance_score` and a queue/utilization-based
`mec_load_balance_score`, then using the MEC-specific score in energy/latency
comparisons.

## 2026-06-04 - MEC-Only Load-Balance Metric Redefinition

**Context:** Follow-up discussion decided not to keep a separate
`connection_balance_score`, because serving-satellite connection spread is not
part of the paper's MEC task-offloading objective and can mislead comparisons.

**Code change:** Redefined `LEOSatelliteEnv._compute_load_balance_score()` so
`avg_load_balance_score` is based only on MEC queue pressure and CPU
utilization. Each satellite's load is computed as `0.5 * queue_ratio + 0.5 *
cpu_utilization`; the final score uses Jain-style fairness across satellites
multiplied by a mean-load activity factor. Empty/full-local MEC usage now
returns `0.0` instead of receiving a high score from balanced connections.

**Verification:**

- Added regression tests showing connection-only distributions do not increase
  `avg_load_balance_score`.
- Added regression tests showing balanced MEC queue/CPU workload scores higher
  than concentrating the same work on one satellite.
- `pytest tests/test_env_metrics.py -q`

**Follow-up decision:** Rerun u20/u30 comparisons after this change. Historical
`avg_load_balance_score` values before this entry are not comparable to future
runs because the metric definition changed.

## 2026-06-04 - g1 300k/600s/u20 MEC-Only LB Re-Evaluation

**Experiment directory overwritten:**
`results/baseline_compare/g1_300k_600s_u20_20260603_201842`

**Command intent:** Re-ran the evaluation stage on CPU after redefining
`avg_load_balance_score`. The system checkpoint, MAPPO(no-HAN) checkpoint, and
Attn+MAPPO checkpoint were re-evaluated for `3` episodes with `max_steps=600`;
the learned baselines reused existing `best_model.pt` files instead of
retraining. Rule-based baselines were re-evaluated under the same seed and
episode settings. Output JSON/CSV/PDF files were regenerated in the same
baseline comparison directory.

**Code support added for safe re-evaluation:** `compare_system_baselines.py`
now rewrites stale artifact `log_path` values to the local `results/logs`
directory when reusing checkpoints, preventing old Linux absolute paths from
breaking Windows CPU evaluation. It also supports
`--reuse-learned-checkpoints`, which evaluates existing learned-baseline
checkpoints in `--output-dir` instead of launching another 300k-step baseline
training run.

**Updated comparison:**

| Method | Selection Score | Avg Delay | Task Success | Service Continuity | MEC LB | Energy / Resolved | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Attn+MAPPO | 0.4887 | 2.589s | 0.837 | 0.984 | 0.001097 | 0.2266 | checkpoint eval |
| HAN+MAPPO | 0.4852 | 2.655s | 0.828 | 0.985 | 0.001235 | 0.2291 | checkpoint eval |
| MAPPO(no-HAN) | 0.4839 | 2.690s | 0.824 | 0.981 | 0.001412 | 0.1961 | checkpoint eval |
| Min-Distance | 0.4813 | 2.786s | 0.814 | 0.985 | 0.001773 | 0.1614 | heuristic eval |
| Joint Greedy | 0.3930 | 4.218s | 0.568 | 0.941 | 0.000000 | 0.4954 | heuristic eval |
| Full-Local | 0.3878 | 4.223s | 0.557 | 0.922 | 0.000000 | 0.4857 | heuristic eval |
| Random | 0.3316 | 3.133s | 0.665 | 0.386 | 0.001509 | 0.1000 | heuristic eval |

**Diagnosis:** The new MEC-only LB metric removes the suspicious advantage of
Full-Local and Joint Greedy: both now score `0.0` because they generate no
meaningful MEC queue/CPU activity. Min-Distance is no longer inflated by
balanced serving-satellite connections and falls behind all three learned
methods under `latency_priority_score`. Attn+MAPPO remains the best method in
this three-episode re-evaluation because it has the lowest delay and highest
task success, but its lead over HAN+MAPPO is small (`0.0035` selection-score
points), so this should be treated as a promising trend rather than a strong
claim without multi-seed confidence intervals.

**Verification:**

- `pytest tests/test_compare_system_baselines_config.py tests/test_env_metrics.py tests/test_reward_function.py tests/test_baseline_plotting.py -q`
- `py_compile scripts/compare_system_baselines.py src/environment/gym_env.py`

## 2026-06-07 - Multi-User Scaling Suite Script

**Code change:** Added `scripts/run_multiuser_scaling_suite.py` to orchestrate
user-count scaling experiments for `20`, `30`, and `40` users. For each user
count, the script trains/reuses a HAN+MAPPO system run, then calls
`scripts/compare_system_baselines.py --run-mode compare_only` with
`attn_mappo`, `mappo_no_han`, `maddpg`, `pdqn`, `random`, `min_distance`,
`full_local`, and `joint_greedy`. Results are isolated under
`results/full_train_latency_priority_multiuser_u<users>_<run_id>` and
`results/baseline_compare/multiuser_scaling_<run_id>/u<users>`.

**Plotting/output change:** The script aggregates per-user
`comparison_summary.csv` files into `multiuser_summary.csv`, then generates:

- `multiuser_reward_convergence.png`
- `multiuser_core_metrics.png`
- `multiuser_resource_metrics.png`
- `suite_manifest.json`

For compact legends, `mappo_no_han` is displayed as `MAPPO`; `maddpg` remains
`MADDPG`.

**Verification:**

- `pytest tests/test_multiuser_scaling_suite.py -q`
- `python scripts/run_multiuser_scaling_suite.py --run-id drycheck --python-executable C:\Users\19704\.conda\envs\satellite.env\python.exe --device cpu --dry-run`
- `pytest tests/test_baseline_plotting.py tests/test_multiuser_scaling_suite.py -q`

**2026-06-07 portability fix:** Changed the default
`--python-executable` for `scripts/run_multiuser_scaling_suite.py` from a
machine-specific Windows Conda path to `sys.executable`. Linux/Windows runs now
inherit the Python interpreter that launched the suite, while still allowing an
explicit `--python-executable` override.

**Verification:**

- `pytest tests/test_multiuser_scaling_suite.py::test_default_python_executable_uses_current_interpreter_without_personal_path -q`
- `pytest tests/test_multiuser_scaling_suite.py tests/test_baseline_plotting.py -q`
- `python scripts/run_multiuser_scaling_suite.py --run-id drycheck_py --device cpu --dry-run`

## 2026-07-04 - Plot-Only Training Artifact Script

**Code change:** Added `scripts/plot_training_artifacts.py` as a plot-only
entry point for existing artifacts. It reads repeated `training_history.json`
inputs, optionally labeled as `LABEL=PATH`, or an existing
`comparison_summary.json`, then writes normalized `comparison_summary.json`,
`comparison_summary.csv`, `plot_manifest.json`, and comparison PNGs without
starting training, checkpoint loading, or environment evaluation.

**Plotting/output change:** The script can regenerate method bars, reward
convergence, training QoS curves, reward component curves, delay-energy
scatter, success-continuity scatter, normalized radar, and a dashboard from
artifact data alone. The first `--history` input is treated as the system
method by default; existing summary files preserve their stored `is_system`
flags. Multi-user scaling remains available through
`scripts/run_multiuser_scaling_suite.py --aggregate-only`, which already
rebuilds suite-level CSV and figures from existing per-user comparison CSVs.

**Verification:**

- `python scripts/plot_training_artifacts.py --help`
- `python -m py_compile scripts/plot_training_artifacts.py`
- `pytest tests/test_plot_training_artifacts.py tests/test_multiuser_scaling_suite.py -q`

## 2026-07-04 - Multi-User Aggregate Method Filter

**Code change:** Added `--include-methods` to
`scripts/run_multiuser_scaling_suite.py`. The option filters methods while
rebuilding aggregate artifacts from existing per-user `comparison_summary.csv`
files, so selected-method figures can be regenerated without retraining or
re-evaluating policies.

**Plotting/output change:** `aggregate_user_summaries()` now applies the
method filter before writing `multiuser_summary.csv`, and the same filter is
used for `multiuser_core_metrics.png`, `multiuser_resource_metrics.png`, and
`multiuser_reward_convergence.png`. Method selectors use the CSV `method`
names such as `han_mappo`, `mappo_no_han`, `random`, `min_distance`,
`full_local`, and `joint_greedy`.

**Verification:**

- `pytest tests/test_multiuser_scaling_suite.py -q`
- `python scripts/run_multiuser_scaling_suite.py --help`
- `python -m py_compile scripts/run_multiuser_scaling_suite.py`

## 2026-07-28 - P1 正确性修复（B05–B09）

**范围：** 修复审查文档中的 B05、B06、B07、B08、B09；B10 已在默认配置
统一工作中完成，本次未重复修改。

**代码变更：**

- B05：明确 `process_queue(t, Δt)` 处理区间为 `[t, t+Δt)`；环境使用
  slot 起始时刻调用 MEC，并按照实际消耗 cycles 插值 slot 内完成时刻。
- B06：HAN+MAPPO 评估改用独立环境和固定
  `seed + 100000 + episode_index`；评估结束后恢复训练环境、图缓存及模型模式。
- B07：任务迁移改为事务式提交。目标队列满时任务保留在源队列，并返回
  `migrated_task_ids`、`failed_task_ids`。
- B08：动作 0 统一为 stay，切换候选排除当前服务卫星；环境对
  `target_sat == serving_satellite` 再做防御性 no-op。训练、启发式基线、
  DQN、MADDPG、PDQN 和 No-HAN 路径共用同一候选接口。
- B09：MEC 更新和 deadline 结算在当前 step 产生的奖励于当前 step 发放，
  terminal/truncated 前清空 pending reward。
- checkpoint 与训练历史增加 `environment_schema_version=3`。旧环境语义下
  的 checkpoint 不允许恢复训练或用于正式比较。

**验证：**

- `pytest tests/test_p1_correctness.py -q`：6 passed
- 受影响模块交叉回归：82 passed
- `pytest -q`：110 passed
- Python 编译检查和 `git diff --check` 通过
- 2 用户、4 step、2 次独立评估的 CPU 冒烟通过；最终 checkpoint 为
  `model=2`、`geometry=2`、`environment=3`，临时结果已清理

**影响：** MEC 时延、deadline 判定、切换动作编号、任务迁移与奖励归因均已
改变。修复前 checkpoint 和评估结果不可与修复后结果直接比较，正式实验必须
重新训练全部方法。

## 2026-07-28 - P0 正确性修复

**代码变更：**

- 修复 HAN 多跳元路径的 64/256 维度错位，按 relation 区分
  `visible=5`、`serving=2`、`nearby=1`、`isl=3` 的边特征维度，并取消
  元路径异常静默跳过。
- 让 `han_num_layers` 实际控制 HAN 堆叠深度。
- rollout 保存图快照，PPO update 使用当前 HAN 参数重新编码；HAN、Actor、
  Critic 由同一损失联合反向传播，并记录 HAN 梯度范数和参数变化量。
- checkpoint schema 更新为版本 2，同时保存和恢复 HAN optimizer。
- GAE 使用当前 transition 的 done 边界，阻止跨 episode 优势泄漏。
- RVT 改为基于未来轨道位置的批量出视预测。
- 增加几何版本；卫星传播后立即刷新可见性缓存，动作继续使用生成观测时的
  候选卫星映射。

**预期影响：** 修复后模型和旧模型对应不同的网络结构、优势估计与环境转移。
旧 checkpoint 不再用于恢复训练或正式比较，后续正式结果必须重新训练。

**验证：**

- `pytest tests/test_p0_correctness.py -q`：8 passed
- `pytest -q`：104 passed
- 最小 PPO 更新中 HAN gradient norm 和 parameter delta 均为非零
- checkpoint 已验证可保存和恢复 HAN optimizer
- 训练入口 8-step CPU 冒烟通过：2 次 update，最终
  `han_grad_norm=0.13836`、`han_parameter_delta=0.09354`；临时结果已清理
- 40 用户几何推进与图构建冒烟：5 步共 `0.073s`，约 `0.015s/step`

## 2026-07-27 - 默认配置统一

**配置基准：**
`results/baseline_compare/multiuser_scaling_multiuser_6_7` 引用的
20、25、30、35、40 用户原始训练记录。

**代码变更：** 统一 `EnvConfig`、`TrainConfig`、训练 CLI 和基线比较入口。
环境与奖励默认值由 `EnvConfig` 维护，训练与评估默认值由 `TrainConfig`
维护，其余入口直接继承，不再重复写奖励常量。

**当前默认值：** `total_timesteps=300000`、`max_steps=600`、
`eval_interval=50000`、`eval_episodes=3`、`save_interval=100000`、
`graph_update_interval=1`、`log_interval=1`、
`best_model_metric=avg_delay`。奖励权重恢复为目标训练使用的时延与 QoS
优先配置。

**影响：** 只影响后续新训练和未读取历史配置的新比较任务；未运行新的训练
实验，也未修改既有结果。

**2026-07-04 suffix update:** Added `--output-suffix` for plot-only aggregate
regeneration. When provided, aggregate artifacts are written with the suffix
before the extension, for example `multiuser_core_metrics_selected.png`,
`multiuser_resource_metrics_selected.png`, `multiuser_summary_selected.csv`,
and `suite_manifest_selected.json`. This keeps the default aggregate files
untouched while generating selected-method figures in the same suite
directory.

**Verification:**

- `pytest tests/test_multiuser_scaling_suite.py -q`
- `python scripts/run_multiuser_scaling_suite.py --help`
- `python -m py_compile scripts/run_multiuser_scaling_suite.py`
