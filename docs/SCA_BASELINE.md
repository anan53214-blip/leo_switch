# SCA 连续卸载优化对比方法

> 2026-09-08 更新：多用户套件默认写入 `results/multiseed_0907/seed{seed}/u{num_users}`，主方法训练也在 `learned_baselines/han_pdqn` 下。`--with-sca` 可调用独立 SCA 评估并校验后汇总；旧 run-id 路径须用 `--legacy-layout`。完整命令见 [多种子目录说明](MULTISEED_LAYOUT.md)。

实现日期：2026-09-07。结果方法键 `han_pdqn_sca`，图例 **SCA**。
主方法仍为 HAN+PDQN；SCA 是复用其冻结离散决策规则的连续动作优化变体，
不是独立、无需训练的联合优化算法，也不是 LOCR。

## 运行与原始结果保护

```powershell
& 'C:\Users\19704\.conda\envs\satellite.env\python.exe' scripts/evaluate_sca_baseline.py `
  --checkpoint '<已有HAN+PDQN检查点的完整路径>' `
  --output-dir results/baseline_compare/han_pdqn_sca_u40_eval_v1 `
  --episodes 5 --device cpu --torch-threads 1
```

将占位路径替换为真实的 `best_model.pt` 或 `final_model.pt`。当前本机参考实验目录
没有 `.pt` 权重，只有训练历史和统计结果，不能从 CSV 重建策略。历史权重可在原训练机器上运行，
或者复制到本机后使用；检查点必须包含 HAN、Q 网络、参数网络及匹配的 schema（model=5、geometry=3、environment=13）。

- 新脚本只评估，不调用训练更新，不保存或覆盖模型。
- `--output-dir` 必须不存在；省略时自动创建带微秒时间戳的新目录。
- 原检查点、训练日志、已有 CSV/JSON/图表都不写入。运行日志放在新输出目录的 `runtime/` 下。
- 同一检查点分别评估 HAN+PDQN 和 SCA，两者使用相同测试 seeds。
- 默认每方法评估 **5 个 episode**，与 `TrainConfig.eval_episodes` 和统一对比脚本共用默认值。
- 默认 seeds 为 `training_seed + 1,000,000 + episode_index`，与统一对比脚本一致。训练 seed=42 时为 `1000042` 至 `1000046`。
- 这一默认值用于对齐现有实验，也与旧训练验证场景重合；若采用独立测试集，应让所有算法统一使用新场景，不能只改变 SCA。
- 可显式指定 `--evaluation-seeds 2000042 2000043 ...`，或设置 `--evaluation-seed-offset`。
- 若要与旧指标作严格配对，仍须确认权重、环境和回合长度一致。显式使用不同测试 seeds 的结果不能直接拼进旧表。
- 本实现不自动将新结果混入原实验目录；正式训练权重缺失时，不产生虚构 SCA 指标。

## 保留哪些决策

冻结的 HAN+PDQN 先计算候选连续参数，并据此选择最大 Q 值对应的卫星动作。
SCA 保留该 handover，并保留隐含的本地/卸载模式：原比例低于环境阈值时原样保留；
原请求卸载且环境准入的用户只在 `[lambda_min, 1]` 内调整比例。
拒绝准入、无任务或不能连接的用户保留原请求，由环境按原规则回退或处理。

连续网络仍参与卫星动作的 Q 值比较，只替换最终执行比例。同一状态下不改变离散规则；
不同卸载比例会改变未来状态，因此不承诺整条轨迹的切换序列与原方法相同。

## 与环境两阶段动作的接入

`FixedDiscreteSCA` 只在私有评估环境实例中临时包装 `_plan_ofdma_uplink_allocations`，
退出上下文（包括异常）立即还原。没有改动 `LEOSatelliteEnv` 类或训练默认流程。

1. 原环境先执行全部 handover，包括实际迁移、失败和重连。
2. 原环境按稳定 user-id 顺序判定准入并计算 OFDMA 带宽。
3. SCA 使用这一确定的服务卫星与准入集合，优化内部解析动作中的比例。
4. 返回原有 allocation 对象，原环境继续执行任务、推进 FCFS、结算 deadline 和原 reward。

这是一种利用已知确定性离散转移的模型辅助实现，不是增加一次可观测时间或运行第二份未来轨迹。
不重新选择卫星，不改变准入顺序，不额外优化功率、带宽或 CPU 频率。

## 实际采用的代理模型

对已准入卸载用户，固定本时隙系数：

```text
T_local(lambda)  = age + local_compute_full * (1-lambda), lambda < 1
T_local(1)       = 0
T_remote(lambda) = age + estimated_queue_wait + propagation + handover_delay
                   + lambda * (upload_airtime_full + download_airtime_full + cycles / task_rate)
E(lambda)        = local_energy_full * (1-lambda) + upload_energy_full * lambda
T(lambda)        = max(T_local(lambda), T_remote(lambda))
```

上行能耗用环境的终端实际取电模型（功放效率和电路功率），传播时延不计入发射能耗。
下行沿用环境自己的下行速率规则，不把共享上行带宽错误套给下行。

单任务计算速率代理：`整星总算力 / min(队列深度 + 本批准入人数, 并行处理槽数)`。
等待代理：`floor((当前队列深度 + 本用户批内准入序号)/处理槽数) * reference_cycles / task_rate`。
`reference_cycles` 默认 `2e9`，是明确记录、需要在验证场景校准的模型参数，不读取当前排队任务的精确剩余 cycles。
不访问私有本地 CPU 预约时间线；本地预约等待采用 **0 秒代理**，因此可能低估繁忙用户的本地等待。
任务年龄使用与观测相同的 `[0, 2*deadline]` 范围。

代理并不复现真实 FCFS、后续用户到达、本地 CPU 时间线或未来迁移。
原环境会照常处理这些因素；优化改善代理代价不等于实际回报一定改善。

## MM 与每轮凸子问题

原能耗代价 `g(E)=E/(E+E_ref)` 为凹函数。在本轮参考能耗 `E_k` 处使用全局仿射上界：

```text
g_tilde(E; E_k) = g(E_k) + E_ref/(E_k + E_ref)^2 * (E - E_k)
```

每轮 split 子问题为 LP：

```text
min  w_delay*d/deadline + w_energy*g_tilde(E(lambda); E_k)
     + M_deadline*xi/deadline
s.t. d >= T_local(lambda), d >= T_remote(lambda), d >= 0
     d <= deadline + xi, xi >= 0
     lambda_min <= lambda <= largest_float32_below_one
```

固定可行域和系数，消去 `d, xi` 后是凸分段线性一维问题。只需比较：
区间端点、两分支时延交点、各分支与 deadline 的交点。代码精确求解该 LP，
无需安装 CVXPY/OSQP。测试使用 SciPy HiGHS 独立解相同 LP，核对最优目标。

`lambda=1` 时本地分支消失，另外计算这个端点并与 LP 解比较，避免保留不存在的本地等待。
因此整体算法是含端点分支比较的 SCA/MM，不能把包含全部分支的原问题称为一个凸 LP。
初值为原网络比例。候选转为实际 float32 动作后重新检查代理代价；数值误差导致增大时保留当前比例。

优化目标为时延、非线性能耗和软 deadline 违约量。它不直接优化原奖励的成功/失败跳变、
Jain 公平性奖励或未来回报；没有额外加入 Lyapunov、RVT 选择代价。
原环境中的这些奖励和状态演化均保持原样。

## 参数及终止

| 参数 | 默认 | 作用 |
|---|---:|---|
| `--sca-max-iterations` | 5 | 每任务 MM 迭代上限 |
| `--sca-tolerance` | 1e-6 | 比例变化或代理代价改善的停止阈值 |
| `--sca-deadline-penalty` | 10 | 归一化软 deadline 违约代价 |
| `--sca-queue-reference-cycles` | 2e9 | 等待代理中的参考任务工作量 |

时延/能耗权重、能耗归一化尺度及卸载阈值从源检查点环境配置继承。
参数只能在验证场景调整；迭代上限触发单独记作 `iteration_limit`，不算收敛。
`numerical_guard` 代表阻止浮点舍入引起的代理代价上升；无效系数等数值异常记作 `fallback`，回退原比例。

## 输出与诊断

- `comparison_summary.json/.csv`、`episode_metrics.csv`：标准接口，两方法配对指标。
- `sca_manifest.json`：源配置、源检查点 SHA-256、测试 seeds、模型近似、SCA 参数、运行状态。
- `sca_diagnostics.json`：尝试/修改次数、迭代、收敛/上限/数值保护/回退、代理前后代价、准入不一致计数。
- `*_step_diagnostics.jsonl`：逐步诊断，含 episode seed 和决策耗时。
- 决策耗时包含构图/HAN、PDQN 前向和 SCA 比例修正，排除普通环境推进；单列修正耗时均值/P95/最大值。
- 新结果图例显示 **SCA**。SCA 没有单独训练，故不伪造训练收敛曲线。

`--no-plots` 可只输出统计。`--max-steps` 仅用于明确指定新评估长度，例如短流程检查。

## 实现与验证位置

2026-09-07 已通过 56 项相关测试。额外命令行检查使用随机初始化的合成测试权重、4 用户、2 回合、
每回合 64 步，验证了统计与绘图流程；这些不是正式实验结果。源参考目录的文件校验用于确认旧实验未改变。

- `src/algorithm/sca_offload.py`：纯数值 MM、解析 LP 和完整卸载端点。
- `src/algorithm/sca_adapter.py`：固定离散/激活模式的环境实例适配。
- `scripts/evaluate_sca_baseline.py`：独立配对评估入口。
- `tests/test_sca_offload.py`：HiGHS 对照、上界性质、单调性、端点、真实环境动作不变、回退、防覆盖和检查点往返。

理论借鉴：Sardellitti et al., *Joint Optimization of Radio and Computational Resources for Multicell Mobile-Edge Computing*,
IEEE TSIPN 2015，https://arxiv.org/abs/1412.8416；Razaviyayn et al., *A Unified Convergence Analysis of Block Successive Minimization Methods*,
SIAM J. Optimization 2013，https://arxiv.org/abs/1209.2385。
这是针对本项目固定动作边界重新推导的简化实现，不是对上述论文全部算法的复现。
