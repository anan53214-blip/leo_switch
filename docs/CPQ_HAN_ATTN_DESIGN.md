# CPQ-HAN+Attn 代码级最终方案

## 目标

新增一个算法分支 `han_attn_cpq`，显示名为 `CPQ-HAN+Attn`。

该方法在现有 `HAN+Attn` 基础上加入显式的队列拥塞、deadline 压力、未来可见性风险和共享资源约束信号。目标不是重写环境或替换 MAPPO，而是在当前最稳定、最可解释的路径上补足状态表征，让策略能同时保住 `HAN+Attn` 的服务连续性优势，并拉开与 `Attn+MAPPO` 在 `avg_delay` 和 `deadline_violation_rate` 上的差距。

当前实验显示：

- `HAN+Attn` 与 `Attn+MAPPO` 的 `avg_delay` 几乎持平。
- `HAN+Attn` 的主要优势是更少 handover、更高 service continuity。
- `Attn+MAPPO` 的能耗更低，但 handover 更频繁、continuity 更差。

因此，最可行的性能提升方向是让 `HAN+Attn` 显式看到“选这个候选卫星以后是否会排队、是否快断链、当前任务 deadline 是否紧张、全局 MEC 是否拥塞”。

## 最终选择

采用单一路线：

```text
Constraint-aware Predictive Queue-Risk HAN+Attention MAPPO
简称：CPQ-HAN+Attn
算法名：han_attn_cpq
```

不在第一阶段引入：

- HAPPO / HATRPO / HARL
- Multi-Agent Transformer / world model
- 完整 ISL routing 或 virtual compute candidate
- 新 reward 权重
- 新环境动作空间

这些方向都有价值，但第一阶段会扩大变量数量，难以判断性能提升来自哪里。CPQ-HAN+Attn 只增强状态和 actor 融合方式，因果更清晰。

## 设计原则

1. 保持现有环境、reward、动作格式不变。
2. 保留 `han_attn`、`attn_mappo`、`han_mappo` 等现有 baseline 不变。
3. 新增 `han_attn_cpq`，不要把旧方法直接替换掉。
4. 所有新增特征必须归一化到 `[0, 1]` 或稳定小范围。
5. 不手写规则强行覆盖策略选择，只把风险信息以可学习特征注入网络。
6. 先做单 seed 300k 验证方向，再做多 seed 复评。

## 代码现状

关键入口：

- `src/features/satellite_load.py`
  - 当前 `SATELLITE_LOAD_FEATURE_DIM = 8`
  - 当前 `build_satellite_load_features()` 已含 utilization、queue ratio、visible demand、SNR、RVT 等粗粒度信息。

- `scripts/train.py`
  - `AttentionMAPPOTrainer`：raw observation + satellite load attention。
  - `HANCandidateAttentionMAPPOTrainer`：HAN satellite embedding + load features。
  - CLI algorithm choices 当前包含 `mappo`、`attn_mappo`、`han_attn`、`maddpg`、`pdqn`。

- `src/model/candidate_attention.py`
  - `CandidateAttentionActor` 当前把 candidate global token 和 candidate link feature 融合。
  - 当前 actor 没有单独的 queue/deadline/RVT risk projection。

- `src/environment/mec.py`
  - `MECServer.get_estimated_wait_time()` 已能估计队列等待时间。

## 新增特征模块

### 新文件或修改文件

修改：

```text
src/features/satellite_load.py
```

新增常量：

```python
SATELLITE_CONTEXT_FEATURE_DIM = 13
SHARED_CONSTRAINT_DIM = 6
SATELLITE_RISK_FEATURE_SLICE = slice(8, 13)
```

保留旧常量：

```python
SATELLITE_LOAD_FEATURE_DIM = 8
```

这样旧算法继续使用 8 维特征，新算法使用 13 维增强上下文特征。

### 新增函数

```python
def build_satellite_context_features(env: LEOSatelliteEnv, num_agents: int) -> np.ndarray:
    ...
```

返回 shape：

```text
(env.num_satellites, SATELLITE_CONTEXT_FEATURE_DIM)
```

前 8 维兼容旧 `build_satellite_load_features()`：

| index | name | meaning |
|---:|---|---|
| 0 | `sat_id_norm` | satellite id normalized |
| 1 | `utilization` | MEC CPU utilization |
| 2 | `queue_ratio` | queue length / max queue size |
| 3 | `connected_ratio` | connected users / num agents |
| 4 | `visible_user_ratio` | visible users / num agents |
| 5 | `visible_task_demand_avg` | average visible task demand |
| 6 | `snr_avg` | average normalized SNR |
| 7 | `rvt_avg` | average normalized remaining visible time |

新增 5 维：

| index | name | computation |
|---:|---|---|
| 8 | `queue_wait_ratio` | `clip(server.get_estimated_wait_time() / 10.0, 0, 1)` |
| 9 | `compute_headroom` | `clip(server.available_freq_ghz / server.total_capacity_ghz, 0, 1)` |
| 10 | `deadline_pressure_avg` | visible users' task pressure average |
| 11 | `rvt_risk_avg` | visible links' low-RVT risk average |
| 12 | `queue_full_risk` | `clip(queue_ratio / 0.85, 0, 1)` or `1.0` when full |

`deadline_pressure_avg` 建议：

```text
if task exists:
    elapsed = max(env.current_time - task.creation_time, 0)
    remaining = max(task.max_delay - elapsed, 0)
    pressure = 1 - clip(remaining / max(task.max_delay, 1e-6), 0, 1)
else:
    pressure = 0
```

`rvt_risk_avg` 建议：

```text
risk = 1 - clip(vis.rvt_seconds / max(env.config.pre_handover_rvt_sec, 1e-6), 0, 1)
```

### 共享约束向量

新增：

```python
def build_shared_constraint_vector(env: LEOSatelliteEnv, num_agents: int) -> np.ndarray:
    ...
```

返回 shape：

```text
(SHARED_CONSTRAINT_DIM,)
```

建议 6 维：

| index | name | meaning |
|---:|---|---|
| 0 | `global_queue_ratio_mean` | all MEC queue ratios mean |
| 1 | `global_queue_ratio_max` | all MEC queue ratios max |
| 2 | `overloaded_satellite_ratio` | ratio of satellites with queue or utilization risk |
| 3 | `visible_candidate_min_wait_avg` | per-user minimum visible candidate wait, averaged |
| 4 | `active_task_pressure` | active user task pressure average |
| 5 | `handover_risk_user_ratio` | users whose serving satellite is missing or low-RVT |

该向量会拼接到每个 user observation 末尾，让 actor 和 centralized critic 都看到全局拥塞压力。

## 新算法 Trainer

修改：

```text
scripts/train.py
```

新增 class：

```python
class CPQHANCandidateAttentionMAPPOTrainer(HANCandidateAttentionMAPPOTrainer):
    algorithm_name = "han_attn_cpq"
```

核心行为：

1. 继承 `HANCandidateAttentionMAPPOTrainer` 的 HAN 编码、candidate masks、candidate ids。
2. `_init_mappo()` 中使用：

```python
fused_sat_dim = self.config.han_out_dim + SATELLITE_CONTEXT_FEATURE_DIM
obs_dim = self.raw_obs_dim + self.config.han_out_dim + 5 + SHARED_CONSTRAINT_DIM
```

3. `_encode_graph_state()` 中：

```python
observations, han_satellites, available_actions, candidate_sat_ids = super()._encode_graph_state()
context_features = build_satellite_context_features(self.env, self.num_agents)
shared_constraints = build_shared_constraint_vector(self.env, self.num_agents)
shared_repeated = np.repeat(shared_constraints[None, :], self.num_agents, axis=0)

observations = np.concatenate([observations, shared_repeated], axis=1)
satellite_tokens = np.concatenate([han_satellites, context_features], axis=1)
return observations, satellite_tokens, available_actions, candidate_sat_ids
```

注意：如果 `super()._encode_graph_state()` 已经返回 `han_satellites + old load_features`，则不要直接用它返回的 `satellite_tokens`。实现时应复用 HAN 基类的 satellite embedding，避免重复拼接旧 load features。可以抽一个小 helper 或在新类中显式调用 `HANMAPPOTrainer._encode_graph_state(self)`。

## Actor 风险融合

修改：

```text
src/model/candidate_attention.py
```

扩展 `CandidateAttentionConfig`：

```python
risk_feature_start: int = 8
risk_feature_dim: int = 5
```

在 `CandidateAttentionActor.__init__()` 中新增：

```python
if config.risk_feature_dim > 0:
    self.risk_proj = nn.Sequential(
        nn.Linear(config.risk_feature_dim, config.hidden_dim),
        nn.ReLU(),
        nn.LayerNorm(config.hidden_dim),
    )
else:
    self.risk_proj = None
```

在 `_policy_features()` 中：

1. 从 `satellite_features` gather 原始候选卫星 feature。
2. 提取 `risk_features = gathered_raw[..., risk_feature_start:risk_feature_start + risk_feature_dim]`。
3. 执行：

```python
candidate_tokens = candidate_tokens + self.risk_proj(risk_features)
```

如果 `sat_feature_dim < risk_feature_start + risk_feature_dim`，则跳过 risk projection，以保持旧算法兼容。

## 算法注册

修改：

```text
scripts/train.py
```

1. CLI choices 加：

```python
"han_attn_cpq"
```

2. trainer map 加：

```python
"han_attn_cpq": CPQHANCandidateAttentionMAPPOTrainer
```

3. `TrainConfig.algorithm` 可保留默认不变。

## Compare 与显示名

修改：

```text
scripts/compare_system_baselines.py
```

1. `DEFAULT_BASELINES` 中不默认加入 `han_attn_cpq`，避免旧 suite 自动变慢。
2. `DISPLAY_NAME_MAP` 加：

```python
"han_attn_cpq": "CPQ-HAN+Attn"
```

3. 新增训练评估函数：

```python
def train_and_evaluate_cpq_han_attn_mappo(...):
    ...
```

行为参考 `train_and_evaluate_han_attn_mappo()`，只改：

```python
exp_name="han_attn_cpq"
config.algorithm = "han_attn_cpq"
method_name="han_attn_cpq"
```

4. baseline dispatch 增加：

```python
elif baseline_name == "han_attn_cpq":
    result = train_and_evaluate_cpq_han_attn_mappo(...)
```

## 新实验 Suite

新增：

```text
scripts/run_latency_priority_g1_300k_600s_u30_cpq_suite.py
```

从当前：

```text
scripts/run_latency_priority_g1_300k_600s_u30_new_metrics_suite.py
```

复制并改名。

建议配置：

```python
exp_name = "han_attn_cpq_latency_priority_g1_300k_600s_u30"
algorithm = "han_attn_cpq"
run_label = "g1_300k_600s_u30_cpq"
```

默认 baselines：

```python
DEFAULT_BASELINES = (
    "han_attn",
    "attn_mappo",
    "han_mappo",
    "mappo_no_han",
    "min_distance",
    "random",
    "joint_greedy",
    "full_local",
)
```

这样最终 comparison 包含：

```text
CPQ-HAN+Attn
HAN+Attn
Attn+MAPPO
HAN+MAPPO
MAPPO
Min-Distance
Random
Joint Greedy
Full-Local
```

## 测试计划

### Feature tests

修改：

```text
tests/test_candidate_attention.py
```

新增测试：

```text
test_satellite_context_features_include_queue_wait_deadline_and_rvt_risk
test_shared_constraint_vector_reflects_global_queue_pressure
test_candidate_attention_logits_change_when_risk_features_change
```

检查点：

- context feature shape 为 `(num_sats, SATELLITE_CONTEXT_FEATURE_DIM)`。
- 所有新增 feature 在 `[0, 1]`。
- queue wait 增大时 `queue_wait_ratio` 增大。
- task 接近 deadline 时 `deadline_pressure_avg` 增大。
- low RVT 时 `rvt_risk_avg` 增大。
- 改变 risk feature 会改变 actor logits。

### Integration tests

修改：

```text
tests/test_han_integration.py
```

新增：

```text
test_cpq_han_attn_trainer_appends_shared_constraints_and_context_features
test_cpq_han_attn_trainer_act_path_accepts_augmented_shapes
```

检查点：

- `algorithm_name == "han_attn_cpq"`。
- observation dim 比 `han_attn` 多 `SHARED_CONSTRAINT_DIM`。
- satellite token dim 为 `han_out_dim + SATELLITE_CONTEXT_FEATURE_DIM`。
- `mappo.act()` 能跑通。
- `candidate_sat_ids` 和 `available_actions` shape 不变。

### CLI / suite tests

修改：

```text
tests/test_reward_function.py
```

新增：

```text
test_g1_u30_cpq_suite_defaults_train_cpq_han_attn_and_compare_core_methods
```

检查：

- suite 默认 algorithm 是 `han_attn_cpq`。
- baselines 包含 `han_attn` 和 `attn_mappo`。
- compare ranking metric 仍是 `avg_delay`。

## 实验计划

### Smoke

先跑极小训练：

```bash
python scripts/train.py \
  --algorithm han_attn_cpq \
  --num_users 2 \
  --max_steps 20 \
  --total_timesteps 128 \
  --n_steps 32 \
  --batch_size 32 \
  --eval_episodes 1 \
  --device cpu \
  --save_path results/han_attn_cpq_smoke
```

目标：

- 训练不崩。
- action mask、candidate ids、risk projection shape 正常。
- training history 能写出。

### 300k 单 seed

运行：

```bash
python scripts/run_latency_priority_g1_300k_600s_u30_cpq_suite.py \
  --run-id <RUN_ID>
```

主对比：

```text
CPQ-HAN+Attn vs HAN+Attn
CPQ-HAN+Attn vs Attn+MAPPO
```

### 多 seed

如果单 seed 有正向信号，再跑：

```text
seed = 42, 43, 44
```

并用现有 multiseed summary 脚本或新增汇总脚本统计 mean/std。

## 验收标准

CPQ-HAN+Attn 相对 HAN+Attn：

```text
avg_delay 降低 >= 1%
deadline_violation_rate 降低 >= 1 个百分点
task_success_rate 提升 >= 0.5 个百分点
service_continuity_rate 下降不超过 1 个百分点
energy_per_successful_task 上升不超过 5%
```

CPQ-HAN+Attn 相对 Attn+MAPPO：

```text
avg_delay 至少不差
deadline_violation_rate 更低
service_continuity_rate 明显更高
handover_frequency 明显更低
```

如果只提升 service continuity，但 `avg_delay` 和 `deadline_violation_rate` 不提升，则说明 CPQ 特征没有解决当前核心瓶颈，需要转向 action gate 或 HAPPO。

## 论文和开源参考

用于论文叙述和方案依据：

- Queue-Aware Multi-Agent DRL for LEO routing: https://arxiv.org/abs/2605.04448
- MARL task offloading with shared constraints: https://arxiv.org/abs/2509.01257
- DRL handover optimization for LEO networks: https://arxiv.org/abs/2310.20215
- HARL / HAPPO reference implementation: https://github.com/PKU-MARL/HARL
- Multi-Agent Transformer reference: https://github.com/PKU-MARL/Multi-Agent-Transformer
- Hypatia LEO simulation reference: https://github.com/snkas/hypatia

## 第一阶段不做的内容

不做 HAPPO：

- 原因：会改变优化器和训练动态，和 CPQ 特征收益混在一起。

不做 MAT/world model：

- 原因：改动训练范式，成本高，第一阶段难以稳定验证。

不做完整 ISL routing：

- 原因：需要扩展环境转发/路由动力学，超出当前 task offloading baseline 的可比范围。

不改 reward：

- 原因：当前目标是证明状态表征增强是否有效；改 reward 会削弱实验解释性。

## 最终判断

CPQ-HAN+Attn 是当前最值得实现的一条路线。它的改动集中、可测试、可解释，并且直接针对现有结果暴露出的瓶颈：`HAN+Attn` 稳但不够低时延，`Attn+MAPPO` 低时延但切换频繁、连续性差。

如果 CPQ 成功，论文叙述可以从“加入 HAN/attention”升级为：

```text
We propose a constraint-aware predictive queue-risk HAN attention policy that jointly models graph structure, candidate link quality, MEC queue pressure, deadline urgency, and future visibility risk for LEO edge offloading.
```

这比单纯说“我们用了 HAN+attention”更有系统贡献，也更贴合 LEO-MEC 的真实瓶颈。
