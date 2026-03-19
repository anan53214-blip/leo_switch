# 第一次训练结果分析 (v1)

> 实验名称: `han_mappo_standard` | 种子: 42 | 设备: CUDA  
> 总步数: ~175K | Episode 长度: 2000 步 | 用户数: 10

---

## 1. 总体结论

训练**部分收敛但存在严重的 PPO 机制失效问题**。Reward 从 -320 上升到 -307 后停滞，评估 reward 持续下降，核心原因是 KL 散度爆炸导致 clip 机制完全失效。

---

## 2. 各指标详细分析

### 2.1 Reward Convergence（奖励收敛）

| 指标 | 值 |
|------|-----|
| 初始 reward | ~-320 |
| 最终 reward | ~-307 |
| 收敛位置 | ~75K 步后平稳 |

- Reward 为负值是**奖励函数设计决定的**（惩罚项累积远大于正向奖励）
- 从 -320 → -307 说明智能体有学习，但 75K 步后陷入局部最优

### 2.2 Loss 曲线

| 指标 | 初始值 | 最终值 | 趋势 |
|------|--------|--------|------|
| Actor Loss | 0.34 | 0.29 | 稳步下降 ✅ |
| Critic Loss | 0.05 | 0.03 | 先升后降 ✅ |

- Loss 曲线表面正常，但由于 clip 失效，actor loss 的下降并不代表策略在有效改进

### 2.3 Entropy & KL ⚠️ 严重问题

| 指标 | 值 | 正常范围 | 状态 |
|------|-----|----------|------|
| Policy Entropy | ~2.06（平坦） | 应缓慢下降 | ❌ 异常 |
| KL Divergence | **~125** | < 0.01~0.02 | ❌ 严重异常 |

- KL = 125 意味着新旧策略差异巨大，PPO 的信任域约束完全失效
- Entropy 快速上升后固化，策略探索能力冻结

### 2.4 PPO Clip Fraction ⚠️ 严重异常

| 指标 | 值 | 正常范围 |
|------|-----|----------|
| Clip Fraction | **1.0**（恒定） | 0.1~0.3 |

- 100% 样本被 clip，PPO 退化为简单 policy gradient
- 根因：10 个 epoch 重复优化 + 无 KL 早停

### 2.5 Handover & Task Rate

| 指标 | 值 | 变化趋势 |
|------|-----|----------|
| Handover Success | ~95% | 恒定（硬编码） |
| Task Completion | ~50% | 无改善 ❌ |
| Deadline Violations | 112~122 | 后期上升 ❌ |

- 切换成功率由代码硬编码 95%，非学习结果
- 任务完成率始终 ~50%，卸载策略未被有效优化

### 2.6 Delay & Energy（震荡）

| 指标 | 范围 | 趋势 |
|------|------|------|
| Avg Delay | 3600~3900 ms | 震荡，后期上升 ❌ |
| Total Energy | 66~74 J | 震荡，后期上升 ❌ |

- 震荡原因：KL 爆炸导致策略每次更新变化剧烈
- 后期上升与评估 reward 下降一致，说明过拟合

### 2.7 Eval Reward（评估奖励）⚠️ 下降

| 指标 | 初始 | 最终 |
|------|------|------|
| Eval Reward | -380 | -387 |

- 训练 reward 上升但评估 reward 下降 → **过拟合**
- 策略泛化能力差

---

## 3. 根本原因

1. **KL 散度爆炸**：`n_epochs=10` 重复优化 + `target_kl=None`（无早停），每次更新策略偏移过大
2. **Clip 机制失效**：Clip Fraction = 1.0，PPO 核心约束无效
3. **奖励信号不平衡**：惩罚项量级远大于正向奖励，梯度信号以惩罚为主
4. **探索不足**：`entropy_coef=0.01` 过小，entropy 快速固化

---

## 4. 修复计划

| 修改项 | 原值 | 新值 | 原因 |
|--------|------|------|------|
| `target_kl` | `None` | `0.015` | 启用 KL 早停，防止策略偏移过大 |
| `n_epochs` | `10` | `4` | 减少重复优化次数 |
| `learning_rate` | `3e-4` | `5e-5` | 降低学习率，稳定训练 |
| `entropy_coef` | `0.01` | `0.05` | 增加探索 |
| 奖励函数 | 正向奖励小 | 增大正向奖励系数 | 平衡奖惩信号 |

---

## 5. 关键代码位置

- MAPPO 配置: `src/algorithm/mappo.py` MAPPOConfig
- PPO 更新: `src/algorithm/mappo.py` MAPPO.update()
- 奖励函数: `src/environment/gym_env.py` _execute_user_action() / _execute_offloading()
- 训练入口: `src/algorithm/runner.py` Runner

---

## 6. v1 → v2 参数修改记录

> 修改时间: 2026-03-16

### 6.1 MAPPO 算法参数 (`src/algorithm/mappo.py` MAPPOConfig)

| 参数 | v1 值 | v2 值 | 修改原因 |
|------|-------|-------|----------|
| `target_kl` | `None` | `0.015` | 启用 KL 早停，防止策略偏移过大（v1 KL=125，远超正常值） |
| `entropy_coef` | `0.01` | `0.05` | 增大熵系数鼓励探索，v1 entropy 过早固化 |
| `learning_rate` | `3e-4` | `5e-5` | 降低学习率 6 倍，稳定训练过程 |
| `n_epochs` | `10` | `4` | 减少 PPO epoch 数，防止单次更新中策略偏移过大 |

### 6.2 KL 早停逻辑修复 (`src/algorithm/mappo.py` MAPPO.update())

```python
# v1（有 bug，分母可能为 0）:
if np.mean(all_kl_divs[-len(all_kl_divs)//self.config.n_epochs:]) > self.config.target_kl:
    break

# v2（修复后，检查最新一个 batch 的 KL）:
if all_kl_divs[-1] > 1.5 * self.config.target_kl:
    break
```

### 6.3 奖励权重 (`src/environment/gym_env.py` EnvConfig)

| 参数 | v1 值 | v2 值 | 修改原因 |
|------|-------|-------|----------|
| `reward_delay_weight` | `0.4` | `1.0` | 增大 2.5 倍，提升时延优化信号 |
| `reward_energy_weight` | `0.3` | `0.8` | 增大 2.7 倍，提升能耗优化信号 |
| `reward_handover_weight` | `0.2` | `0.5` | 增大 2.5 倍，提升切换质量信号 |
| `reward_qos_weight` | `0.1` | `0.3` | 增大 3 倍，提升 QoS 满足信号 |

### 6.4 切换正向奖励 (`src/environment/gym_env.py` _execute_handover())

| 奖励项 | v1 值 | v2 值 |
|--------|-------|-------|
| 仰角奖励系数 | `0.1` | `0.3` |
| RVT 奖励系数 | `0.1` | `0.3` |

### 6.5 训练脚本同步更新

**`scripts/train.py` TrainConfig:**

| 参数 | v1 值 | v2 值 |
|------|-------|-------|
| `learning_rate` | `3e-4` | `5e-5` |
| `entropy_coef` | `0.01` | `0.05` |
| `n_epochs` | `10` | `4` |

**`scripts/run_server_training.py` CONFIG:**

| 参数 | v1 值 | v2 值 |
|------|-------|-------|
| `learning_rate` | `1e-4` | `5e-5` |
| `entropy_coef` | `0.03` | `0.05` |
| `n_epochs` | `10` | `4` |

### 6.6 预期改善

- Clip Fraction 从 1.0 降至 0.1~0.3（正常范围）
- KL Divergence 从 ~125 降至 < 0.02
- 正向奖励量级增大 ~2.5 倍，平衡奖惩信号
- Eval Reward 不再持续下降（减轻过拟合）

---

# 第二次训练结果分析 (v2)

> 实验名称: `han_mappo_standard` | 种子: 42 | 设备: CUDA
> 总步数: 65,536 / 1,000,000（6.5%，被早停终止） | Episode 长度: 2000 步 | 用户数: 10
> 训练时间: 9545s (~2.65h)

---

## 1. 总体结论

v2 的参数调整（降低 lr、减少 epoch、增大 entropy_coef、启用 target_kl）**未能解决核心问题**。KL 散度从 v1 的 ~125 降至 ~60-83，Clip Fraction 仍恒定为 1.0。训练仅完成 6.5% 即被早停终止。

**根本原因不在超参数，而在 Actor 网络的连续动作分布实现缺陷**：使用 `Normal` 分布 + `clamp(0,1)` 导致 `log_prob` 在采样时与评估时不一致，ratio 爆炸，PPO 信任域约束彻底失效。

---

## 2. 各指标详细分析

### 2.1 Reward Convergence（奖励收敛）

| 指标 | v1 值 | v2 值 | 变化 |
|------|-------|-------|------|
| 初始 reward | ~-320 | ~394 | ↑（奖励权重增大） |
| 最终 reward | ~-307 | ~397 | ↑ |
| Best eval reward | -380 | 264 | ↑（正值化） |
| 收敛趋势 | 75K 后平稳 | 全程平坦 ❌ | 无改善 |

- v2 reward 为正值是因为奖励权重增大了 2.5 倍，**并非策略改善**
- recent_mean_reward 从 394 → 397，32 个 update 仅提升 0.8%，本质上无学习

### 2.2 Loss 曲线

| 指标 | v1 初始→最终 | v2 初始→最终 | 状态 |
|------|-------------|-------------|------|
| Actor Loss | 0.34 → 0.29 | 0.35 → 0.31 | 微降，无实质改善 ❌ |
| Critic Loss | 0.05 → 0.03 | 0.089 → 0.099 | 略升 ❌ |

- Critic loss 上升说明价值函数拟合变差，可能因为每次 update 只执行 1 个 mini-batch（KL 早停立即触发）

### 2.3 Entropy & KL ⚠️ 核心问题未解决

| 指标 | v1 值 | v2 值 | 正常范围 | 状态 |
|------|-------|-------|----------|------|
| Policy Entropy | ~2.06 | 1.38 → 1.62 | 应缓慢下降 | ❌ 持续上升 |
| KL Divergence | **~125** | **60 → 83** | < 0.02 | ❌ 仍超正常值 4000 倍 |

- KL 从 125 降至 60-83，降幅 ~35%，但仍超正常值数千倍
- `target_kl=0.015` 的早停在每个 epoch 的第一个 batch 就触发（实际 KL >> 0.0225），导致每次 update 只训练 1 个 mini-batch
- Entropy 持续上升（1.38→1.62）说明 `entropy_coef=0.05` 在推动探索，但策略本身未收敛

### 2.4 PPO Clip Fraction ⚠️ 完全未改善

| 指标 | v1 值 | v2 值 | 正常范围 |
|------|-------|-------|----------|
| Clip Fraction | **1.0** | **1.0** | 0.1~0.3 |

- 100% 样本被 clip，与 v1 完全相同
- 证明问题不在超参数（lr、n_epochs），而在 log_prob 计算本身

### 2.5 Handover & Task Rate

| 指标 | v1 值 | v2 值 | 变化 |
|------|-------|-------|------|
| Handover Success | ~95% | ~95% | 无变化（硬编码） |
| Task Completion | ~50% | ~48% | 无改善 ❌ |
| Deadline Violations | 112~122 | 100~140 | 震荡加剧 ❌ |

### 2.6 Delay & Energy

| 指标 | v1 范围 | v2 范围 | 趋势 |
|------|---------|---------|------|
| Avg Delay | 3600~3900 ms | 3.5~4.5 s | 震荡 ❌ |
| Total Energy | 66~74 J | 56~87 J | 震荡加剧 ❌ |

- 能耗震荡范围从 v1 的 8J 扩大到 v2 的 31J，策略更不稳定

### 2.7 Eval Reward（评估奖励）

| 评估点 | Steps | Eval Mean Reward | Eval Std |
|--------|-------|-----------------|----------|
| 第1次 | 20,480 | 257.48 | 4.56 |
| 第2次 | 40,960 | 256.05 | 6.66 |
| 第3次 | 61,440 | 263.94 | 7.11 |

- Eval reward 基本平坦（257→264），无显著学习趋势
- 标准差逐渐增大（4.56→7.11），策略稳定性下降

---

## 3. 根本原因

### 3.1 核心 Bug：Normal 分布 + clamp 导致 log_prob 不一致

**位置**: `src/model/actor.py` `HybridActor.sample()` (L232-233) 与 `HybridActor.evaluate()` (L266-271)

**机制**:
1. `sample()` 中：`offload_action = offload_dist.rsample()` 后执行 `torch.clamp(offload_action, 0.0, 1.0)`
2. clamp 后的 action 存入 buffer，对应的 `old_log_prob` 是用 clamp 前的分布参数计算的
3. `evaluate()` 中：用新的网络参数重建 Normal 分布，对 clamp 后的 action 计算 `new_log_prob`
4. 当 action 被 clamp 到 0 或 1 边界时，新旧分布的 mean/std 差异会导致 `log_prob` 差异巨大
5. `ratio = exp(new_log_prob - old_log_prob)` 爆炸 → KL 爆炸 → Clip Fraction = 1.0

**数学解释**:
- Normal(μ₁, σ) 在 x=0 处的 log_prob = -0.5 * (0-μ₁)²/σ²
- Normal(μ₂, σ) 在 x=0 处的 log_prob = -0.5 * (0-μ₂)²/σ²
- 差值 = 0.5 * (μ₂² - μ₁²) / σ²，当 σ=0.5 时，μ 变化 0.1 就能导致 ratio 变化 e^0.4 ≈ 1.49

### 3.2 次要问题

1. **KL 早停过于激进**：每次 update 只执行 1 个 mini-batch，Critic 训练严重不足
2. **奖励信号仍不平衡**：正向奖励增大后 reward 变正，但策略梯度信号仍被 ratio 爆炸淹没
3. **缺少 ratio clamp 保护**：PPO 实现中没有对 ratio 做额外 clamp，极端 ratio 值影响梯度

---

## 4. 修复计划

| 优先级 | 修改项 | 原实现 | 新实现 | 原因 |
|--------|--------|--------|--------|------|
| P0 | 连续动作分布 | `Normal` + `clamp(0,1)` | `Beta` 分布 | Beta 分布天然支持 [0,1]，log_prob 一致 |
| P0 | ratio 保护 | 无 | `ratio = clamp(ratio, 0.0, 10.0)` | 防止极端 ratio 导致梯度爆炸 |
| P1 | KL 早停阈值 | `1.5 * 0.015 = 0.0225` | `1.5 * 0.02 = 0.03` | 略微放宽，确保每次至少训练几个 batch |
| P1 | action_std_init | `0.5` | 移除（Beta 分布不需要） | Beta 分布用 α,β 参数控制形状 |

---

## 5. 关键代码位置

- **Bug 位置**: `src/model/actor.py` L148-154（Normal 分布参数）、L232-233（clamp）、L266-271（evaluate log_prob）
- MAPPO 更新: `src/algorithm/mappo.py` L320-332（ratio 计算）、L398-401（KL 早停）
- 奖励函数: `src/environment/gym_env.py` L409-639
- 训练入口: `scripts/run_server_training.py` L56-96

---

## 6. v2 → v3 参数修改记录

> 修改时间: 2026-03-17

### 6.1 Actor 连续动作分布重构 (`src/model/actor.py` HybridActor)

| 项目 | v2 实现 | v3 实现 |
|------|---------|---------|
| 分布类型 | `Normal(mean, std)` + `clamp(0,1)` | `Beta(alpha, beta)` |
| 参数化 | `offload_mean` (Sigmoid) + `offload_log_std` (Parameter) | `offload_alpha` + `offload_beta` (Softplus+1) |
| 采样 | `rsample()` + `clamp` | `rsample()`（天然 [0,1]） |
| log_prob | 不一致（clamp 破坏） | 一致（Beta 分布原生支持） |

### 6.2 MAPPO ratio 保护 (`src/algorithm/mappo.py` MAPPO.update())

```python
# v2: 无保护
ratio = torch.exp(new_log_probs - old_log_probs)

# v3: 添加 ratio clamp
log_ratio = new_log_probs - old_log_probs
log_ratio = torch.clamp(log_ratio, -20.0, 2.0)
ratio = torch.exp(log_ratio)
```

### 6.3 KL 早停微调 (`src/algorithm/mappo.py` MAPPOConfig)

| 参数 | v2 值 | v3 值 | 修改原因 |
|------|-------|-------|----------|
| `target_kl` | `0.015` | `0.02` | 略微放宽，确保 Critic 有足够训练 |

### 6.4 预期改善

- KL Divergence 从 ~60-83 降至 < 0.05（Beta 分布消除 log_prob 不一致）
- Clip Fraction 从 1.0 降至 0.1~0.3
- Reward 出现明显上升趋势
- Eval Reward 与 Train Reward 趋势一致（消除过拟合）

---

# 第三次训练结果分析 (v3)

> 实验名称: `han_mappo_standard` | 种子: 42 | 设备: CUDA
> 总步数: 200,704 / 1,000,000（20%，被早停终止） | Episode 长度: 2000 步 | 用户数: 10
> 训练时间: 30866s (~8.6h) | 总更新次数: 98 | 评估次数: 10

---

## 1. 总体结论

v3 的 Beta 分布重构**彻底解决了 v1/v2 的 KL 爆炸和 Clip 失效问题**（KL: 0.0002~0.006，Clip Fraction: 0~0.01）。PPO 信任域约束恢复正常工作。

但训练暴露出**新的瓶颈**：
1. **Reward 先升后降**：从 305 上升到峰值 461（~50K 步），随后回落至 373，被早停终止
2. **Entropy 持续上升**（0.57→0.63），策略在变得更随机而非更精确
3. **Eval reward（~550）远高于 Train reward（~400）**，确定性策略优于随机策略，说明 `entropy_coef=0.05` 过大
4. **任务完成率停滞**在 35%~50%，卸载策略未被有效优化

---

## 2. 各指标详细分析

### 2.1 Reward Convergence（奖励收敛）

| 指标 | v2 值 | v3 值 | 变化 |
|------|-------|-------|------|
| 初始 reward | ~394 | ~305 | ↓（正常波动） |
| 峰值 reward | ~397 | **~461**（50K步） | ↑ 有学习 ✅ |
| 最终 reward | ~397 | ~373 | ↓ 回落 ❌ |
| Best eval reward | 264 | **583.5** | ↑ 大幅改善 ✅ |
| 收敛趋势 | 全程平坦 | 先升后降 | 有学习但不稳定 ⚠️ |

- 前 50K 步 reward 从 305 → 461，说明 Beta 分布修复后策略确实在学习
- 50K 步后 reward 持续下降至 373，被早停终止（patience=30）
- Eval reward 远高于 train reward（583 vs 461），说明确定性策略质量好，但训练时随机性过高

### 2.2 Loss 曲线

| 指标 | v2 初始→最终 | v3 初始→最终 | 状态 |
|------|-------------|-------------|------|
| Actor Loss | 0.35 → 0.31 | -0.0002 → -0.0002 | 极小且平坦 ⚠️ |
| Critic Loss | 0.089 → 0.099 | 0.06 → 0.10 | 先升后降 ⚠️ |

- Actor loss 量级极小（~1e-4），说明 ratio 非常接近 1.0（clip fraction 极低），策略更新步长过小
- Critic loss 在中期升至 0.145 后回落，价值估计不够准确

### 2.3 Entropy & KL ✅ KL 问题已解决

| 指标 | v2 值 | v3 值 | 正常范围 | 状态 |
|------|-------|-------|----------|------|
| Policy Entropy | 1.38 → 1.62 | 0.57 → **0.63** | 应缓慢下降 | ❌ 持续上升 |
| KL Divergence | **60 → 83** | **0.0005 → 0.001** | < 0.02 | ✅ 正常 |

- KL 从 v2 的 60~83 降至 v3 的 0.0002~0.006，**完全恢复正常**
- 但 Entropy 持续上升（0.57→0.63），`entropy_coef=0.05` 过大，熵奖励主导了梯度信号
- 策略在被推向更随机而非更优

### 2.4 PPO Clip Fraction ✅ 已修复但过低

| 指标 | v2 值 | v3 值 | 正常范围 |
|------|-------|-------|----------|
| Clip Fraction | **1.0** | **0.0~0.01** | 0.1~0.3 |

- 从 v2 的 1.0 降至 v3 的 ~0.003，**clip 机制恢复正常**
- 但 clip fraction 过低（< 0.01），说明策略更新步长过小，学习效率低
- 原因：`learning_rate=5e-5` 过低 + `n_epochs=4` 偏少，策略每次更新变化极小

### 2.5 Handover & Task Rate

| 指标 | v2 值 | v3 值 | 变化 |
|------|-------|-------|------|
| Handover Success | ~95% | ~94% | 无变化（硬编码） |
| Task Completion | ~48% | **35%~54%** | 震荡，无稳定改善 ❌ |
| Deadline Violations | 100~140 | 100~130 | 无改善 ❌ |

- 任务完成率在 eval 中仅 ~35%，比 train 中的 ~48% 更低
- 说明确定性策略的卸载比例选择不够好，Beta 分布的 mean 偏离最优

### 2.6 Delay & Energy

| 指标 | v2 范围 | v3 范围 | 趋势 |
|------|---------|---------|------|
| Avg Delay | 3.5~4.5 s | 4.0~5.1 s | 偏高 ❌ |
| Total Energy | 56~87 J | 68~80 J | 震荡收窄 ⚠️ |

- 时延偏高（4~5s），eval 中更高（~5.0s），卸载策略未有效降低时延
- 能耗震荡范围从 v2 的 31J 收窄到 v3 的 12J，稳定性改善

### 2.7 Eval Reward（评估奖励）

| 评估点 | Steps | Eval Mean Reward | Eval Std | Task Rate |
|--------|-------|-----------------|----------|-----------|
| 第1次 | 20,480 | 549.16 | 10.94 | 35.0% |
| 第2次 | 40,960 | 530.26 | 5.71 | 34.5% |
| 第3次 | 61,440 | 526.54 | 5.35 | 34.3% |
| 第4次 | 81,920 | 507.72 | 10.40 | 33.4% |
| **第5次** | **100,352** | **583.51** | **18.22** | **37.8%** |
| 第6次 | 120,832 | **102.11** | 11.23 | 39.3% |
| 第7次 | 141,312 | 540.72 | 11.12 | 35.5% |
| 第8次 | 161,792 | 513.97 | 7.49 | 33.3% |
| 第9次 | 180,224 | 577.33 | 8.88 | 37.3% |
| 第10次 | 200,704 | 567.54 | 2.61 | 36.4% |

- Eval reward 整体在 507~583 范围，远高于 train reward（305~461）
- **第6次评估异常暴跌至 102**（120K步），可能是环境随机性或策略突变
- Eval std 最终降至 2.61，策略稳定性在改善
- 但 task completion rate 始终 ~35%，未随训练改善

---

## 3. 根本原因

### 3.1 核心问题：Entropy 过大导致策略过度探索

- `entropy_coef=0.05` 在 Beta 分布下产生的熵奖励梯度过大
- Beta 分布的 entropy 范围与 Normal 分布不同，0.05 的系数使熵奖励主导了 actor loss
- 表现：entropy 持续上升（0.57→0.63），策略越来越随机，reward 先升后降

### 3.2 学习率过低 + Clip Fraction 过低

- `learning_rate=5e-5` 配合 Beta 分布后，策略更新步长极小
- Clip fraction ~0.003（正常应 0.1~0.3），说明 ratio 几乎不触发 clip
- 策略每次更新变化太小，学习效率低，100K 步后被 entropy 推向随机

### 3.3 Critic 训练不足

- `batch_size=128`，`n_steps=2048`，每个 epoch 仅 ~16 个 mini-batch
- Critic loss 偏高（0.06~0.14），价值估计不准导致 advantage 信号噪声大
- 建议增大 `n_epochs` 或减小 `batch_size` 以增加 Critic 训练量

### 3.4 Eval 异常暴跌

- 第6次评估（120K步）reward 从 583 暴跌至 102，可能原因：
  - 策略在该阶段 entropy 上升导致确定性策略退化
  - 环境随机性（单次评估 5 个 episode 样本量小）

---

## 4. 修复计划

| 优先级 | 修改项 | v3 值 | v4 值 | 原因 |
|--------|--------|-------|-------|------|
| P0 | `entropy_coef` | `0.05` | `0.01` | Beta 分布下 0.05 过大，降低以停止 entropy 上升 |
| P0 | `learning_rate` | `5e-5` | `3e-4` | 提高学习率，增大策略更新步长，提升 clip fraction |
| P1 | `n_epochs` | `4` | `10` | 增加每次更新的训练轮数，充分利用数据 |
| P1 | `batch_size` | `128` | `64` | 减小 batch 增加 mini-batch 数量，Critic 训练更充分 |
| P2 | `entropy_coef` 衰减 | 固定 | 线性衰减 0.01→0.001 | 训练后期减少探索，收敛到更优策略 |
| P2 | `early_stop_patience` | `30` | `50` | 放宽早停，给策略更多学习时间 |

---

## 5. 关键代码位置

- Entropy 系数: [`MAPPOConfig.entropy_coef`](src/algorithm/mappo.py:85)
- 学习率: [`MAPPOConfig.learning_rate`](src/algorithm/mappo.py:88)
- PPO 更新: [`MAPPO.update()`](src/algorithm/mappo.py:267)
- 训练配置: [`STANDARD_CONFIG`](scripts/run_server_training.py:56)
- 训练脚本: [`TrainConfig`](scripts/train.py:70)

---

## 6. v3 → v4 参数修改记录

> 修改时间: 2026-03-18

### 6.1 MAPPO 算法参数 (`src/algorithm/mappo.py` MAPPOConfig)

| 参数 | v3 值 | v4 值 | 修改原因 |
|------|-------|-------|----------|
| `entropy_coef` | `0.05` | `0.01` | Beta 分布下 0.05 过大，entropy 持续上升导致策略退化 |
| `learning_rate` | `5e-5` | `3e-4` | 提高学习率 6 倍，clip fraction 从 0.003 提升到正常范围 |
| `n_epochs` | `4` | `10` | 增加训练轮数，充分利用每次 rollout 数据 |
| `batch_size` | `64` | `64` | 保持不变（mappo 默认已是 64） |
| `target_kl` | `0.02` | `0.02` | 保持不变，KL 已正常 |

### 6.2 Entropy 衰减机制 (`src/algorithm/mappo.py` MAPPO.update())

```python
# v3: 固定 entropy_coef
entropy_coef = self.config.entropy_coef

# v4: 线性衰减 entropy_coef
entropy_coef = self.config.entropy_coef * max(1.0 - self.train_step / 300, 0.1)
```

### 6.3 训练脚本同步更新

**`scripts/run_server_training.py` STANDARD_CONFIG:**

| 参数 | v3 值 | v4 值 |
|------|-------|-------|
| `learning_rate` | `5e-5` | `3e-4` |
| `entropy_coef` | `0.05` | `0.01` |
| `n_epochs` | `4` | `10` |
| `batch_size` | `128` | `64` |
| `early_stop_patience` | `30` | `50` |
| `save_path` | `results/full_train_v3` | `results/full_train_v4` |

**`scripts/train.py` TrainConfig:**

| 参数 | v3 值 | v4 值 |
|------|-------|-------|
| `learning_rate` | `5e-5` | `3e-4` |
| `entropy_coef` | `0.05` | `0.01` |
| `n_epochs` | `4` | `10` |
| `batch_size` | `64` | `64` |

### 6.4 深层代码漏洞修复

**Bug 1: GAE 使用 mean_reward 抹平个体差异** (`src/algorithm/buffer.py` L454-491)

| 项目 | v3 实现 | v4 实现 |
|------|---------|---------|
| advantage 维度 | `(buffer_size,)` 共享标量 | `(buffer_size, num_agents)` per-agent |
| reward 使用 | `self.rewards[step].mean()` | `self.rewards[step]` 每个 agent 独立 |
| get_batches 展平 | `np.repeat(advantages, num_agents)` | `advantages.reshape(-1)` 直接展平 |

影响：v3 中所有 agent 共享相同 advantage，个体奖励差异被抹平，agent 无法学到差异化卸载策略。

**Bug 2: Critic 训练/推理输入不一致** (`src/algorithm/mappo.py` L338-344)

```python
# v3: update() 中只传单 agent obs，Critic 只看到 1 个 agent
obs_reshaped = obs.unsqueeze(1)  # (batch, 1, obs_dim)
new_values = self.critic(obs_reshaped, satellite_embeddings)

# v4: 使用 global_states 重构所有 agent 信息，与 act() 一致
global_states = batch['global_states']
gs_reshaped = global_states.view(-1, num_agents, obs_dim)
new_values = self.critic(gs_reshaped, satellite_embeddings)
```

影响：v3 中 act() 时 Critic 看到所有 agent 的 mean pooling，update() 时只看到单 agent，value 估计不一致导致 critic loss 偏高。

**Bug 3: 早停阈值过严** (`scripts/train.py` L719)

```python
# v3: 绝对阈值 0.5，对 reward 量级 300~500 仅 0.1%
current_reward > es_best_reward + 0.5

# v4: 相对阈值 0.1%
current_reward > es_best_reward * 1.001
```

影响：v3 中 reward 正常波动即触发早停计数，导致训练仅完成 20% 就被终止。

### 6.5 预期改善

- Entropy 从持续上升转为缓慢下降（0.5 → 0.3）
- Clip Fraction 从 ~0.003 提升至 0.1~0.3（正常范围）
- Reward 持续上升而非先升后降
- Train reward 与 Eval reward 差距缩小
- 任务完成率从 ~35% 提升至 >50%（per-agent advantage 使个体策略可优化）
- Critic loss 下降（训练/推理输入一致）
- 训练不再过早终止（早停阈值合理化）

### 6.6 全面代码审查补充修复

**漏洞 7：奖励双重除法** (`scripts/train.py` collect_rollouts)
```python
# v3 BUG: 环境返回 mean_reward，再除以 num_agents → 每个 agent 只拿到 1/N 的均值
agent_rewards = np.full(self.num_agents, reward / self.num_agents)
# v4: 环境返回的已是均值，直接分配给每个 agent
agent_rewards = np.full(self.num_agents, reward)
```
影响：每个 agent 的 reward 信号被缩小 N 倍（N=5 时缩小 5 倍），导致 advantage 偏小、策略更新动力不足。

**漏洞 8：Runner 缺少 satellite_embeddings** (`src/algorithm/runner.py` train)
```python
# v3 BUG: buffer.add() 缺少 satellite_embeddings 参数，导致 TypeError
# v4: 显式传入 None
self.buffer.add(..., satellite_embeddings=None, ...)
```

**漏洞 9：RVT 上升/下降阶段公式相同** (`src/environment/gym_env.py` _estimate_rvt)
```python
# v3 BUG: 上升和下降阶段使用完全相同的公式
remaining_fraction = (math.pi - phase) / math.pi  # 两个分支一样
# v4: 下降阶段剩余时间更短
if is_descending:
    remaining_fraction = (math.pi / 2 - phase) / math.pi
else:
    remaining_fraction = (math.pi - phase) / math.pi
```
影响：下降阶段高估 RVT，导致 agent 在卫星即将不可见时仍不切换，增加被动切换和阻塞。

### 6.7 端到端验证与冒烟测试

**数据流维度验证**（通过静态审查确认）：
- FeatureExtractor → sat(66,10), user(5,13), edge_us(E,6), edge_isl(E,3) ✅
- HAN 投影 → hidden_dim, 多头注意力 → output_proj → (N, out_dim=64) ✅
- 观测拼接 = HAN(64) + rvt_warning(1) + task(4) = 69 ✅
- Buffer shapes: obs(T,N,69), rewards(T,N), advantages(T,N), returns(T,N) ✅
- Critic: user_encoder(69→H/2) + sat_encoder(64→H/2) → concat → value_net ✅
- get_batches 展平: (T*N, ...) + global_states/satellite_embeddings 按 agent 复制 ✅

**冒烟测试结果**（conda env: satellite.env, torch 2.5.1+cu121）：
```
1. 环境创建 → obs shape (5, 73) ✅
2. HAN 编码 → user (5, 64), sat (66, 64) ✅
3. MAPPO act → handover + offload + value ✅
4. Buffer add × 10 步 → pos=10 ✅
5. GAE → advantages (10, 5), returns (10, 5) ✅
6. PPO Update → actor_loss=-0.0001, critic_loss=2.59, entropy=2.31, kl=0.0001 ✅
```

**结论**：所有模块 import 正常，端到端数据流无崩溃，维度一致性验证通过。系统已准备好进行 v4 训练。

---

## 7. v4 性能优化记录

> 修改时间: 2026-03-19

### 7.1 问题描述

v4 训练速度过慢，profiling 发现 512 步 rollout 在 CPU 上需要 113s，97% 时间消耗在 PyTorch 模型推理（`torch.linear` 57%、`torch.dropout` 11%、`torch.softmax` 8%）。环境计算仅占 3%。

### 7.2 优化内容

#### 优化 1: 星座位置更新向量化 (`src/environment/constellation.py`)

| 项目 | 优化前 | 优化后 |
|------|--------|--------|
| `_update_all_positions()` | Python 循环 66 颗卫星，逐个做三角函数+矩阵旋转 | numpy 批量计算，预分配数组 `_all_pos_ecef`/`_all_vel_eci` |
| 新增 `_init_vectorized_arrays()` | 无 | 预计算旋转矩阵 `_R0`/`_R1`、初始真近点角 `_ta0` |

#### 优化 2: 可见性计算向量化 (`src/environment/gym_env.py`)

| 项目 | 优化前 | 优化后 |
|------|--------|--------|
| `_get_visible_satellites()` | 调用 `VisibilityCalculator` 逐卫星循环 | 新增 `_compute_visibility_batch()` numpy 向量化 |
| 用户几何 | 每次重新计算 ECEF/ENU | `_precompute_user_geometry()` 预计算（用户不移动） |

#### 优化 3: 可见性缓存 bug 修复 (`src/environment/gym_env.py`)

```python
# BUG: _invalidate_visibility_cache 设置 cache_step = current_step
# 但 step() 中 current_step += 1 在 _get_observation() 之前执行
# 导致 _get_observation() 中缓存检查 cache_step != current_step，80% 缓存失效

# 修复: 改用 dict.clear() 简单清空，不依赖 step 编号
def _invalidate_visibility_cache(self):
    self._visibility_cache.clear()

def _get_visible_satellites(self, user):
    uid = user.user_id
    if uid in self._visibility_cache:  # 不再检查 step
        return self._visibility_cache[uid]
    ...
```

效果：`_compute_visibility_batch` 调用次数从 26038 降至 5150（-80%），耗时从 1.58s 降至 0.44s。

#### 优化 4: HAN 图编码缓存 (`scripts/train.py` `_encode_graph_state()`)

| 项目 | 优化前 | 优化后 |
|------|--------|--------|
| 图构建 + HAN 推理 | 每步调用（2048 次/rollout） | 每 10 步调用一次（~205 次/rollout） |
| 轻量级路径 | 调用完整 `extract_node_features()`（含卫星特征+可见性） | 内联提取 task/rvt 特征（仅访问 user 对象） |

#### 优化 5: 特征提取向量化 (`src/graph/features.py`)

| 项目 | 优化前 | 优化后 |
|------|--------|--------|
| `_extract_satellite_features()` | 逐卫星循环提取位置/速度 | 直接切片 `env.constellation._all_pos_ecef` |
| `_extract_inter_satellite_edges()` | 逐边循环计算距离 | numpy 批量 `all_pos[src] - all_pos[dst]` |

#### 优化 6: 推理时 eval 模式 (`scripts/train.py` + `src/algorithm/mappo.py`)

```python
# collect_rollouts() 开始时关闭 dropout
self.mappo.actor.eval()
self.mappo.critic.eval()
self.han_encoder.eval()

# mappo.update() 开始时恢复 dropout
self.actor.train()
self.critic.train()
```

### 7.3 性能对比

| 指标 | 优化前 (CPU) | 优化后 (GPU) | 加速比 |
|------|-------------|-------------|--------|
| 512 步 rollout | 113.3s | 5.1s | **22x** |
| 2048 步训练 (含 PPO update) | ~450s (估) | 58.0s | **~8x** |
| FPS | ~4.5 | 35.3 | **8x** |

### 7.4 当前瓶颈分布 (GPU, 512 步 profiling)

| 组件 | 耗时 | 占比 |
|------|------|------|
| `mappo.act()` GPU 推理 | 2.6s | 51% |
| `env.step()` 环境模拟 | 1.7s | 34% |
| `_encode_graph_state()` | 0.6s | 12% |
| 其他 | 0.2s | 3% |

瓶颈已从环境计算转移到 GPU 模型推理，属于正常分布。

### 7.5 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `src/environment/constellation.py` | `_init_vectorized_arrays()` + 向量化 `_update_all_positions()` |
| `src/environment/gym_env.py` | `_precompute_user_geometry()` + `_compute_visibility_batch()` + 缓存 bug 修复 |
| `src/graph/features.py` | 向量化 `_extract_satellite_features()` + `_extract_inter_satellite_edges()` |
| `scripts/train.py` | HAN 缓存 + 内联特征提取 + eval/train 模式切换 |
| `src/algorithm/mappo.py` | `update()` 中添加 `self.actor.train()` / `self.critic.train()` |
