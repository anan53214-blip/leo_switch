# 时延单目标训练结果分析 (Delay-Only v1)

> 实验名称: `han_mappo_delay_only` | 种子: 42 | 设备: CUDA  
> 场景: 10 用户, 66 卫星, Episode 长度 1000  
> 目标: 单次实验仅优化时延最小化

---

## 1. 总体结论

本轮 `delay-only` 训练呈现“**前期改善，后期退化**”的典型现象：

- 训练曲线中 `EpReward` 从约 `-23.5k` 改善到 `-18k` 区间；
- 但评估奖励出现明显恶化（中后期回落），泛化稳定性不足；
- 图5中时延在中段达到较低值后反弹，能耗反而继续下降，说明策略存在目标漂移风险。

结论：当前版本尚未稳定收敛到“持续压低时延”的策略，需要对目标约束和训练稳定性继续优化。

---

## 2. 各指标详细分析

### 2.1 Reward Convergence（奖励收敛）

| 指标 | 观测结果 |
|------|----------|
| 早期训练奖励 | 约 `-23.5k` |
| 中后期训练奖励 | 改善至约 `-18.3k` |
| 趋势 | 非单调，先改善后波动 |

说明：训练 reward 改善不代表策略稳定收敛，需结合评估曲线判断。

### 2.2 Critic / Actor Loss

| 指标 | 早期 | 中期 | 状态 |
|------|------|------|------|
| Critic Loss | `1e5` 量级（首轮） | 下降到 `3e3~4e3` 区间 | 有下降，但仍偏高 |
| Actor Loss | 约 `-1e-3` 量级 | 同量级波动 | 正常小幅更新 |

说明：Critic Loss 大幅回落是正向信号，但量级仍高、波动仍明显。

### 2.3 Entropy & KL（策略更新强度）

从可视化仪表盘可见：

- Entropy 持续下降（策略更确定）；
- KL 持续上升（新旧策略差异增大）。

解释：训练从探索转向利用是正常的，但若 KL 持续上行且伴随评估变差，通常表示更新方向开始偏离最优策略区域。

### 2.4 Handover & Task Rate

| 指标 | 观测 |
|------|------|
| Handover Success | 约 95%（基本稳定） |
| Task Completion | 约 49% 下降到约 43% |

说明：任务完成率下降与“时延后期反弹”一致，表明策略在服务质量层面出现退化。

### 2.5 Delay & Energy（目标相关）

| 指标 | 现象 |
|------|------|
| Delay (ms) | 中段最低，后段反弹 |
| Energy (J) | 后段继续下降 |

关键观察：在 `delay-only` 目标下出现“时延不再下降、能耗继续下降”，说明原奖励口径存在可被利用的空间（策略可能通过牺牲任务质量来间接改善某些统计项）。

### 2.6 Eval Reward（泛化能力）

评估奖励表现为明显波动与退化（例如从约 `-11k` 附近恶化到约 `-19k ~ -23k` 区间）。

结论：训练指标改善并未稳定转化为评估收益，当前策略泛化稳定性不足。

---

## 3. 根本问题归纳

1. **目标约束不够严格**：原 `delay-only` 仅惩罚 `Δtotal_delay`，未显式惩罚超时与失败切换。  
2. **策略后期漂移**：Entropy 下降 + KL 上升 + Eval 退化，说明更新后期偏离有效区域。  
3. **价值学习尺度偏大**：Critic Loss 前期异常高，导致训练早期不稳定。  
4. **评估开销偏大**：评估周期内耗时长，影响训练迭代效率与调参反馈速度。

---

## 4. 本次改动优化（已落地）

> 本节记录“这一次”已实施的代码优化，用于专门支撑时延最小化目标。

### 4.1 时延目标强化（P0）

**文件**: `scripts/train_delay_only.py`

将时延实验奖励从：

- `objective_reward = -Δtotal_delay`

升级为时延等价成本：

- `objective_reward = -(Δdelay + λ1·Δdeadline_violations + λ2·Δfailed_handovers)`

新增参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--delay_violation_penalty` | `5.0` | 每个超时任务的时延等价惩罚 |
| `--failed_handover_penalty` | `1.0` | 每个失败切换的时延等价惩罚 |

预期效果：防止策略通过牺牲任务完成质量“钻目标空子”，让优化更贴近真实时延质量。

### 4.2 Critic 稳定化（P1）

**文件**: `src/algorithm/mappo.py`, `scripts/train.py`

新增与启用：

- `value_loss_type = huber`
- `normalize_returns = True`
- `clip_range_vf = 0.2`

预期效果：缓解 Critic Loss 前期尖峰，提升中后期训练稳定性。

### 4.3 训练提速（P1）

**文件**: `scripts/train.py`, `scripts/train_delay_only.py`

新增与调整：

- `graph_update_interval` 参数化（默认提高到 `20`）
- `eval_episodes` 默认降低到 `2`（delay-only 脚本）

预期效果：降低图重建与评估开销，加快单次实验反馈。

---

## 5. 关键代码位置

- 时延单目标环境：`scripts/train_delay_only.py` (`DelayOnlyEnv.step`)  
- 通用训练配置：`scripts/train.py` (`TrainConfig`, `parse_args`)  
- PPO / Critic 更新：`src/algorithm/mappo.py` (`MAPPO.update`)

---

## 6. 下一次复现实验建议（Delay-Only）

建议先跑一组稳定性验证配置：

```powershell
C:/Users/19704/.conda/envs/satellite.env/python.exe scripts/train_delay_only.py --num_users 10 --total_timesteps 500000 --eval_interval 20000 --eval_episodes 2 --graph_update_interval 20 --delay_violation_penalty 5.0 --failed_handover_penalty 1.0
```

若仍出现“时延反弹 + 评估退化”，可提高惩罚：

```powershell
C:/Users/19704/.conda/envs/satellite.env/python.exe scripts/train_delay_only.py --num_users 10 --total_timesteps 500000 --delay_violation_penalty 8.0 --failed_handover_penalty 2.0
```

---

## 7. 验证状态

- `train_delay_only.py` 新增参数与逻辑已完成语法/入口检查。  
- 本文档为 `delay-only` 的 v1 结果记录与本轮优化基线。  
- 下一步需基于优化后重新完整训练并更新 v2 结果对照表。
