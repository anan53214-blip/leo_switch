# 能耗单目标训练结果分析 (Energy-Only v1)

> 实验名称: `han_mappo_energy_only` | 种子: 42 | 设备: CUDA  
> 场景: 10 用户, 66 卫星, Episode 长度 1000  
> 目标: 单次实验仅优化能耗最小化

---

## 1. 总体结论

本轮 `energy-only` 训练整体表现为“**有效学习但中后期存在震荡**”：

- `-ΔEnergy Reward` 明显提升（从约 `-6100` 改善到约 `-1700`）；
- Critic/Actor 损失快速收敛到低位，训练过程未出现失稳爆炸；
- 但能耗曲线在中后期出现反弹，任务完成率长期维持在 `45%~50%` 区间，说明存在“省电与服务质量”的权衡冲突。

结论：当前版本已具备能耗优化能力，但尚未实现“稳定低能耗 + 高服务质量”的双优平衡。

---

## 2. 各指标详细分析

### 2.1 Reward Convergence（奖励收敛）

| 指标 | 观测结果 |
|------|----------|
| 初始目标奖励 | 约 `-6100` |
| 中后期目标奖励 | 提升到约 `-1700` |
| 趋势 | 前期快速改善，后期平台化并轻微回落 |

说明：reward 改善显著，表明策略学到了降低能耗的行为；后段回落对应策略更新震荡。

### 2.2 Critic / Actor Loss

| 指标 | 早期 | 中后期 | 状态 |
|------|------|--------|------|
| Critic Loss | 约 `0.5` | 迅速降至 `0.01` 附近 | 收敛良好 ✅ |
| Actor Loss | 小幅波动 | 维持接近 0 | 正常 ✅ |

说明：损失曲线健康，未见显著优化器失稳现象。

### 2.3 Entropy & KL（策略更新强度）

从可视化仪表盘可见：

- Entropy 持续下降（策略逐步确定化）；
- KL 散度整体上升并伴随中后期波动（局部更新幅度增大）。

解释：训练进入“由探索转利用”阶段是正常现象；但 KL 中后段抬升对应能耗曲线反弹，提示策略存在阶段性漂移。

### 2.4 Handover & Task Rate

| 指标 | 观测 |
|------|------|
| Handover Success | 约 95%（稳定） |
| Task Completion | 长期约 `45%~50%` |

说明：在能耗单目标下，任务完成率未同步改善，表明当前约束惩罚对服务质量牵引仍偏弱。

### 2.5 Energy & Delay（目标相关）

| 指标 | 现象 |
|------|------|
| Energy (J) | 总体下降，但中后期有反弹与震荡 |
| Delay (ms) | 与能耗协同波动，未单调改善 |

关键观察：能耗下降后再回升，多发生于 KL 波动抬升区间，符合“策略局部漂移导致短期退化”的特征。

### 2.6 Eval Reward（泛化能力）

| 阶段 | 观测 |
|------|------|
| 早期 | 评估奖励波动较大（出现明显低谷） |
| 中后期 | 回升并稳定在较优区间（约 `-1000` 附近） |

结论：泛化能力较早期有改善，但稳定性仍有提升空间。

---

## 3. 根本问题归纳

1. **目标权衡冲突**：主目标为能耗最小，服务质量（完成率/时延）仅通过惩罚项约束，导致 completion 提升不足。  
2. **中后期策略漂移**：Entropy 持续下降 + KL 波动上升，易触发能耗反弹段。  
3. **惩罚系数偏保守**：默认 QoS/超时约束权重可能不足以持续拉住服务指标。  
4. **多目标耦合强**：能耗、时延、任务完成率在队列与可见性机制下强耦合，单目标优化天然会产生边际退化风险。

---

## 4. 本次改动优化（已落地）

> 本节记录“这一次”已实施的代码优化，用于支撑能耗最小化实验并抑制服务质量退化。

### 4.1 能耗目标约束强化（P0）

**文件**: `scripts/train_energy_only.py`

将能耗实验奖励从：

- `objective_reward = -Δtotal_energy`

升级为约束化能耗成本：

- `objective_reward = -(Δenergy + λ1·Δunmet_tasks + λ2·Δdeadline_violations + λ3·Δfailed_handovers)`

新增参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--qos_unmet_task_penalty` | `0.5` | 每个未完成任务（QoS缺口）的能耗等价惩罚 |
| `--delay_violation_penalty` | `1.0` | 每个超时任务的能耗等价惩罚 |
| `--failed_handover_penalty` | `0.5` | 每个失败切换的能耗等价惩罚 |

预期效果：避免策略走向“纯省电但服务退化”的极端解。

### 4.2 Critic 稳定化（P1）

**文件**: `src/algorithm/mappo.py`, `scripts/train.py`

新增与启用：

- `value_loss_type = huber`
- `normalize_returns = True`
- `clip_range_vf = 0.2`

预期效果：缓解 critic 前期尖峰与训练抖动，降低中后期策略漂移概率。

### 4.3 训练提速（P1）

**文件**: `scripts/train.py`, `scripts/train_energy_only.py`

新增与调整：

- `graph_update_interval` 参数化（默认 `20`）
- `eval_episodes` 默认降低到 `2`

预期效果：减少图重建与评估开销，加快实验迭代速度。

---

## 5. 关键代码位置

- 能耗单目标环境：`scripts/train_energy_only.py` (`EnergyOnlyEnv.step`)  
- 通用训练配置：`scripts/train.py` (`TrainConfig`, `parse_args`)  
- PPO / Critic 更新：`src/algorithm/mappo.py` (`MAPPO.update`)

---

## 6. 下一次复现实验建议（Energy-Only）

建议先跑一组平衡配置（保证能耗继续下降并抑制 completion 退化）：

```powershell
C:/Users/19704/.conda/envs/satellite.env/python.exe scripts/train_energy_only.py --num_users 10 --total_timesteps 500000 --eval_interval 20000 --eval_episodes 2 --graph_update_interval 20 --qos_unmet_task_penalty 1.5 --delay_violation_penalty 2.0 --failed_handover_penalty 1.0
```

若能耗已稳定但任务完成率仍低，可继续增强 QoS 约束：

```powershell
C:/Users/19704/.conda/envs/satellite.env/python.exe scripts/train_energy_only.py --num_users 10 --total_timesteps 500000 --qos_unmet_task_penalty 2.0 --delay_violation_penalty 2.5 --failed_handover_penalty 1.5
```

---

## 7. 验证状态

- `train_energy_only.py` 的约束惩罚参数与逻辑已完成语法/入口检查。  
- 本文档为 `energy-only` 的 v1 结果记录与本轮优化基线。  
- 下一步建议基于上述两组参数完成对照实验并产出 v2 结果对比。
