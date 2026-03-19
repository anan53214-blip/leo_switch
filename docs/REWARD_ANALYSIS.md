# Reward 计算详细分析

本文档梳理训练数据中 reward 值的完整计算逻辑，包括所有得分项和扣分项，以及对应的代码位置。

---

## 1. 奖励权重配置

定义在 [`EnvConfig`](src/environment/gym_env.py:31) 中：

```python
reward_delay_weight: float = 0.4     # 时延奖励权重
reward_energy_weight: float = 0.3    # 能耗奖励权重
reward_handover_weight: float = 0.2  # 切换奖励权重
reward_qos_weight: float = 0.1      # QoS奖励权重
```

---

## 2. 总奖励计算流程

入口在 [`step()`](src/environment/gym_env.py:325)，每个时间步：

1. 对每个用户调用 [`_execute_user_action()`](src/environment/gym_env.py:392) 计算即时奖励
2. 加上 `pending_rewards`（上一步 MEC 队列完成的延迟奖励）
3. 所有用户奖励取均值作为全局奖励

```python
total_reward = np.mean(user_rewards)  # gym_env.py:364
```

---

## 3. 得分项（正奖励）

### 3.1 切换成功 — 目标卫星质量奖励

位置：[`_execute_handover()`](src/environment/gym_env.py:494)

| 得分项 | 公式 | 最大值 |
|--------|------|--------|
| 仰角奖励 | `+0.1 × (elevation / 90.0)` | +0.1 |
| RVT奖励 | `+0.1 × (rvt_seconds / 600.0)` | +0.1 |

```python
reward += 0.1 * (target_sat.elevation_deg / 90.0)   # 仰角越高越好
reward += 0.1 * (target_sat.rvt_seconds / 600.0)    # RVT越长越好
```

### 3.2 任务入队成功奖励

位置：[`_execute_offloading()`](src/environment/gym_env.py:592)

| 得分项 | 值 |
|--------|----|
| 成功入队 MEC | +0.05 |

```python
reward += 0.05  # 入队成功
```

### 3.3 完全本地执行且满足 deadline

位置：[`_execute_offloading()`](src/environment/gym_env.py:626)

| 得分项 | 公式 |
|--------|------|
| 时延奖励 | `+0.4 × max(1.0 - local_delay/max_delay, 0)` |
| 能耗奖励 | `+0.3 × clip(1.0 - local_energy/10.0, 0, 1)` |
| QoS奖励 | `+0.1 × 1.0` |

```python
delay_reward = 1.0 - (local_delay / task.max_delay)
energy_reward = 1.0 - (local_energy / max_energy)
reward += self.config.reward_delay_weight * max(delay_reward, 0.0)      # +0.4×delay
reward += self.config.reward_energy_weight * np.clip(energy_reward, 0, 1) # +0.3×energy
reward += self.config.reward_qos_weight * 1.0                            # +0.1
```

### 3.4 队列满退化本地执行且满足 deadline

位置：[`_execute_offloading()`](src/environment/gym_env.py:609)

| 得分项 | 公式 |
|--------|------|
| 时延奖励 | `+0.4 × max(1.0 - local_delay/max_delay, 0)` |

```python
delay_reward = 1.0 - (local_delay / task.max_delay)
reward += self.config.reward_delay_weight * max(delay_reward, 0.0)
```

### 3.5 MEC 队列完成任务（延迟发放）

位置：[`_update_environment()`](src/environment/gym_env.py:655)

当卸载到卫星的任务在 MEC 队列中处理完成且满足 deadline 时：

| 得分项 | 公式 |
|--------|------|
| 时延奖励 | `+0.4 × max(1.0 - total_delay/max_delay, 0)` |
| 能耗奖励 | `+0.3 × max(1.0 - upload_energy/10.0, 0)` |
| QoS奖励 | `+0.1 × 1.0` |

```python
delay_reward = 1.0 - (task_info['total_delay'] / task_info['max_delay'])
energy_reward = 1.0 - min(upload_energy / 10.0, 1.0)
task_reward = self.config.reward_delay_weight * delay_reward
task_reward += self.config.reward_energy_weight * max(energy_reward, 0.0)
task_reward += self.config.reward_qos_weight * 1.0
```

该奖励通过 `pending_rewards` 在下一步发放给对应用户。

---

## 4. 扣分项（负奖励/惩罚）

### 4.1 切换时延惩罚

位置：[`_execute_handover()`](src/environment/gym_env.py:462)

每次执行切换都会扣分（无论成功与否）：

| 扣分项 | 公式 | 默认值 |
|--------|------|--------|
| 切换时延 | `-handover_delay_sec / 5.0` | -0.16 |

```python
delay_penalty = self.config.handover_delay_sec / 5.0  # 0.8/5.0 = 0.16
reward -= delay_penalty
```

### 4.2 任务迁移惩罚

位置：[`_execute_handover()`](src/environment/gym_env.py:488)

切换成功后，旧卫星上的排队任务需要迁移：

| 扣分项 | 公式 |
|--------|------|
| 迁移成功的任务 | `-0.05 × 迁移数量` |
| 迁移失败的任务 | `-0.1 × 失败数量` |

```python
reward -= 0.05 * migration_result['migrated']
reward -= 0.1 * migration_result['failed']
```

### 4.3 切换失败惩罚

位置：[`_execute_handover()`](src/environment/gym_env.py:499)

| 扣分项 | 值 |
|--------|----|
| 切换失败 | -0.5 |

```python
reward -= 0.5  # 切换失败惩罚
```

### 4.4 用户阻塞惩罚

位置：[`_execute_user_action()`](src/environment/gym_env.py:429)

当前卫星不可见且无其他可见卫星时：

| 扣分项 | 值 |
|--------|----|
| 阻塞（无可用卫星） | -1.0 |

```python
reward -= 1.0  # 阻塞惩罚
```

### 4.5 无连接惩罚

位置：[`_execute_offloading()`](src/environment/gym_env.py:531)

卸载时用户未连接卫星或卫星不可见：

| 扣分项 | 值 |
|--------|----|
| 无连接/不可见 | -0.5 |

```python
return -0.5  # 无连接惩罚
```

### 4.6 MEC 队列满惩罚

位置：[`_execute_offloading()`](src/environment/gym_env.py:606)

| 扣分项 | 值 |
|--------|----|
| 队列满无法卸载 | -0.3 |

```python
reward -= 0.3  # 队列满导致无法卸载
```

### 4.7 Deadline 违约惩罚

出现在三处：

1. 完全本地执行超时 — [`_execute_offloading()`](src/environment/gym_env.py:637)
2. 队列满退化本地执行超时 — [`_execute_offloading()`](src/environment/gym_env.py:615)
3. MEC 队列任务超时 — [`_update_environment()`](src/environment/gym_env.py:673)

| 扣分项 | 值 |
|--------|----|
| 任务超时未完成 | -0.5 |

```python
reward -= 0.5  # deadline 违约
# 或
task_reward = -0.5  # MEC 队列超时
```

---

## 5. 奖励汇总表

| 类别 | 事件 | 奖励值 | 代码位置 |
|------|------|--------|----------|
| ✅ 得分 | 切换到高仰角卫星 | +0.0~0.1 | gym_env.py:494 |
| ✅ 得分 | 切换到长RVT卫星 | +0.0~0.1 | gym_env.py:495 |
| ✅ 得分 | 任务成功入队MEC | +0.05 | gym_env.py:592 |
| ✅ 得分 | 本地执行满足deadline（时延） | +0.0~0.4 | gym_env.py:632 |
| ✅ 得分 | 本地执行满足deadline（能耗） | +0.0~0.3 | gym_env.py:633 |
| ✅ 得分 | 本地执行满足deadline（QoS） | +0.1 | gym_env.py:634 |
| ✅ 得分 | MEC完成任务满足deadline | +0.0~0.8 | gym_env.py:668-670 |
| ❌ 扣分 | 每次切换的时延开销 | -0.16 | gym_env.py:462 |
| ❌ 扣分 | 任务迁移（每个成功） | -0.05/个 | gym_env.py:488 |
| ❌ 扣分 | 任务迁移（每个失败） | -0.1/个 | gym_env.py:489 |
| ❌ 扣分 | 切换失败 | -0.5 | gym_env.py:499 |
| ❌ 扣分 | 用户阻塞（无卫星） | -1.0 | gym_env.py:429 |
| ❌ 扣分 | 无连接/卫星不可见 | -0.5 | gym_env.py:531 |
| ❌ 扣分 | MEC队列满 | -0.3 | gym_env.py:606 |
| ❌ 扣分 | Deadline违约 | -0.5 | gym_env.py:615,637,673 |

---

## 6. 设计意图

奖励函数引导智能体学习以下策略：

1. **减少不必要切换** — 每次切换都有固定时延惩罚 (-0.16)
2. **选择高质量卫星** — 高仰角和长 RVT 的卫星获得正奖励
3. **合理卸载决策** — 成功卸载到 MEC 有入队奖励，队列满则惩罚
4. **满足时延约束** — deadline 内完成任务获得与剩余时间成正比的奖励
5. **降低能耗** — 能耗越低，能耗奖励越高
6. **保持连接** — 阻塞状态受到最重惩罚 (-1.0)
