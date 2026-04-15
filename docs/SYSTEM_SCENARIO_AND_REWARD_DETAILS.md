# 系统场景与 Reward 机制代码级说明

这份文档基于当前仓库中的实际实现整理，核心依据是：

- `src/environment/gym_env.py`
- `src/environment/mec.py`
- `src/environment/task.py`
- `src/environment/user.py`
- `src/environment/channel.py`
- `src/environment/constellation.py`
- `src/environment/visibility.py`
- `src/graph/features.py`
- `src/graph/builder.py`
- `scripts/train.py`

本文以“代码实际运行行为”为准，不以论文表述或注释中的理想化描述为准。仓库里曾有过 `config/constellation.yaml`，但它没有接入运行时流程，现已移除。

## 1. 系统到底设计了一个什么场景

### 1.1 场景总览

这是一个 **LEO 卫星网络中的联合切换与任务卸载多智能体强化学习场景**。

- 地面有一批固定位置用户。
- 空中有一组 Walker 星座 LEO 卫星，每颗卫星挂载一个 MEC 服务器。
- 每个时间步，用户都可能产生计算任务。
- 每个用户智能体要同时做两类决策：
  - 是否切换到别的可见卫星。
  - 当前任务有多少比例卸载到卫星 MEC，多少比例留在本地算。

系统的目标不是只追求一个指标，而是综合考虑：

- 任务时延
- 本地与传输能耗
- 切换质量
- 负载均衡
- QoS / 是否满足时延约束

### 1.2 运行时默认场景参数

以环境代码 `EnvConfig` 和 `MECConfig` 的默认值为准，系统默认场景是：

#### 卫星星座

- 轨道面数：`6`
- 每轨道面卫星数：`11`
- 总卫星数：`66`
- 轨道高度：`550 km`
- 轨道倾角：`53 deg`
- Walker 相位因子：`1`
- 星座起始时间：`2026-01-15 00:00:00`

#### 地面用户

- 环境默认用户数：`10`
- 用户分布中心：`(39.9, 116.4)`，即北京附近
- 用户分布半径：`5.0 deg`
- 用户位置在圆形区域内随机均匀生成
- 用户默认不移动，位置在整个 episode 内固定

这里要特别注意：

- `5.0 deg` 不是 `5 km`，也不是 `100 km`。
- 仓库早期曾有一个 `config/constellation.yaml`，其中地面区域半径写成 `100 km`，但环境运行时并没有读取它。
- 因此 **实际代码场景** 以 `EnvConfig.user_radius_deg = 5.0` 为准。

#### 链路与可见性

- 最小可见仰角阈值：`10 deg`
- 用户最多保留 `10` 个可见候选卫星
- 候选卫星排序规则：
  - 先按 `RVT` 从大到小
  - 再按仰角从大到小
  - 再按距离从小到大

#### 时间推进

- 环境时间步长：`1 s`
- 环境默认 episode 长度：`3600` 步，也就是 `1 小时`
- `terminated` 始终为 `False`
- episode 结束条件实际上只有 `truncated = (current_step >= max_steps)`

#### MEC 服务器

每颗卫星挂一个 MEC 服务端，默认参数：

- 单核频率：`5 GHz`
- 核数：`4`
- 总计算能力：`20 GHz`
- 队列最大长度：`20`
- 用户本地 CPU 频率：`0.5 GHz`
- 返回结果数据量比例：`0.1`

### 1.3 训练脚本层面的默认覆盖

环境默认值和训练脚本默认值不是完全一致的。

`scripts/train.py` 的 `TrainConfig` 默认会覆盖一部分环境参数。当前默认配置已统一到 `results/full_train_delay_focus` 这次实验：

- `exp_name = han_mappo_delay_focus_fast`
- `num_users = 10`
- `max_steps = 2000`
- `total_timesteps = 1_000_000`
- `n_steps = 2048`
- `n_epochs = 4`
- `batch_size = 256`
- `eval_interval = 100_000`
- `eval_episodes = 3`
- `graph_update_interval = 100`
- `save_interval = 200_000`
- `save_path = results/full_train_delay_focus`
- reward 权重默认是：
  - `delay = 1.4`
  - `energy = 0.4`
  - `handover = 0.3`
  - `load_balance = 0.1`
  - `qos = 0.4`

但训练脚本没有显式传入的环境参数，仍然沿用环境默认值，例如：

- `task_arrival_prob = 0.45`
- `min_elevation_deg = 10`
- `handover_delay_sec = 0.6`
- `reward_enqueue_bonus = 0.02`
- `reward_invalid_action_penalty = 0.5`
- `reward_blocked_penalty = 1.0`
- `reward_queue_full_penalty = 0.3`
- `reward_failed_handover_penalty = 0.6`
- `reward_deadline_penalty = 1.0`
- `reward_energy_reference = 10.0`

所以如果你在分析某次实验，不能只看环境默认值，也不能只看训练脚本默认值，必须看两层配置合并后的实际值。

## 2. 场景里的实体分别在做什么

### 2.1 用户

每个用户都是一个智能体，具备以下核心状态：

- 地理位置
- 当前连接状态
- 当前服务卫星 ID
- 切换统计
- 当前前台任务 `user_tasks[user_id]`

用户状态只有四种：

- `IDLE`
- `CONNECTED`
- `HANDOVER`
- `BLOCKED`

代码里的实际含义是：

- `CONNECTED`：已经连上某颗卫星，可以执行卸载。
- `HANDOVER`：切换过程中的临时状态。
- `BLOCKED`：当前没有可用服务卫星。
- `IDLE`：初始化或手动断开后的空闲态。

### 2.2 卫星

每颗卫星有三层角色：

- 轨道节点：在 Walker 星座中按轨道动力学传播位置。
- 通信节点：与用户形成上行/下行链路。
- 计算节点：承载一个 MEC 队列和若干连接用户。

### 2.3 链路

系统显式建模了星地链路的以下物理量：

- 距离
- 仰角
- SNR
- 信道容量
- 数据率
- 传播时延

默认物理参数包括：

- 载波频率：`20 GHz`
- 带宽：`50 MHz`
- 卫星发射功率：`40 dBm`
- 用户发射功率：`33 dBm`

信道容量使用香农公式：

```text
C = B * log2(1 + SNR)
```

传输时延使用：

```text
T_trans = D / R + T_prop
T_prop = d / c
```

### 2.4 MEC 计算层

MEC 层负责两件事：

- 接收用户卸载来的任务分片
- 在卫星侧排队和处理这些任务

每个 MECServer 维护：

- `task_queue`
- `connected_users`
- `available_freq_ghz`
- `total_capacity_ghz`

## 3. 用户到底在做什么任务

### 3.1 任务类型

任务分三类：

- `LIGHT`
- `MEDIUM`
- `HEAVY`

默认生成概率：

- 轻任务：`30%`
- 中任务：`50%`
- 重任务：`20%`

### 3.2 每类任务参数范围

#### 轻任务

- 数据量：`0.5e6 ~ 2e6 bits`
- 计算量：`0.1e9 ~ 0.5e9 cycles`
- 最大允许时延：`0.5 ~ 2.0 s`

#### 中任务

- 数据量：`2e6 ~ 10e6 bits`
- 计算量：`0.5e9 ~ 2e9 cycles`
- 最大允许时延：`1.0 ~ 5.0 s`

#### 重任务

- 数据量：`10e6 ~ 50e6 bits`
- 计算量：`2e9 ~ 10e9 cycles`
- 最大允许时延：`2.0 ~ 10.0 s`

### 3.3 任务什么时候生成

每个时间步开始时，环境先尝试生成新任务。

生成条件非常明确：

- 该用户当前 `user_tasks[user_id] is None`
- 该用户状态是 `CONNECTED`
- 以概率 `task_arrival_prob = 0.45` 生成

这意味着：

- 只有连着卫星的用户才会产生新任务。
- 每个用户前台最多只有一个“待决策任务”。
- 但这不代表该用户系统内只有一个任务，因为旧任务的卸载分片可能已经进入卫星队列，还没完成。

换句话说：

- `user_tasks` 只约束“前台待决策任务”
- MEC 队列里同一用户可以积压多个历史卸载任务

## 4. 一个任务是怎么被处理的

### 4.1 任务生成后先放在哪里

新任务一旦生成：

- 放入 `self.user_tasks[user_id]`
- 同时调用 `TaskManager.add_task(task)`
- `stats['total_tasks'] += 1`

但要注意：

- `TaskManager` 在当前实现里主要只用来登记任务数量
- `start_task / complete_task / fail_task` 这些生命周期函数几乎没有被环境主流程真正使用
- 真正的运行态主要由 `user_tasks` 和各个 MEC server 的 `task_queue` 决定

### 4.2 卸载比例怎么定义

若任务总数据量为 `D`、总计算量为 `C`、卸载比例为 `o`，则：

```text
local_ratio = 1 - o
local_cycles = (1 - o) * C
offload_cycles = o * C
offload_data_bits = o * D
```

也就是说：

- 本地部分按计算比例切分
- 卫星卸载部分的数据量和计算量都按同一个比例切分

### 4.3 本地部分怎么处理

本地部分是“立即解析式计算”，不会进入队列。

本地时延：

```text
T_local = C_local / f_user
```

本地能耗：

```text
E_local = kappa * C_local * f_user^2
```

默认本地 CPU 频率：

- `f_user = 0.5 GHz`

### 4.4 卫星卸载部分怎么处理

若 `offload_ratio > 0`：

1. 先根据当前服务卫星链路算上传/下载时延
2. 再算上传能耗
3. 然后尝试把任务分片塞入当前服务卫星 MEC 队列

上传/下载时延：

```text
upload_delay = T_upload(offload_data_bits)
download_delay = T_download(result_bits)
result_bits = offload_data_bits * 0.1
```

上传能耗：

```text
E_upload = P_tx * T_upload
```

### 4.5 入队成功后发生什么

如果队列没满：

- 该分片进入卫星的 `task_queue`
- 环境立刻记录：
  - `task.offload_ratio`
  - `task.local_delay`
  - `task.local_energy`
  - `task.transmission_energy`
  - `self._offload_task_meta[(user_id, task_id)] = {'local_delay', 'local_energy'}`
- 当前步只给一个“成功入队 bonus”
- 真正的任务主 reward 不在当前步发放，而是等卫星侧完成或超时后，放入 `pending_rewards[user_id]`，在下一步加回该用户奖励

这点很关键：

- **本地执行的任务 reward 是即时给的**
- **排队卸载的任务 reward 是延迟给的**

### 4.6 入队失败会怎样

如果队列已满：

- 卸载失败
- 系统不会简单丢弃任务
- 而是退化为“把原本想卸载的部分重新本地执行”

具体行为是：

```text
fallback_cycles = offload_cycles
fallback_delay = local_delay(fallback_cycles)
fallback_energy = local_energy(fallback_cycles)
```

然后：

- 本地总时延变成“本来本地部分 + fallback 本地部分”
- 本地总能耗也随之叠加
- 同时吃一个 `queue_full` 惩罚
- 然后立刻按“全本地”逻辑计算任务 reward

这里还有一个代码级细节：

- 系统是在尝试入队前就先算好了上传时延和上传能耗
- 所以即使最终队列满了，reward 中仍然会把这部分 `upload_energy` 计入总能耗
- 这等于把“无效上传尝试的代价”也算进去了

### 4.7 队列里怎么排队

严格按代码实现，MEC 队列不是严格 FCFS，而是更接近 **有限容量 + 处理器共享** 模型。

代码注释写了 FCFS，但实际行为是：

- 每个时间步把所有 `queued` 和 `processing` 任务都视为活跃任务
- 卫星总算力平均分给所有活跃任务
- 每个活跃任务在同一个时间步都获得相同的 cycles 处理额度

即：

```text
freq_per_task = total_capacity / num_active_tasks
cycles_per_task = freq_per_task * 1e9 * time_step
```

因此代码实际语义是：

- 队列容量有限
- 入队顺序会影响 `arrival_time`
- 但开始处理后不是严格一个做完再做下一个
- 而是所有活跃任务并行分享算力

### 4.8 任务什么时候算完成

对卸载到卫星的任务，如果在某一步 `remaining_cycles <= 0`：

```text
queue_wait = start_processing_time - arrival_time
processing_time = current_time - start_processing_time + time_step
total_delay = upload_delay + queue_wait + processing_time + download_delay
```

然后：

- 任务从队列移除
- 标记为 completed
- 放入 `completed_tasks`

### 4.9 任务什么时候算超时

如果任务还没算完，但：

```text
elapsed = current_time - arrival_time + upload_delay
elapsed > max_delay
```

则：

- 标记为 timeout
- 从队列移除
- `deadline_met = False`
- `total_delay = elapsed`

这里也要注意一个实现细节：

- 超时分支使用的是“截至当前的已消耗时间”
- 不是“如果继续执行到完成的总时延”
- 所以超时任务的 `total_delay` 更像“被判死刑时刻的累计耗时”

### 4.10 分片任务最终总时延如何定义

对拆分任务，环境最终采用：

```text
T_total = max(T_local, T_offloaded)
```

也就是本地和卸载部分并行执行，最终由较慢的一边决定整个任务完成时间。

这是在 `_update_environment()` 中真正用于 reward 的总时延。

## 5. 卫星切换是怎么工作的

### 5.1 初始接入

环境 reset 后，每个用户会自动连接到：

- 当前可见卫星中 **仰角最高** 的那颗

如果一个可见卫星都没有：

- 用户直接进入 `BLOCKED`

### 5.2 每步的切换动作语义

每个用户动作由两个分量组成：

- `handover_action`
- `offload_ratio`

其中：

- `handover_action = 0` 表示“不主动切换”
- `handover_action = 1..K` 表示切换到第 `k` 个候选可见卫星

这里的候选编号对应的是排序后的候选列表，不是卫星全局 ID。

### 5.3 不切换时系统会不会自动切换

会。

如果 `handover_action = 0`，但当前服务卫星已经不可见：

- 若仍有其他可见卫星，环境会自动切到仰角最高的那颗
- 若没有任何可见卫星，用户进入 `BLOCKED` 并吃 blocked penalty

因此：

- `action = 0` 不等于“绝对保持原链路”
- 它只表示“用户不主动指定目标”

### 5.4 正向切换时怎么选目标

如果 `handover_action > 0`：

- 环境取排序后候选列表的第 `handover_action - 1` 颗卫星作为目标

这还有一个实现上的严格细节：

- 候选列表没有显式排除“当前正在服务的卫星”
- 所以理论上正动作也可能选中当前卫星本身
- 代码会把这种情况也当成一次 handover 流程来执行

### 5.5 切换成功概率怎么计算

环境没有把切换成功率写死，而是按目标卫星质量动态算：

```text
elevation_score = clip((elev - min_elev) / (90 - min_elev), 0, 1)
rvt_score       = clip(rvt / (2 * rvt_threshold), 0, 1)
snr_score       = clip((snr_db + 5) / 30, 0, 1)
load_headroom   = 1 - utilization
queue_headroom  = 1 - queue_ratio
migration_penalty = clip(migration_load / 5, 0, 1)
```

然后：

```text
success_prob =
    0.35
  + 0.20 * elevation_score
  + 0.15 * rvt_score
  + 0.15 * snr_score
  + 0.10 * load_headroom
  + 0.10 * queue_headroom
  - 0.10 * migration_penalty
```

最后再裁剪到：

```text
success_prob in [0.1, 0.995]
```

这说明什么情况下更容易切换成功：

- 仰角更高
- RVT 更长
- SNR 更好
- 目标卫星更空闲
- 目标队列更短
- 待迁移历史任务更少

### 5.6 切换成功后会发生什么

切换成功后：

- 用户状态设为 `CONNECTED`
- 用户连接从旧卫星迁到新卫星
- 旧卫星上该用户的排队任务会尝试迁移到新卫星

迁移规则：

- 找出旧卫星队列里属于该用户的所有任务
- 从旧队列移除
- 如果新队列有空位，就迁入
- 迁入时给这些任务的 `upload_delay` 额外加上 `handover_delay`
- 如果新队列满了，这些任务直接丢失

这里的严格实现含义是：

- **迁移失败的任务不会重新放回旧队列**
- **也没有单独再记任务失败统计**
- 只通过 handover cost 里的迁移惩罚间接体现损失

### 5.7 切换失败后会发生什么

切换失败后：

- 用户状态直接变成 `BLOCKED`
- 当前不回退到旧卫星连接
- `failed_handovers += 1`
- 奖励吃 `failed_handover_penalty`

这意味着切换失败代价是比较重的。

## 6. Reward 是怎么计算的

## 6.1 环境返回的总 reward 结构

每个时间步，环境先为每个用户算一个 `user_reward_i`，最后返回：

```text
R_env(t) = mean_i user_reward_i(t)
```

也就是说：

- 环境步奖励是所有用户奖励的平均值
- 不是求和
- 这让不同用户数下 reward 尺度更稳定

### 6.2 单用户 reward 的组成

单用户一步的即时 reward 由三部分叠加：

```text
user_reward =
    handover_related_reward
  + immediate_offloading_reward
  + delayed_task_reward_from_pending_pool
```

其中：

- `handover_related_reward`：本步切换产生的增减分
- `immediate_offloading_reward`：本地立即算完或成功入队带来的增减分
- `pending_pool`：上一步或更早之前完成的卫星卸载任务，在这一时刻补发

## 6.3 任务 reward 的核心公式

对一个任务，先定义：

```text
delay_ratio = total_delay / max_delay
```

### 6.3.1 时延正奖励

```text
delay_reward_raw = max(1 - min(delay_ratio, 1), 0)
reward_delay = w_delay * delay_reward_raw
```

解释：

- 若 `delay_ratio <= 1`，越快越接近 `1`
- 若 `delay_ratio >= 1`，这一项直接归零

所以：

- 提前完成任务会加分
- 一旦超时，这一项没有任何正奖励

### 6.3.2 能耗正奖励

```text
energy_reward_raw = max(1 - min(total_energy / E_ref, 1), 0)
reward_energy = w_energy * energy_reward_raw
```

其中默认：

```text
E_ref = reward_energy_reference = 10.0
```

解释：

- 能耗越低越加分
- 当 `total_energy >= 10 J` 时，这项正奖励归零

### 6.3.3 QoS 奖励

```text
qos_reward_raw = 1  if delay_ratio <= 1
qos_reward_raw = -1 if delay_ratio > 1
reward_qos = w_qos * qos_reward_raw
```

解释：

- 按时完成就直接给正 QoS 奖励
- 超时就直接给负 QoS 奖励

### 6.3.4 Deadline 额外惩罚

若超时：

```text
penalty_deadline = -P_deadline * min(delay_ratio - 1, 2)
```

若没超时：

```text
penalty_deadline = 0
```

这意味着：

- 只要超时，就不仅失去 delay 正奖励和 QoS 正奖励
- 还会额外吃一层 deadline penalty
- 惩罚上限按 `delay_ratio - 1 <= 2` 截断

### 6.3.5 单任务 reward 汇总

```text
task_reward =
    reward_delay
  + reward_energy
  + reward_qos
  + penalty_deadline
```

## 6.4 不同任务执行方式下，total_delay 和 total_energy 怎么取

### 6.4.1 完全本地执行

```text
total_delay  = local_delay
total_energy = local_energy
```

reward 立即结算。

### 6.4.2 分片卸载且成功入队

任务刚入队时：

- 先不给完整 `task_reward`
- 只给 `enqueue_bonus`

等卫星侧完成或超时时：

```text
total_delay  = max(local_delay, offloaded_total_delay)
total_energy = local_energy + upload_energy
```

这里要注意：

- 总能耗只包含 **本地计算能耗 + 上传能耗**
- **没有**把卫星侧计算能耗计入 reward
- 也**没有**把下载端能耗计入 reward

### 6.4.3 队列满导致回退本地

```text
total_delay  = local_delay + fallback_delay
total_energy = local_energy + upload_energy + fallback_energy
```

然后立即计算任务 reward，并额外叠加：

```text
penalty_queue_full = -reward_queue_full_penalty
```

## 6.5 切换相关 reward

### 6.5.1 切换成功时的正项

成功后定义：

```text
elevation_score = clip(target_elev / 90, 0, 1)
rvt_score       = clip(target_rvt / rvt_threshold, 0, 1)
handover_score  = 0.5 * elevation_score + 0.5 * rvt_score
reward_handover = w_handover * handover_score
```

解释：

- 目标卫星仰角越高，切换正奖励越高
- 目标卫星剩余可见时间越长，切换正奖励越高

### 6.5.2 切换成本惩罚

切换成本被写成：

```text
delay_penalty = min(handover_delay_sec / 2, 1)
migration_penalty = 0.05 * migrated + 0.1 * failed
penalty_handover_cost = -w_handover * (delay_penalty + migration_penalty)
```

解释：

- 切换本身有信令/重建链路代价
- 如果迁移了任务，还要额外付任务迁移代价
- 迁移失败的代价高于迁移成功

### 6.5.3 负载均衡项

系统定义：

```text
active_load_sat = queue_length + connected_users
load_balance_score = 1 / (1 + std(active_loads))
```

只统计活跃负载大于 0 的卫星。

然后：

```text
balance_gain = balance_after - balance_before
reward_load_balance = w_lb * balance_gain
```

含义是：

- 如果切换后负载分布更均衡，`balance_gain > 0`，加分
- 如果切换后更不均衡，`balance_gain < 0`，减分

### 6.5.4 切换成功时的总切换 reward

```text
handover_reward_total =
    reward_handover
  + penalty_handover_cost
  + reward_load_balance
```

### 6.5.5 切换失败时的 reward

切换失败没有正项，只有：

```text
penalty_failed_handover = -reward_failed_handover_penalty
```

## 6.6 其他惩罚项

### 6.6.1 无效切换动作

当 `handover_action > 可见候选数` 时：

```text
penalty_invalid_action = -reward_invalid_action_penalty
```

同时还会：

- `total_handovers += 1`
- `failed_handovers += 1`

### 6.6.2 阻塞惩罚

当用户选择不切换，且当前卫星不可见、同时又没有其他可见卫星时：

```text
penalty_blocked = -reward_blocked_penalty
```

### 6.6.3 成功入队奖励

如果卸载分片成功进入队列：

```text
queue_margin = 1 - queue_length_after_enqueue / max_queue_size
reward_enqueue = reward_enqueue_bonus * max(queue_margin, 0)
```

这里用的是 **入队后的** 队列长度，因此：

- 队列越空，bonus 越高
- 队列越满，bonus 越低

## 6.7 用一句话概括 reward 的增减逻辑

### 什么时候加分

- 任务时延低于 deadline，且越低越好
- 任务能耗低于参考能耗 `10J`，且越低越好
- 任务按时完成
- 切到仰角高、RVT 长、目标更空闲的卫星
- 切换后系统负载更均衡
- 卸载分片成功入队，且目标队列越空 bonus 越大

### 什么时候减分

- 任务超时
- 能耗高到超过参考上限，不再拿到正能耗奖励
- 切换失败
- 选择了无效的切换动作
- 当前链路彻底丢失且无替代卫星
- 目标队列已满，导致卸载失败
- 切换需要额外时延，或迁移任务过多 / 迁移失败
- 切换后负载更不均衡

## 7. 统计指标是怎么定义的

环境最后统计的核心指标不是拍脑袋算的，而是这样定义的：

```text
resolved_tasks = completed_tasks + deadline_violations
pending_tasks  = total_tasks - resolved_tasks
```

### 7.1 任务相关

```text
task_completion_rate = completed_tasks / resolved_tasks
task_resolution_rate = resolved_tasks / total_tasks
pending_task_rate    = pending_tasks / total_tasks
avg_delay            = total_delay / resolved_tasks
```

含义要分清：

- `task_completion_rate` 不是“所有任务按时完成率”
- 它是“已经出结果的任务里，按时完成的比例”

### 7.2 切换相关

```text
handover_success_rate = successful_handovers / total_handovers
service_continuity_rate = 1 - forced_disconnects / (total_handovers + forced_disconnects)
```

这里非常重要：

- `service_continuity_rate` **不把 `failed_handovers` 算进去**
- 它只对 `forced_disconnects` 敏感

因此：

- 即使切换失败很多，只要不是“传播后失联”，连续性指标也可能仍然很好看

## 8. 智能体实际看到的状态和动作

### 8.1 环境原始观测

每个用户的原始观测维度是：

```text
3 + 1 + 5 + 10*6 + 4 = 73
```

包含：

- 用户位置 `(lat, lon, alt)`
- 用户状态
- 当前服务卫星的：
  - id
  - distance
  - elevation
  - snr
  - rvt
- 最多 10 个候选卫星的：
  - id
  - distance
  - elevation
  - snr
  - rvt
  - load
- 当前任务的：
  - data_size
  - computation
  - deadline
  - type

### 8.2 图表示

训练不是直接吃这 73 维原始观测，而是把系统编码成异构图：

- 节点类型：
  - `user`
  - `satellite`
- 边类型：
  - `user -> satellite`
  - `satellite -> user`
  - `satellite -> satellite`

图中的主要特征：

- 卫星节点特征：位置、速度、CPU 利用率、队列长度、连接用户数、可用频率
- 用户节点特征：位置、状态、服务卫星、任务参数、handover 历史、RVT warning
- 用户-卫星边特征：距离、仰角、SNR、速率、RVT、是否当前服务星
- 星间边特征：距离、传播时延、链路类型

### 8.3 实际送入 MAPPO 的用户观测

训练阶段真正送入 MAPPO 的是：

```text
HAN user embedding (64) + RVT warning (1) + task features (4) = 69 维
```

动作是混合动作：

- 离散动作：切换到第几个候选卫星
- 连续动作：卸载比例 `offload_ratio in [0, 1]`

离散动作头是 `Categorical`，连续动作头是 `Beta` 分布。

## 9. 代码级严格补全与实现注意点

下面这些点很容易在阅读论文式描述时被忽略，但对理解系统非常重要。

### 9.1 历史上的 `constellation.yaml` 不是运行时真配置

仓库里曾经存在 `config/constellation.yaml`，但环境从未把它加载进主流程，因此现在已经删除。

因此：

- 运行时真实配置以 `EnvConfig` / `MECConfig` / `TrainConfig` 为准
- 它更像历史参考说明，不是权威生效配置

### 9.2 队列不是严格 FCFS

虽然注释写了 FCFS，但代码行为是：

- 有限容量 admission
- 活跃任务并行均分 CPU

所以它更像 processor-sharing，而不是严格先来先服务。

### 9.3 `server.utilization` 在当前实现里几乎是二值的

因为 `process_queue()` 里只要存在活跃任务，就把：

- `available_freq_ghz = 0`

于是：

```text
utilization = 1 - available_freq / total_capacity = 1
```

这意味着：

- 只要队列非空，利用率基本就是 `1.0`
- handover success probability 里的 `load_headroom`
- 以及候选卫星观测里的 `load`

都更像“忙/不忙”二值信号，而不是细粒度利用率。

### 9.4 用户服务时间统计特征在环境里没有被持续更新

`User.total_service_time` 和 `total_blocked_time` 有定义，`UserManager.update_all_statistics()` 也有实现，但环境主循环没有调用它。

因此图特征中的：

- `connection_time`

在当前实现中基本不会像注释设想那样正常演化。

### 9.5 图特征里的部分归一化常数和真实上限并不完全一致

图特征提取器里使用了一些固定归一化常数，例如：

- `max_queue_length = 100`
- `max_users_per_sat = 50`

但实际运行时 MEC 队列上限是：

- `max_queue_size = 20`

这意味着：

- 图中的 `queue_len` 特征并不是按真实队列容量精确归一化的
- 某些输入特征更像“经验缩放”，不是严格物理归一化

### 9.6 `TaskManager` 不是任务生命周期主控制器

当前环境只是把新任务 `add_task` 进去做登记。

真正控制任务命运的是：

- `user_tasks`
- `MECServer.task_queue`
- `_offload_task_meta`
- `pending_rewards`

所以如果你要改任务机制，重点应该看环境主流程，而不是只看 `TaskManager`。

### 9.7 某些负 reward 没有细分到统计项里

例如 `_execute_offloading()` 在下列情况下直接返回 `-0.5`：

- `sat_id < 0`
- 当前卫星不可见
- 当前卫星没有 MEC server

这些 `-0.5` 会进入用户 reward，但不会被拆分记录到：

- `penalty_invalid_action`
- `penalty_blocked`
- `penalty_queue_full`

等 reward breakdown 字段里。

因此 reward breakdown 不是“绝对完备分解”。

### 9.8 卫星侧计算能耗没有纳入 reward

当前 `task_reward` 使用的 `total_energy` 只包含：

- 本地计算能耗
- 上传能耗

不包含：

- 卫星 MEC 计算能耗
- 下载能耗

所以 reward 里的 energy 更准确地说是：

- **用户侧能耗代理**

而不是全系统总能耗。

### 9.9 切换迁移失败的队列任务会丢失

若 handover 成功，但新卫星队列满：

- 旧卫星上的该用户任务会先被移除
- 新卫星装不下时直接记为迁移失败
- 这些任务不会回滚到旧卫星

这会带来两个含义：

- 任务可能在统计上“消失”
- 主要损失只通过 migration penalty 间接体现

### 9.10 新任务生成发生在动作执行之前

每个 step 的顺序是：

1. 清空可见性缓存
2. 生成任务
3. 执行动作
4. 更新环境

所以从强化学习时序看：

- 本步动作会作用在“本步刚刚到达的新任务”上
- 而这些新任务并不在上一步返回的观测里

这是一种显式的外生随机到达设定。

### 9.11 episode 是固定时长，不是任务完成驱动

环境不会因为：

- 所有任务完成
- 所有用户阻塞
- 某种 QoS 达标

而提前终止。

它只会在步数到 `max_steps` 时截断。

## 10. 最简短的系统本质总结

如果只用一句话概括这个系统：

> 这是一个以北京附近一组固定地面用户和 66 颗 LEO 卫星为背景的、多用户联合切换与部分卸载场景；每步用户要在“选哪颗卫星服务”和“任务卸载多少到卫星 MEC”之间做联合决策，任务可能本地完成、入队排队、迁移、超时或失联，而 reward 通过时延、能耗、QoS、切换质量、队列准入和负载均衡共同塑造策略。

如果只用一句话概括 reward：

> reward 的核心思想是“按时、低时延、低用户侧能耗、切到更稳更空闲的星、别把系统切乱”，并且对超时、切换失败、无效动作、队列满和阻塞做显式惩罚。
