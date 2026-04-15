# LEO 卫星网络 HAN+MAPPO 联合优化系统 — 完整技术文档

> **文档版本**: v4.0
> **更新日期**: 2026-04-15
> **适用范围**: 系统架构、模块设计、数据流、参数规格
> **说明**: 本文档以 `results/full_train_delay_focus/training_history.json` 与当前训练入口默认配置为准

---

## 一、研究问题与创新点

### 1.1 核心问题

低轨（LEO）卫星网络因高动态拓扑（轨道周期约 95.5 分钟、单颗卫星过境仅 5-10 分钟），导致：

1. **切换频繁** — 地面用户需频繁切换服务卫星，切换失败造成业务中断
2. **卸载耦合** — 计算任务卸载到卫星 MEC 服务器后，切换会导致迁移开销甚至任务丢失
3. **决策耦合** — 切换目标选择影响 MEC 队列负载，卸载比例影响本地能耗与时延

### 1.2 优化目标

$$
\min_{\pi} \quad \mathbb{E}\left[ \sum_{t=0}^{T} \left( w_1 \cdot T_{delay} + w_2 \cdot E_{energy} + w_3 \cdot C_{handover} - w_4 \cdot R_{QoS} \right) \right]
$$

| 目标 | 权重 | 代码配置 |
|------|------|----------|
| 优化任务时延 $T_{delay}$ | `1.4` | `reward_delay_weight` |
| 优化能耗 $E_{energy}$ | `0.4` | `reward_energy_weight` |
| 优化切换质量/切换成本 $C_{handover}$ | `0.3` | `reward_handover_weight` |
| 优化负载均衡 | `0.1` | `reward_load_balance_weight` |
| 优化 QoS 满足率 $R_{QoS}$ | `0.4` | `reward_qos_weight` |

### 1.3 解决方案

本系统将**切换决策**与**任务卸载决策**联合建模为一个**多智能体强化学习**问题，核心创新在于：

- 用**异质图注意力网络（HAN）** 编码用户-卫星拓扑关系
- 用**多智能体 PPO（MAPPO）** 进行分布式执行、集中式训练
- **混合动作空间**同时处理离散切换决策和连续卸载决策

### 1.4 智能体建模

> **每个地面用户是一个独立智能体。**

- 所有用户的 Actor **参数共享**（完全共享），通过各自不同的观测来区分
- Critic 为**集中式**，训练时可访问全局信息（所有用户嵌入 + 所有卫星嵌入）
- 每个智能体输出：**切换决策**（离散，Categorical 分布）+ **卸载比例**（连续，Beta 分布）

### 1.5 创新点对比

| 创新维度 | 论文一（宋晓勤-DDPG卸载） | 论文二（付一阳-HAN切换） | **本研究** |
|----------|--------------------------|------------------------|-----------|
| 优化问题 | 任务卸载 | 卫星切换 | **联合优化** ⭐ |
| 网络表征 | 无 | 异质图 | **增强异质图** ⭐ |
| 动作空间 | 连续 | 离散 | **混合动作** ⭐ |
| 智能体 | 单用户 | 多用户 | 多用户协作 |
| RL 算法 | DDPG | MAPPO | MAPPO |
| MEC 建模 | ✅ | ❌ | ✅ |
| 任务模型 | 可拆分 | ❌ | 可拆分 |

---

## 二、系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                      训练循环 (HANMAPPOTrainer)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐      ┌───────────────────────────────┐   │
│  │   LEO 仿真环境    │      │        决策模型                │   │
│  │  ┌──────────────┐│      │  ┌─────────────────────────┐  │   │
│  │  │Walker星座 66星││ 环境 │  │ HAN 编码器 (571K参数)    │  │   │
│  │  │地面用户 5~20 ││ 状态 │  │   3元路径 × 4头 × 2层   │  │   │
│  │  │信道/可见性   ││──────►  │   输出: 64维 节点嵌入    │  │   │
│  │  │MEC 任务队列  ││      │  └───────────┬─────────────┘  │   │
│  │  └──────────────┘│      │              │                 │   │
│  │        ▲          │      │              ▼                 │   │
│  │        │          │      │  ┌─────────────────────────┐  │   │
│  │        │   动作   │      │  │ 拼接: HAN(64)+RVT(1)+   │  │   │
│  │        │◄─────────│──────│  │        Task(4) = 69维   │  │   │
│  │        │          │      │  └───────────┬─────────────┘  │   │
│  └────────┼──────────┘      │              │                 │   │
│           │                 │              ▼                 │   │
│           │                 │  ┌─────────────────────────┐  │   │
│           │                 │  │ MAPPO                    │  │   │
│     奖励 + 新状态           │  │  Actor(68K): 切换+卸载   │  │   │
│           │                 │  │  Critic(183K): 集中式    │  │   │
│           ▼                 │  └─────────────────────────┘  │   │
│    MultiAgentRolloutBuffer  └───────────────────────────────┘   │
│           │                                                      │
│           ▼                                                      │
│      GAE + PPO Update → 更新 Actor/Critic/HAN 参数              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、模块详解

### 3.1 仿真环境 (`src/environment/`)

#### 3.1.1 Walker 星座 (`constellation.py`)

| 参数 | 值 | 说明 |
|------|----|------|
| 轨道面数 P | 6 | |
| 每面卫星数 S | 11 | 总计 66 颗 |
| 轨道高度 | 550 km | |
| 轨道倾角 | 53.0° | |
| 轨道周期 | ~95.5 min | 自动计算 |
| 相位因子 F | 1 | Walker-δ 53°:66/6/1 |

每颗卫星携带 MEC 服务器 (CPU 10 GHz, 队列容量 100)。

#### 3.1.2 地面用户 (`user.py`)

- **用户数**: 可配置 (3~20)，`train.py` 默认 5；`run_server_training.py` 标准方案默认 10
- **分布区域**: 北京 (39.9°N, 116.4°E) 为中心，半径约 100 km
- **状态机**: `IDLE → CONNECTED ⇄ HANDOVER → BLOCKED`
- **每个用户是一个独立智能体**

#### 3.1.3 信道模型 (`channel.py`)

```
接收功率 = 发射功率 + 天线增益 - 自由空间路损 - 大气衰减
SNR = 接收功率 - 噪声功率
容量 C = B × log₂(1 + SNR)
```

| 参数 | 值 | 代码字段 |
|------|----|----------|
| 载波频率 | **20 GHz (Ka 频段)** | `carrier_frequency_ghz = 20.0` |
| 信道带宽 | 50 MHz | `bandwidth_mhz = 50.0` |
| 卫星发射功率 | 40 dBm (10W) | `satellite_tx_power_dbm = 40.0` |
| 卫星天线增益 | 34 dB (相控阵) | `satellite_antenna_gain_db = 34.0` |
| 用户发射功率 | 33 dBm (2W) | `user_tx_power_dbm = 33.0` |
| 用户天线增益 | 38.5 dB (相控阵终端) | `user_antenna_gain_db = 38.5` |
| 等效噪声温度 | 354.81 K | `noise_temperature_k = 354.81` |
| 大气吸收损耗 | 0.3 dB | `atmospheric_loss_db = 0.3` |
| 最低仰角 | 10° | `min_elevation_deg = 10.0` |

#### 3.1.4 MEC 服务器 (`mec.py`)

- **调度**: FCFS + CPU 时间片平均分配
- **时延构成**: 上传延迟 + 排队延迟 + 处理延迟 + 下载延迟
- **切换迁移**: `migrate_user_tasks()` 将用户任务从旧卫星迁移到新卫星
- **超时处理**: 超过 `max_delay` 标记为 timeout 并触发惩罚
- **任务完成后奖励**: 通过 `pending_rewards` 机制延迟发放

#### 3.1.5 Gymnasium 环境 (`gym_env.py`)

**每步状态转移：**

```
1. 为每个用户生成新任务（概率触发）
2. 对每个用户执行动作：
   a. 切换动作 → _execute_handover()
   b. 卸载动作 → _execute_offloading() (本地+MEC入队)
3. 收集 pending_rewards（MEC 完成的任务奖励）
4. 更新 MEC 队列 → _update_environment()
5. 计算全局奖励 = mean(用户奖励)
6. 检查终止条件
```

**关键统计量（每 episode）：**

| 指标 | 含义 |
|------|------|
| `total_handovers` | 总切换尝试次数 |
| `successful_handovers` | 成功切换次数 |
| `failed_handovers` | 切换失败次数 |
| `total_tasks` | 总任务数 |
| `completed_tasks` | 完成的卸载任务数 |
| `deadline_violations` | 超时任务数 |
| `total_delay` | 累计时延 |
| `total_energy` | 累计能耗 |

---

### 3.2 异质图建模 (`src/graph/`)

#### 节点特征

**用户节点 (13 维):** `features.py → _extract_user_features()`

| 索引 | 特征名称 | 维度 | 归一化方式 |
|------|----------|------|------------|
| 0-2 | 位置 (ECEF: pos_x, pos_y, pos_z) | 3 | / 7000.0 km |
| 3 | 连接状态 (IDLE=0/CONNECTED=1/HANDOVER=2/BLOCKED=3) | 1 | / 3.0 |
| 4 | 服务卫星 ID | 1 | / num_satellites (未连接=-0.1) |
| 5 | 已连接时长 | 1 | / 600.0 s |
| 6 | 任务数据量 | 1 | / max_data_size |
| 7 | 任务计算量 | 1 | / max_computation |
| 8 | 任务时延要求 | 1 | / max_delay |
| 9 | 任务类型 (LIGHT=0/MEDIUM=1/HEAVY=2) | 1 | / 2.0 |
| 10 | 切换次数 | 1 | / 20 |
| 11 | 服务质量 (成功切换率) | 1 | 直接比率 |
| 12 | RVT 预警信号 (0 或 1) | 1 | 二值 |

**卫星节点 (10 维):** `features.py → _extract_satellite_features()`

| 索引 | 特征名称 | 维度 | 归一化方式 |
|------|----------|------|------------|
| 0-2 | 位置 (ECEF: pos_x, pos_y, pos_z) | 3 | / 7000.0 km |
| 3-5 | 速度 (ECI: vel_x, vel_y, vel_z) | 3 | / max_velocity |
| 6 | CPU 利用率 | 1 | [0, 1] |
| 7 | 任务队列长度 | 1 | / max_queue_length |
| 8 | 已连接用户数 | 1 | / max_users_per_sat |
| 9 | 可用计算资源比例 | 1 | available / max |

#### 边特征

**用户-卫星边 (6 维):** `features.py → _extract_user_satellite_edges()`

| 索引 | 特征名称 | 物理含义 |
|------|----------|----------|
| 0 | distance | 星地距离 (归一化) |
| 1 | elevation | 仰角 (归一化到 [0,1]，/90°) |
| 2 | snr | 信噪比 (归一化) |
| 3 | data_rate | 可达传输速率 (归一化) |
| 4 | rvt | 剩余可见时间 (归一化) |
| 5 | is_serving | 是否为当前服务卫星 (0 或 1) |

**星间链路边 (3 维):** `features.py → _extract_inter_satellite_edges()`

| 索引 | 特征名称 | 物理含义 |
|------|----------|----------|
| 0 | distance | 卫星间距离 (归一化，/5000 km) |
| 1 | prop_delay | 传播时延 (归一化，/20 ms) |
| 2 | link_type | 链路类型 (同轨道=0, 跨轨道=1) |

#### 元路径

| 元路径 | 语义 |
|--------|------|
| User → Sat → User | 共同可见卫星的用户关联（资源竞争关系） |
| Sat → User → Sat | 共同服务用户的卫星关联（切换候选关系） |
| Sat → Sat → Sat | 星间链路的卫星邻居关系 |

---

### 3.3 HAN 编码器 (`src/model/hetero_gnn.py`)

```
输入:
  user_features:  (N_u, 13)    # N_u 个用户
  sat_features:   (N_s, 10)    # N_s 个卫星(66)
  edge_index:     各类型边索引
  edge_features:  各类型边特征

处理:
  1. 节点类型投影:  user(13→64), sat(10→64)
  2. 元路径编码:    3条元路径 × GAT(4头, 2层)
  3. 语义级注意力:  加权融合 3 条元路径的输出
  4. 残差 + LayerNorm

输出:
  node_embeddings: (N_u + N_s, 64)    # 每个节点 64 维嵌入
```

**参数规模: 571,328**

---

### 3.4 MAPPO 算法 (`src/algorithm/mappo.py`)

#### Actor — 分布式执行 (`src/model/actor.py`)

```
输入: 观测 obs (69维) = HAN嵌入(64) + RVT预警(1) + 任务特征(4)

网络: shared_net MLP (69 → 256 → 128) → 双头输出
  ├── 离散头: Linear(128→64→K+1) → Categorical  → 切换动作 (0=不切换, 1~K=候选卫星)
  └── 连续头: alpha_head / beta_head → Softplus + 1 → Beta(α, β) → 卸载比例 λ ∈ (0, 1)

参数量: 76,589
```

> **注意**: 卸载比例使用 `Beta` 分布，原生支持 `[0,1]` 区间，采样后仅做数值安全 clamp。

#### Critic — 集中式训练 (`src/model/critic.py`)

```
输入:
  - 所有用户观测:  (N_u, 69)  → user_encoder Linear(69→128) → ReLU → mean pooling → (128,)
  - 所有卫星嵌入:  (N_s, 64)  → sat_encoder  Linear(64→128) → ReLU → mean pooling → (128,)
  - 拼接 → (256,) → value_net MLP (256 → 256 → 256 → 128 → 1)

输出: V(s) 状态值估计

参数量: 183,169
```

#### PPO 更新（`train.py` 默认）

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `learning_rate` | 3e-4 | Adam 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE λ |
| `clip_range` | 0.2 | PPO clip |
| `n_epochs` | 4 | 每次更新迭代次数 |
| `batch_size` | 256 | 小批量大小 |
| `entropy_coef` | 0.01 | 熵正则系数 |
| `clip_range_vf` | 0.2 | 价值函数 clip |
| `value_loss_type` | `huber` | Critic 损失 |
| `normalize_returns` | `true` | value loss 前标准化 returns |
| `max_grad_norm` | 0.5 | 梯度裁剪 |

#### 训练与早停参数（`train.py` 默认）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `exp_name` | `han_mappo_delay_focus_fast` | 默认实验名 |
| `num_users` | 10 | 默认用户数 |
| `max_steps` | 2000 | 每个 episode 最大步数 |
| `total_timesteps` | 1,000,000 | 总训练步数 |
| `n_steps` | 2048 | 每次更新收集步数 |
| `eval_interval` | 100,000 | 评估间隔 |
| `eval_episodes` | 3 | 每次评估 episode 数 |
| `graph_update_interval` | 100 | 图重建/重编码间隔 |
| `save_interval` | 200,000 | 检查点保存间隔 |
| `save_path` | `results/full_train_delay_focus` | 默认模型输出目录 |
| `log_path` | `results/logs` | 默认日志目录 |
| `early_stop_patience` | 30 | 连续 N 次更新无改善则提前停止（0=禁用） |

#### 服务器训练方案参数（`run_server_training.py`）

| 方案 | 用户数 | `max_steps` | `total_timesteps` | `n_steps` | `n_epochs` | `batch_size` | `eval_interval` | 输出目录 |
|------|--------|-------------|-------------------|-----------|------------|--------------|-----------------|----------|
| `standard` | 10 | 2000 | 1,000,000 | 2048 | 4 | 256 | 100,000 | `results/full_train_delay_focus` |
| `standard_fast` | 同 standard | 同 standard | 同 standard | 同 standard | 同 standard | 同 standard | 同 standard | `results/full_train_delay_focus` |
| `quick` | 5 | 1000 | 100,000 | 1024 | 4 | 256 | 10,000 | `results/quick_test_delay_focus` |
| `large` | 20 | 3000 | 2,000,000 | 4096 | 4 | 256 | 50,000 | `results/large_train_delay_focus` |
| `multi_seed` | 基于 standard | 同 standard | 同 standard（可被 `--steps` 覆盖） | 同 standard | 同 standard | 同 standard | 同 standard | `results/multi_seed/seed_*` |

---

### 3.5 奖励函数 (`gym_env.py`)

奖励在 `_execute_user_action()` 中逐用户计算，最终取所有用户均值。

```
用户奖励 R = R_handover + R_offload + R_pending

─── R_handover（切换奖励） ───
  切换时延惩罚:  -handover_delay_sec / 5.0  (每次切换)
  切换成功:      +0.1×(elevation/90) + 0.1×(rvt/600)
  切换失败:      -0.5
  迁移代价:      -0.05 × 迁移成功数 - 0.1 × 迁移失败数
  当前卫星不可见且无可见卫星: -1.0 (阻塞惩罚)

─── R_offload（本步立即奖励） ───
  无连接/不可见/无 MEC:  -0.5
  入队成功:               本步仅计入本地计算部分
  入队失败（队列满）:     惩罚

─── R_pending（延迟发放奖励） ───
  MEC 任务完成: 根据实际时延/能耗计算
  MEC 任务超时: 惩罚

全局奖励 = mean(所有用户的 R)
```

---

## 四、训练数据流

```
训练步 t:
│
├── 1. env.reset() / env.step()
│     └── 返回 observation, reward, done, info
│
├── 2. HeteroGraphBuilder.build()
│     └── 构建异质图 (节点特征, 边索引, 边特征)
│
├── 3. HANEncoder.forward()
│     └── 输出 node_embeddings (N_u+N_s, 64)
│
├── 4. 拼接观测: concat(han_embed[64], rvt_warning[1], task_feat[4]) = 69维
│
├── 5. Actor.forward(obs, candidate_mask)
│     └── 输出 actions = {handover: int, offload: float}
│
├── 6. Critic.forward(all_user_obs, sat_embeddings)
│     └── 输出 value estimate V(s)
│
├── 7. env.step(actions)
│     └── 返回 next_obs, reward, done, info
│
├── 8. Buffer.add(obs, action, reward, value, log_prob)
│
└── 9. 每 n_steps 步: MAPPO.update()
       ├── compute_returns_and_advantages (GAE)
       └── PPO clip update × n_epochs
```

---

## 五、输出指标

训练过程记录以下指标到 `training_history.json`：

| 类别 | 指标 | 说明 |
|------|------|------|
| **奖励** | `recent_mean_reward` | 最近 100 episode 平均奖励 |
| | `mean_reward` | 当前 rollout 中完成 episode 的平均奖励 |
| | `rollout_total_reward` | 当前 rollout 累计奖励 |
| **损失** | `actor_loss` | Actor 策略损失 |
| | `critic_loss` | Critic 值函数损失 |
| **策略** | `entropy` | 策略熵（探索程度） |
| | `kl_divergence` | 新旧策略 KL 散度 |
| | `clip_fraction` | PPO clip 触发比例 |
| **环境** | `handover_success_rate` | 切换成功率 |
| | `task_completion_rate` | 任务完成率 |
| | `avg_delay` | 平均任务时延 |
| | `total_energy` | 总能耗 |
| | `deadline_violations` | 超时任务数 |

---

## 六、模型总参数量

```
┌────────────────────┬───────────┬──────────────────────────────┐
│ 组件               │ 参数量     │ 结构                         │
├────────────────────┼───────────┼──────────────────────────────┤
│ HAN 编码器         │ 571,328   │ 2层×4头×3元路径, 64维输出     │
│ Actor (Beta Hybrid)│  76,589   │ MLP(69→256→128) + 双头       │
│ Critic (Centralized│ 183,169   │ 用户编码+卫星编码+MLP(256²→1)│
├────────────────────┼───────────┼──────────────────────────────┤
│ 总计               │ ~831K     │ 3090 完全够用                │
└────────────────────┴───────────┴──────────────────────────────┘
```

---

## 七、与参考论文的关系

### 论文一：宋晓勤《基于 DDPG 的任务卸载算法》

| 借鉴内容 | 本研究应用 |
|----------|-----------|
| 部分卸载模型 $\lambda \in [0,1]$ | 连续卸载比例 |
| 时延公式 $T = \max(T_{local}, T_{trans} + T_{sat})$ | MEC 时延模型 |
| 能耗公式 $E = E_{local} + E_{trans}$ | 能耗计算 |
| MEC 服务器模型 | 星载 MEC CPU/队列 |

**改进**: 单用户 → 多用户协作 (MAPPO)；只卸载 → 联合切换+卸载；无拓扑 → HAN 异质图；DDPG → MAPPO

### 论文二：付一阳《基于异质图的切换方法》

| 借鉴内容 | 本研究应用 |
|----------|-----------|
| 异质图建模 | 用户-卫星二部图 + ISL |
| HAN 网络 | 元路径 + 节点级/语义级注意力 |
| MAPPO 框架 | 参数共享 Actor + 集中式 Critic |
| Walker 星座 | 6×11=66 颗卫星 |

**改进**: 只切换 → 联合优化；离散动作 → 混合动作空间；无 MEC → 完整 MEC 模型

---

## 八、代码模块与文件对照

| 模块 | 文件 | 对应方法 |
|------|------|----------|
| Walker 星座 | `src/environment/constellation.py` | 轨道仿真 |
| 信道模型 | `src/environment/channel.py` | Ka 频段链路 |
| MEC 服务器 | `src/environment/mec.py` | 时延/能耗/队列 |
| 用户管理 | `src/environment/user.py` | 状态机/位置 |
| Gym 环境 | `src/environment/gym_env.py` | RL 环境接口 |
| 特征提取 | `src/graph/features.py` | 节点/边特征 |
| 图构建 | `src/graph/builder.py` | 异质图拓扑 |
| HAN 编码器 | `src/model/hetero_gnn.py` | 图注意力网络 |
| Actor 网络 | `src/model/actor.py` | 混合动作输出 |
| Critic 网络 | `src/model/critic.py` | 集中式价值估计 |
| MAPPO 算法 | `src/algorithm/mappo.py` | PPO 更新 |
| 训练入口 | `scripts/train.py` | 端到端训练 |
| 可视化 | `scripts/plot_results.py` | 训练曲线绘图 |
| 服务器训练 | `scripts/run_server_training.py` | 批量训练 |
