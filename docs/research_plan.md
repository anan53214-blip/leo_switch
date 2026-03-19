# 星地融合网络切换与任务卸载联合优化研究计划

> **项目名称**：基于异质图神经网络的LEO卫星网络切换与任务卸载联合优化方法  
> **创建日期**：2026年1月15日  
> **最后更新**：2026年1月16日  
> **版本**：v2.1  
> **作者**：[待填写]

---

## 一、研究背景与目标

### 1.1 研究背景

低轨（LEO）卫星网络因其低时延、广覆盖的特性，成为6G通信的重要组成部分。然而，LEO卫星高速运动带来以下挑战：

| 挑战 | 描述 | 影响 |
|------|------|------|
| 频繁切换 | 单星覆盖时间短（约8-10分钟） | 服务中断、切换开销 |
| 计算资源受限 | 星载MEC计算能力有限 | 任务处理时延增加 |
| 动态拓扑 | 网络拓扑持续变化 | 决策复杂度高 |
| 多用户竞争 | 大量用户同时接入 | 资源分配冲突 |

### 1.2 研究问题定义

（待补充）

### 1.3 研究目标

| 目标层次 | 具体内容 |
|---------|---------|
| **核心目标** | 设计基于异质图神经网络的多智能体切换与卸载联合优化算法 |
| **性能目标** | 降低任务时延、减少能耗、提高切换成功率、保证服务连续性 |
| **创新目标** | 异质图建模 + 混合动作空间 + MAPPO多智能体协作 |

### 1.4 参考论文

1. 《星地融合网络中基于异质图表征的多智能体协作切换方法》- 付一阳等
2. 《基于深度确定性策略梯度的星地融合网络可拆分任务卸载算法》- 宋晓勤等

---

## 二、研究创新点

### 2.1 现有工作的不足

（待补充）

### 2.2 本研究的创新贡献

（待补充）

### 2.3 与现有方法对比

| 维度 | 论文一(宋晓勤) | 论文二(付一阳) | **本研究** |
|------|---------------|---------------|-----------|
| **网络表征** | 无 | 异质图(HAN) | 增强型异质图 |
| **决策问题** | 任务卸载 | 卫星切换 | **联合优化** |
| **动作空间** | 连续 | 离散 | **混合** |
| **智能体数** | 单用户 | 多用户 | 多用户 |
| **算法** | DDPG | MAPPO | MAPPO(混合) |
| **优化目标** | 时延+能耗 | 切换成功率 | **综合优化** |
| **MEC建模** | 有 | 无 | **有** |

---

## 三、研究内容与方法

### 3.1 研究内容

1. 基于异质图神经网络的LEO卫星网络建模
2. 多智能体协作的任务卸载与资源分配算法
3. 网络切换策略优化

### 3.2 研究方法

1. 图神经网络
2. 强化学习
3. 优化算法

### 3.2 任务卸载模型

#### 3.2.1 任务模型

用户 $i$ 在时隙 $t$ 的计算任务表示为：

$$
\mathcal{T}_i(t) = \{D_i(t), C_i(t), T_i^{max}(t)\}
$$

| 符号 | 含义 | 单位 |
|------|------|------|
| $D_i(t)$ | 任务数据量 | bits |
| $C_i(t)$ | 任务计算量 | CPU cycles |
| $T_i^{max}(t)$ | 最大容忍时延 | s |

#### 3.2.2 卸载决策

用户 $i$ 的卸载比例 $\lambda_i \in [0, 1]$：
- $\lambda_i = 0$：完全本地执行
- $\lambda_i = 1$：完全卸载到卫星MEC
- $0 < \lambda_i < 1$：部分卸载

#### 3.2.3 时延模型

**本地执行时延：**
$$
T_i^{local} = \frac{(1-\lambda_i) \cdot C_i}{f_i^{local}}
$$

**传输时延：**
$$
T_i^{trans} = \frac{\lambda_i \cdot D_i}{R_i}
$$

其中传输速率 $R_i$ 由香农公式计算：
$$
R_i = B \cdot \log_2(1 + \frac{P_i \cdot G_i}{N_0 \cdot B})
$$

**卫星计算时延：**
$$
T_i^{sat} = \frac{\lambda_i \cdot C_i}{f_i^{sat}}
$$

**总时延：**
$$
T_i^{total} = \max(T_i^{local}, T_i^{trans} + T_i^{sat})
$$

#### 3.2.4 能耗模型

**本地计算能耗：**
$$
E_i^{local} = \kappa \cdot (1-\lambda_i) \cdot C_i \cdot (f_i^{local})^2
$$

**传输能耗：**
$$
E_i^{trans} = P_i \cdot T_i^{trans}
$$

**总能耗：**
$$
E_i^{total} = E_i^{local} + E_i^{trans}
$$

### 3.3 切换模型

#### 3.3.1 切换触发条件

用户 $i$ 触发切换的条件：
1. 当前服务卫星即将离开可见范围（RVT < 阈值）
2. 存在更优的目标卫星（更高SNR/更长RVT/更低负载）
3. 当前链路质量下降到阈值以下

#### 3.3.2 切换开销

$$
C_{handover} = C_{signaling} + C_{interruption} + C_{context}
$$

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $C_{signaling}$ | 信令开销 | 100ms |
| $C_{interruption}$ | 服务中断时间 | 50-200ms |
| $C_{context}$ | 上下文迁移开销 | 取决于任务状态 |

### 3.4 异质图模型

#### 3.4.1 图结构定义

$$
\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{X}, \mathcal{A})
$$

**节点类型 $\mathcal{V}$：**

| 类型 | 符号 | 数量 | 特征维度 |
|------|------|------|----------|
| 卫星节点 | $v_s$ | $N_s$ | 10 |
| 用户节点 | $v_u$ | $N_u$ | 12 |

**边类型 $\mathcal{E}$：**

| 类型 | 符号 | 说明 |
|------|------|------|
| 星间链路 | ISL | 同轨道/跨轨道相邻卫星 |
| 用户-卫星链路 | UDL | 用户到可见卫星 |
| 卫星-用户链路 | SDL | 卫星到用户（UDL反向边） |

#### 3.4.2 节点特征设计

**卫星节点特征 $\mathbf{x}_s \in \mathbb{R}^{10}$：**

```python
satellite_features = [
    position_x, position_y, position_z,  # 位置 (3)
    velocity_x, velocity_y, velocity_z,  # 速度 (3)
    cpu_capacity,                         # 计算能力 (1)
    cpu_utilization,                      # CPU利用率 (1)
    queue_length,                         # 任务队列长度 (1)
    connected_users,                      # 已连接用户数 (1)
]
```

**用户节点特征 $\mathbf{x}_u \in \mathbb{R}^{12}$：**

```python
user_features = [
    position_x, position_y, position_z,  # 位置 (3)
    velocity_x, velocity_y, velocity_z,  # 速度 (3)
    task_data_size,                       # 当前任务数据量 (1)
    task_compute_demand,                  # 当前任务计算量 (1)
    current_satellite_id,                 # 当前连接卫星 (1)
    connection_duration,                  # 已连接时长 (1)
    current_snr,                          # 当前SNR (1)
    remaining_visible_time,               # 剩余可见时间 (1)
]
```

**用户-卫星边特征 $\mathbf{x}_{us} \in \mathbb{R}^{6}$：**

```python
udl_edge_features = [
    distance,           # 距离 (km)
    elevation_angle,    # 仰角 (deg)
    azimuth_angle,      # 方位角 (deg)
    snr,                # 信噪比 (dB)
    data_rate,          # 可达速率 (Mbps)
    remaining_vis_time, # 剩余可见时间 (s)
]
```

---

## 四、研究计划与进度

| 阶段 | 任务 | 时间 |
|------|------|------|
| 1 | 文献调研与问题定义 | 2026.1 - 2026.3 |
| 2 | 网络模型与算法设计 | 2026.4 - 2026.6 |
| 3 | 算法优化与仿真实验 | 2026.7 - 2026.9 |
| 4 | 论文撰写与总结 | 2026.10 - 2026.12 |

---

## 五、预期成果

1. 发表高水平学术论文
2. 申请相关专利
3. 完成研究报告

---

## 六、系统参数配置

### 6.1 星座参数

| 参数 | 符号 | 数值 | 单位 | 说明 |
|------|------|------|------|------|
| 轨道平面数 | $P$ | 6 | - | Walker星座轨道平面数 |
| 每轨道卫星数 | $S$ | 11 | - | 每个轨道平面的卫星数量 |
| 卫星总数 | $N_s$ | 66 | - | $P \times S = 66$ |
| 轨道高度 | $h$ | 550 | km | LEO轨道高度 |
| 轨道倾角 | $i$ | 53.0 | deg | 轨道倾角 |
| 轨道周期 | $T$ | ~95.5 | min | 由轨道高度决定 |
| 最小仰角 | $\theta_{min}$ | 10 | deg | 用户可见性阈值 |
| 相位因子 | $F$ | 1 | - | Walker相位因子 |

### 6.2 信道参数

| 参数 | 符号 | 数值 | 单位 | 说明 |
|------|------|------|------|------|
| 载波频率 | $f_c$ | 20 | GHz | Ka频段 |
| 系统带宽 | $B$ | 50 | MHz | 信道带宽 |
| 用户发射功率 | $P_t$ | 33 (2W) | dBm | 用户终端发射功率 |
| 天线增益 | $G$ | 38.5 | dB | 用户天线增益 |
| 噪声功率谱密度 | $N_0$ | -174 | dBm/Hz | 热噪声 |
| 大气损耗 | $L_{atm}$ | 0.5 | dB | 大气衰减（晴空） |
| 雨衰 | $L_{rain}$ | 3.0 | dB | 降雨衰减（中雨） |

**典型链路性能（550km高度）：**
| 仰角 | 距离(km) | FSPL(dB) | SNR(dB) | 速率(Mbps) |
|------|----------|----------|---------|------------|
| 90° | 550 | ~182 | ~25 | ~420 |
| 45° | ~760 | ~185 | ~22 | ~365 |
| 10° | ~1500 | ~191 | ~16 | ~265 |

### 6.3 MEC计算参数

| 参数 | 符号 | 数值 | 单位 | 说明 |
|------|------|------|------|------|
| 卫星CPU频率 | $f^{sat}$ | 10 | GHz | 星载MEC服务器 |
| 用户CPU频率 | $f^{local}$ | 1 | GHz | 用户终端计算能力 |
| 能耗系数 | $\kappa$ | $10^{-27}$ | - | 计算能耗系数 |
| 最大队列长度 | $Q_{max}$ | 100 | - | MEC任务队列上限 |

### 6.4 任务参数

| 参数 | 符号 | 数值范围 | 单位 | 说明 |
|------|------|----------|------|------|
| 数据量 | $D_i$ | 0.5-2.0 | MB | 任务输入数据大小 |
| 计算需求 | $C_i$ | 100-1000 | Mcycles | 任务计算量 |
| 时延约束 | $T_{deadline}$ | 0.5-5.0 | s | 任务截止时间 |
| 任务类型 | - | 0-2 | - | 计算密集/数据密集/混合 |

### 6.5 图特征参数

| 节点/边类型 | 特征维度 | 特征内容 |
|-------------|----------|----------|
| 卫星节点 | 10 | 位置(3) + 速度(3) + CPU容量(1) + CPU利用率(1) + 队列长度(1) + 连接用户数(1) |
| 用户节点 | 12 | 位置(3) + 速度(3) + 任务数据量(1) + 任务计算量(1) + 当前卫星ID(1) + 连接时长(1) + SNR(1) + RVT(1) |
| 用户-卫星边(UDL) | 6 | 距离(1) + 仰角(1) + 方位角(1) + SNR(1) + 速率(1) + RVT(1) |
| 星间链路边(ISL) | 3 | 距离(1) + 同轨道标志(1) + 链路质量(1) |

### 6.6 HAN网络参数

| 参数 | 数值 | 说明 |
|------|------|------|
| 隐藏层维度 | 64 | 节点嵌入维度 |
| 注意力头数 | 4 | 多头注意力 |
| 网络层数 | 2 | HAN层数 |
| Dropout率 | 0.1 | 正则化 |
| 激活函数 | ELU | 非线性激活 |

### 6.7 MAPPO算法参数

| 参数 | 符号 | 数值 | 说明 |
|------|------|------|------|
| 学习率 | $\alpha$ | 3e-4 | Adam优化器 |
| 折扣因子 | $\gamma$ | 0.99 | 奖励折扣 |
| GAE参数 | $\lambda$ | 0.95 | 广义优势估计 |
| 裁剪范围 | $\epsilon$ | 0.2 | PPO裁剪 |
| 训练轮数 | epochs | 10 | 每次更新的epoch数 |
| 批大小 | batch_size | 64 | Mini-batch大小 |
| 缓冲区大小 | buffer_size | 2048 | 经验回放大小 |
| 熵系数 | $c_H$ | 0.01 | 探索激励 |
| 值函数系数 | $c_V$ | 0.5 | 值损失权重 |
| 梯度裁剪 | max_grad_norm | 0.5 | 梯度裁剪阈值 |

### 6.8 奖励函数权重

| 权重 | 符号 | 数值 | 说明 |
|------|------|------|------|
| 时延权重 | $w_1$ | 0.4 | 时延惩罚系数 |
| 能耗权重 | $w_2$ | 0.3 | 能耗惩罚系数 |
| 切换权重 | $w_3$ | 0.2 | 切换惩罚系数 |
| QoS权重 | $w_4$ | 0.1 | QoS满足奖励系数 |

---

## 七、代码模块说明

### 7.1 环境模块 (`src/environment/`)

| 文件 | 主要类/函数 | 功能描述 | 实现状态 |
|------|-------------|----------|----------|
| `constellation.py` | `WalkerConstellation` | Walker星座模型，计算卫星位置、速度、轨道参数，支持reset()时间步进 | ✅ 已完成 |
| `visibility.py` | `SatelliteVisibility` | 卫星可见性计算，包括仰角、方位角、距离、剩余可见时间(RVT)等 | ✅ 已完成 |
| `user.py` | `User`, `UserManager` | 用户模型，管理用户位置、状态、当前连接卫星、任务分配 | ✅ 已完成 |
| `task.py` | `Task`, `TaskGenerator` | 任务模型，定义数据量、计算需求、截止时间、任务类型 | ✅ 已完成 |
| `channel.py` | `ChannelConfig`, `SatelliteChannel`, `MultiUserChannel` | 卫星信道模型，计算自由空间路径损耗(FSPL)、SNR、Shannon容量 | ✅ 已完成 |
| `mec.py` | `MECConfig`, `MECServer`, `OffloadingCalculator` | MEC计算模型，计算本地/卸载时延和能耗，支持部分卸载 | ✅ 已完成 |
| `gym_env.py` | `EnvConfig`, `LEOSatelliteEnv` | Gymnasium强化学习环境，整合所有模块，定义状态/动作/奖励 | ✅ 已完成 |

**关键接口：**
```python
# LEOSatelliteEnv - 主环境接口
env = LEOSatelliteEnv(config)
obs, info = env.reset()           # 重置环境
obs, reward, done, truncated, info = env.step(actions)  # 执行动作
```

### 7.2 图模块 (`src/graph/`)

| 文件 | 主要类/函数 | 功能描述 | 实现状态 |
|------|-------------|----------|----------|
| `features.py` | `NodeFeatures`, `EdgeFeatures`, `FeatureExtractor` | 特征提取器，从环境中提取卫星/用户节点特征和边特征 | ✅ 已完成 |
| `builder.py` | `HeteroGraphData`, `HeteroGraphBuilder` | 异质图构建器，构建包含多种节点和边的异质图，支持PyG/DGL格式 | ✅ 已完成 |

**关键接口：**
```python
# 构建异质图
builder = HeteroGraphBuilder(env, feature_extractor)
graph_data = builder.build()           # 构建异质图
pyg_graph = builder.to_pyg(graph_data) # 转为PyG格式
metapaths = builder.get_metapaths()    # 获取元路径
```

**元路径定义：**
- `user-connect-sat-serve-user`：用户→卫星→用户（共享卫星的用户关系）
- `sat-isl-sat`：卫星→星间链路→卫星（卫星网络拓扑）

### 7.3 模型模块 (`src/model/`)

| 文件 | 主要类/函数 | 功能描述 | 实现状态 |
|------|-------------|----------|----------|
| `layers.py` | `MLP`, `GraphAttentionLayer`, `HeterogeneousAttentionLayer`, `SemanticAttention` | 基础层实现，包括MLP、图注意力层、异质图注意力层、语义注意力 | ✅ 已完成 |
| `hetero_gnn.py` | `HANConfig`, `MetapathEncoder`, `HeterogeneousAttentionNetwork`, `HANEncoder` | 异质图注意力网络(HAN)，基于元路径的节点嵌入学习 | ✅ 已完成 |
| `actor.py` | `ActorConfig`, `HybridActor`, `MultiAgentActor` | 混合动作Actor网络，输出离散切换决策+连续卸载比例 | ✅ 已完成 |
| `critic.py` | `CriticConfig`, `SharedCritic`, `CentralizedCritic`, `create_global_state()` | 中心化Critic网络，基于全局状态评估价值函数 | ✅ 已完成 |

**关键接口：**
```python
# HAN编码器
encoder = HANEncoder(config)
node_embeddings = encoder(node_features, edge_index_dict)

# 混合Actor
actor = MultiAgentActor(config)
actions, log_probs, entropy = actor(observations, available_actions)

# 中心化Critic
critic = CentralizedCritic(config)
values = critic(global_state)
```

### 7.4 算法模块 (`src/algorithm/`)

| 文件 | 主要类/函数 | 功能描述 | 实现状态 |
|------|-------------|----------|----------|
| `buffer.py` | `RolloutBuffer`, `MultiAgentRolloutBuffer` | 经验回放缓冲区，存储轨迹数据，计算GAE优势估计 | ✅ 已完成 |
| `mappo.py` | `MAPPOConfig`, `MAPPO` | MAPPO算法实现，包含act()、get_value()、update()、save/load等 | ✅ 已完成 |
| `runner.py` | `RunnerConfig`, `Runner` | 训练运行器，管理训练循环、评估、日志记录 | ✅ 已完成 |

**关键接口：**
```python
# MAPPO算法
mappo = MAPPO(config)
actions, log_probs, value = mappo.act(observations, candidate_masks)  # 选择动作
value = mappo.get_value(observations)                                  # 计算价值
loss_info = mappo.update()                                             # 更新网络
mappo.save("checkpoint.pt")                                            # 保存模型

# 训练器（推荐使用）
from scripts.train import HANMAPPOTrainer, TrainConfig
config = TrainConfig(num_users=10, total_timesteps=500000)
trainer = HANMAPPOTrainer(config)
trainer.train()                                                        # 开始训练
```

### 7.5 测试文件 (`tests/`)

| 文件 | 测试内容 | 状态 |
|------|----------|------|
| `test_channel.py` | 信道模型测试：FSPL、SNR、Shannon容量 | ✅ 通过 |
| `test_mec.py` | MEC模型测试：时延计算、能耗计算、部分卸载 | ✅ 通过 |
| `test_gym_env.py` | Gym环境测试：reset、step、reward | ✅ 通过 |
| `test_graph.py` | 图模块测试：特征提取、图构建、格式转换 | ✅ 通过 |

### 7.6 脚本文件 (`scripts/`)

| 文件 | 功能描述 | 状态 |
|------|----------|------|
| `test_graph_integration.py` | 图模块集成测试 | ✅ 已完成 |
| `test_model.py` | 模型模块测试 | ✅ 已完成 |
| `test_algorithm.py` | 算法模块测试 | ✅ 已完成 |
| `train.py` | 完整训练脚本（HAN+MAPPO） | ✅ 已完成 |
| `evaluate.py` | 评估脚本 | 🔄 待实现 |
| `visualize.py` | 可视化脚本 | 🔄 待实现 |

### 7.7 训练脚本使用方法 (`scripts/train.py`)

训练脚本整合了 HAN（异质图注意力网络）+ MAPPO（多智能体PPO）进行联合优化训练。

#### 基本使用

```bash
# 默认训练（500,000步）
python scripts/train.py

# 快速测试（短期训练）
python scripts/train.py --total_timesteps 1000 --n_steps 200 --device cpu

# 指定用户数和episode长度
python scripts/train.py --num_users 10 --max_steps 500 --device cpu
```

#### 完整参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--exp_name` | `han_mappo_leo` | 实验名称 |
| `--seed` | `42` | 随机种子 |
| `--device` | `auto` | 设备（cuda/cpu/auto） |
| `--num_users` | `5` | 用户数量 |
| `--max_steps` | `1000` | 每episode最大步数 |
| `--total_timesteps` | `500000` | 总训练步数 |
| `--n_steps` | `2048` | 每次更新收集步数 |
| `--learning_rate` | `3e-4` | 学习率 |
| `--batch_size` | `64` | 批大小 |
| `--han_hidden_dim` | `64` | HAN隐藏维度 |
| `--han_num_heads` | `4` | 注意力头数 |
| `--han_num_layers` | `2` | HAN层数 |
| `--eval_interval` | `10000` | 评估间隔 |
| `--save_path` | `results/models` | 模型保存路径 |
| `--load_path` | `None` | 加载检查点路径 |
| `--eval_only` | `False` | 仅评估模式 |

#### 示例命令

```bash
# 完整训练（推荐配置）
python scripts/train.py \
    --num_users 10 \
    --total_timesteps 1000000 \
    --n_steps 2048 \
    --eval_interval 50000 \
    --device cpu

# 从检查点恢复训练
python scripts/train.py \
    --load_path results/models/checkpoint_100000.pt \
    --total_timesteps 500000

# 仅评估已保存模型
python scripts/train.py \
    --load_path results/models/best_model.pt \
    --eval_only
```

#### 输出文件

训练完成后会生成以下文件：

```
results/
├── models/
│   ├── best_model.pt          # 最佳奖励模型
│   ├── final_model.pt         # 最终模型
│   └── checkpoint_*.pt        # 定期检查点
└── logs/
    └── han_mappo_leo_*.log    # 训练日志
```

---

## 八、参考文献

1. 付一阳等. "星地融合网络中基于异质图表征的多智能体协作切换方法" - 异质图+多智能体切换
2. 宋晓勤等. "基于深度强化学习的LEO卫星边缘计算任务卸载策略" - DDPG任务卸载

---

## 九、联合优化算法框架

````markdown
┌─────────────────────────────────────────────────────────────────────┐
│                    联合优化算法框架                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                      环境状态获取                              │ │
│  │  • 卫星位置/速度/负载                                         │ │
│  │  • 用户位置/任务/连接状态                                     │ │
│  │  • 链路质量/可见性                                            │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    异质图构建与编码                            │ │
│  │  ┌─────────────────┐    ┌─────────────────┐                  │ │
│  │  │  图构建器       │───▶│  HAN编码器      │                  │ │
│  │  │  HeteroGraph    │    │  节点嵌入       │                  │ │
│  │  └─────────────────┘    └─────────────────┘                  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                   MAPPO多智能体决策                            │ │
│  │                                                               │ │
│  │   ┌─────────────────────────────────────────────────────┐    │ │
│  │   │                   共享 Critic                        │    │ │
│  │   │         输入: 全局图嵌入 → 输出: V(s)                │    │ │
│  │   └─────────────────────────────────────────────────────┘    │ │
│  │                           ▲                                   │ │
│  │       ┌───────────────────┼───────────────────┐              │ │
│  │       ▼                   ▼                   ▼              │ │
│  │   ┌────────┐          ┌────────┐          ┌────────┐        │ │
│  │   │Actor 1 │          │Actor 2 │          │Actor N │        │ │
│  │   │(User 1)│          │(User 2)│          │(User N)│        │ │
│  │   └───┬────┘          └───┬────┘          └───┬────┘        │ │
│  │       │                   │                   │              │ │
│  │       ▼                   ▼                   ▼              │ │
│  │   ┌────────┐          ┌────────┐          ┌────────┐        │ │
│  │   │混合动作│          │混合动作│          │混合动作│        │ │
│  │   │h₁, λ₁ │          │h₂, λ₂ │          │hₙ, λₙ │        │ │
│  │   └────────┘          └────────┘          └────────┘        │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                      执行与奖励计算                            │ │
│  │  • 执行切换动作                                               │ │
│  │  • 执行任务卸载                                               │ │
│  │  • 计算综合奖励                                               │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
````

````markdown
LEO_switch/
├── config/                          # 配置文件
│   └── constellation.yaml           # 星座参数配置
│
├── docs/                            # 文档
│   └── research_plan.md             # 研究计划文档（本文件）
│
├── src/                             # 源代码
│   ├── __init__.py
│   │
│   ├── environment/                 # 仿真环境 ✅
│   │   ├── __init__.py
│   │   ├── constellation.py         # Walker星座模型
│   │   ├── visibility.py            # 可见性计算
│   │   ├── user.py                  # 用户模型
│   │   ├── task.py                  # 任务模型
│   │   ├── channel.py               # 卫星信道模型（Ka频段）
│   │   ├── mec.py                   # MEC计算模型
│   │   └── gym_env.py               # Gymnasium环境
│   │
│   ├── graph/                       # 图构建 ✅
│   │   ├── __init__.py
│   │   ├── builder.py               # 异质图构建器
│   │   └── features.py              # 特征提取器
│   │
│   ├── model/                       # 神经网络 ✅
│   │   ├── __init__.py
│   │   ├── layers.py                # 基础层（MLP、注意力层）
│   │   ├── hetero_gnn.py            # HAN异质图注意力网络
│   │   ├── actor.py                 # 混合动作Actor
│   │   └── critic.py                # 中心化Critic
│   │
│   ├── algorithm/                   # RL算法 ✅
│   │   ├── __init__.py
│   │   ├── buffer.py                # 经验缓冲区（GAE）
│   │   ├── mappo.py                 # MAPPO算法实现
│   │   └── runner.py                # 训练运行器
│   │
│   └── utils/                       # 工具 🔄
│       ├── __init__.py
│       ├── logger.py                # 日志（待实现）
│       ├── metrics.py               # 评价指标（待实现）
│       └── visualization.py         # 可视化（待实现）
│
├── scripts/                         # 执行脚本
│   ├── test_graph_integration.py    # 图模块测试 ✅
│   ├── test_model.py                # 模型测试 ✅
│   ├── test_algorithm.py            # 算法测试 ✅
│   ├── train.py                     # 训练入口 ✅
│   ├── evaluate.py                  # 评估脚本（待实现）
│   └── visualize.py                 # 可视化脚本（待实现）
│
├── baselines/                       # 对比算法 🔄
│   ├── random_policy.py             # 随机策略（待实现）
│   ├── greedy_policy.py             # 贪婪策略（待实现）
│   └── ddpg_offload.py              # DDPG对比（待实现）
│
├── tests/                           # 单元测试 ✅
│   ├── test_channel.py              # 信道模型测试
│   ├── test_mec.py                  # MEC模型测试
│   ├── test_gym_env.py              # Gym环境测试
│   └── test_graph.py                # 图模块测试
│
├── results/                         # 实验结果
│   ├── models/                      # 保存的模型
│   ├── logs/                        # 训练日志
│   └── figures/                     # 生成的图表
│
├── LEO_offloading/                  # STK仿真场景文件
│   ├── *.sa, *.sa3                  # 卫星对象文件（66颗）
│   ├── MEC_base.*                   # 地面站/MEC基站
│   └── Receiver1.*                  # 接收机
│
├── environment.yml                  # Conda环境配置
├── requirements.txt                 # pip依赖
└── README.md                        # 项目说明
````

class HybridActionSpace:
    """
    混合动作空间：离散切换决策 + 连续卸载比例
    """
    
    def __init__(self, num_satellites, num_users):
        self.num_satellites = num_satellites
        self.num_users = num_users
        
        # 每个用户的动作空间
        # 离散部分：选择目标卫星 (0=不切换, 1~K=切换到卫星k)
        self.handover_dim = num_satellites + 1
        
        # 连续部分：卸载比例 λ ∈ [0, 1]
        self.offload_dim = 1
    
    def sample(self):
        """采样一个混合动作"""
        handover = np.random.randint(0, self.handover_dim)
        offload = np.random.uniform(0, 1)
        return {'handover': handover, 'offload': offload}

class HybridActor(nn.Module):
    """
    混合动作Actor网络
    输出：切换概率分布 + 卸载比例的高斯分布参数
    """
    
    def __init__(self, input_dim, hidden_dim, num_satellites):
        super().__init__()
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 离散动作头：切换决策
        self.handover_head = nn.Linear(hidden_dim, num_satellites + 1)
        
        # 连续动作头：卸载比例
        self.offload_mean = nn.Linear(hidden_dim, 1)
        self.offload_log_std = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.shared(x)
        
        # 切换概率
        handover_logits = self.handover_head(features)
        handover_probs = F.softmax(handover_logits, dim=-1)
        
        # 卸载比例分布
        offload_mean = torch.sigmoid(self.offload_mean(features))
        offload_std = torch.exp(self.offload_log_std(features).clamp(-20, 2))
        
        return {
            'handover_probs': handover_probs,
            'offload_mean': offload_mean,
            'offload_std': offload_std,
        }

---

## 十、实现进度追踪

### 10.1 已完成模块

| 模块 | 完成日期 | 测试状态 | 备注 |
|------|----------|----------|------|
| `constellation.py` | 2026-01-15 | ✅ | Walker星座模型 |
| `visibility.py` | 2026-01-15 | ✅ | 可见性计算 |
| `user.py` | 2026-01-15 | ✅ | 用户模型 |
| `task.py` | 2026-01-15 | ✅ | 任务模型 |
| `channel.py` | 2026-01-15 | ✅ | 信道模型，Ka频段 |
| `mec.py` | 2026-01-15 | ✅ | MEC计算模型 |
| `gym_env.py` | 2026-01-15 | ✅ | Gymnasium环境 |
| `graph/features.py` | 2026-01-15 | ✅ | 特征提取器 |
| `graph/builder.py` | 2026-01-15 | ✅ | 异质图构建器 |
| `model/layers.py` | 2026-01-15 | ✅ | 基础网络层 |
| `model/hetero_gnn.py` | 2026-01-15 | ✅ | HAN网络 |
| `model/actor.py` | 2026-01-15 | ✅ | 混合Actor |
| `model/critic.py` | 2026-01-15 | ✅ | 中心化Critic |
| `algorithm/buffer.py` | 2026-01-15 | ✅ | 经验缓冲区 |
| `algorithm/mappo.py` | 2026-01-15 | ✅ | MAPPO算法 |
| `algorithm/runner.py` | 2026-01-15 | ✅ | 训练运行器 |
| `scripts/train.py` | 2026-01-16 | ✅ | 完整训练脚本 |

### 10.2 待实现模块

| 模块 | 优先级 | 说明 |
|------|--------|------|
| `baselines/random_policy.py` | 中 | 随机策略基线 |
| `baselines/greedy_policy.py` | 中 | 贪婪策略基线 |
| `baselines/ddpg_offload.py` | 中 | DDPG对比算法 |
| `utils/logger.py` | 中 | TensorBoard日志 |
| `utils/metrics.py` | 中 | 评价指标计算 |
| `utils/visualization.py` | 低 | 结果可视化 |
| `scripts/evaluate.py` | 中 | 评估脚本 |
| `scripts/ablation.py` | 低 | 消融实验 |

### 10.3 测试运行结果

**算法测试 (`scripts/test_algorithm.py`):**
```
测试结果:
- 训练100个时间步
- 完成2个episode
- 平均奖励: ~8.86
- 训练速度: ~90 FPS
- 所有测试通过 ✓
```

**训练脚本测试 (`scripts/train.py`):**
```
测试配置:
- 用户数: 5
- 总步数: 1,000
- 每轮步数: 200

测试结果:
- HAN参数量: 571,264
- Actor参数量: 69,213
- Critic参数量: 184,833
- 训练速度: ~55 FPS
- 评估奖励: 33.12 ± 2.16
- 模型保存: best_model.pt, final_model.pt ✓
- 所有功能正常 ✓
```

---

## 十一、快速开始指南

### 11.1 环境配置

```bash
# 创建conda环境
conda create -n leo_switch python=3.10
conda activate leo_switch

# 安装依赖
pip install torch numpy gymnasium pyyaml
pip install torch-geometric  # 可选，用于图神经网络
```

### 11.2 快速训练

```bash
# 进入项目目录
cd LEO_switch

# 运行短期训练测试
python scripts/train.py --total_timesteps 2000 --n_steps 200 --num_users 5 --device cpu

# 运行完整训练
python scripts/train.py --total_timesteps 500000 --num_users 10
```

### 11.3 训练输出示例

```
2026-01-16 14:34:08 | INFO | 初始化环境...
[星座初始化] Walker 53.0:66/6/1
  - 轨道高度: 550.0 km
  - 轨道周期: 95.50 分钟
  - 总卫星数: 66
2026-01-16 14:34:08 | INFO |   - 观测维度: 73
2026-01-16 14:34:08 | INFO |   - 全局状态维度: 365
2026-01-16 14:34:08 | INFO | 初始化HAN编码器...
2026-01-16 14:34:08 | INFO |   - HAN参数量: 571,264
2026-01-16 14:34:08 | INFO | 初始化MAPPO...
2026-01-16 14:34:08 | INFO |   - Actor参数量: 69,213
2026-01-16 14:34:08 | INFO |   - Critic参数量: 184,833
2026-01-16 14:34:08 | INFO | ============================================================
2026-01-16 14:34:08 | INFO | 开始训练
2026-01-16 14:34:08 | INFO |   总步数: 1,000
2026-01-16 14:34:08 | INFO |   每轮步数: 200
2026-01-16 14:34:08 | INFO |   设备: cpu
2026-01-16 14:34:08 | INFO | ============================================================
2026-01-16 14:34:11 | INFO | Update    1 | Steps:     200 | Episodes:     1 | Reward:    37.35 | FPS:     57 | Actor Loss: 0.2726 | Critic Loss: 0.0380
2026-01-16 14:34:15 | INFO | Update    2 | Steps:     400 | Episodes:     2 | Reward:    37.70 | FPS:     55 | Actor Loss: 0.2552 | Critic Loss: 0.0451
...
2026-01-16 14:34:33 | INFO | 评估结果: 奖励 = 33.12 ± 2.16, 长度 = 200
2026-01-16 14:34:33 | INFO | 新的最佳奖励: 33.12
2026-01-16 14:34:33 | INFO | 模型已保存: results\models\best_model.pt
```

### 11.4 模型评估

```bash
# 加载并评估已保存的模型
python scripts/train.py --load_path results/models/best_model.pt --eval_only
```

---

## 十二、下一步计划

### 短期目标（1-2周）
1. ⬜ 实现对比算法基线（Random, Greedy, DDPG）
2. ⬜ 添加TensorBoard日志可视化
3. ⬜ 完善评估指标（切换成功率、QoS满足率等）

### 中期目标（1个月）
1. ⬜ 完成大规模训练实验
2. ⬜ 对比实验与结果分析
3. ⬜ 消融实验验证各模块贡献

### 长期目标（2-3个月）
1. ⬜ 撰写学术论文
2. ⬜ 整理开源代码
3. ⬜ 准备专利申请材料

