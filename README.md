# LEO 卫星网络切换与任务卸载联合优化（HAN + MAPPO）

基于**异质图注意力网络 (HAN)** 和**多智能体近端策略优化 (MAPPO)** 的 LEO 卫星网络切换与计算卸载联合优化系统。

---

## 项目结构

```
LEO_switch/
├── config/
│   └── （当前无运行时 YAML 配置）
├── docs/
│   ├── SYSTEM_ARCHITECTURE.md      # 完整技术文档（与代码逐行校对）
│   ├── TRAINING_GUIDE.md           # 训练使用方法
│   └── research_plan.md            # 研究计划
├── src/
│   ├── environment/                # 仿真环境
│   │   ├── constellation.py        #   Walker 星座 (6×11=66 颗卫星)
│   │   ├── channel.py              #   Ka 频段信道模型 (20 GHz)
│   │   ├── mec.py                  #   MEC 服务器 & 任务队列
│   │   ├── user.py                 #   地面用户 & 状态机
│   │   ├── task.py                 #   可拆分计算任务
│   │   ├── visibility.py           #   星地可见性计算
│   │   └── gym_env.py              #   Gymnasium RL 环境封装
│   ├── graph/                      # 异质图
│   │   ├── features.py             #   节点/边特征提取
│   │   └── builder.py              #   异质图构建器
│   ├── model/                      # 神经网络
│   │   ├── hetero_gnn.py           #   HAN 编码器 (571K 参数)
│   │   ├── actor.py                #   HybridActor (Categorical + Normal)
│   │   ├── critic.py               #   CentralizedCritic
│   │   └── layers.py               #   MLP / GAT / SemanticAttention
│   └── algorithm/                  # RL 算法
│       ├── mappo.py                #   MAPPO (PPO-Clip + GAE)
│       ├── buffer.py               #   多智能体 Rollout Buffer
│       └── replay_buffer.py        #   Off-policy 经验回放
├── scripts/
│   ├── train.py                    # 训练入口
│   ├── compare_system_baselines.py  # 系统与基线统一对比
│   ├── plot_training_artifacts.py   # 从训练/对比产物重画图表
│   └── run_multiuser_scaling_suite.py # 多用户扩展聚合
├── tests/                          # 单元测试
└── results/                        # 训练输出 (模型/日志/图表)
```

---

## 核心设计

### 智能体
- **每个地面用户是一个独立智能体**，所有用户共享 Actor 参数
- 训练范式：**CTDE**（集中训练，分布式执行）

### 状态表示 (69 维)
环境原始状态 → 异质图 → **HAN 编码** → 拼接额外特征：

| 组成部分 | 维度 | 来源 |
|----------|------|------|
| HAN 节点嵌入 | 64 | 用户节点 13 维 + 卫星节点 10 维 → HAN 编码 |
| RVT 预警信号 | 1 | 剩余可见时间是否低于阈值 |
| 任务特征 | 4 | 数据量、计算量、时延要求、类型 |

### 动作空间（混合）
| 动作 | 类型 | 分布 | 含义 |
|------|------|------|------|
| 切换决策 | 离散 | Categorical(K+1) | 0 = 不切换，1~K = 切换到第 k 个可见卫星 |
| 卸载比例 | 连续 | Normal(μ, σ) | λ ∈ [0,1]，0 = 全本地，1 = 全卸载 |

### 奖励函数
- **任务奖励**：deadline 内完成时为 `1 - 0.60×时延比例 - 0.10×能耗比例`；超时或最终失败固定为 `-1`
- **连接惩罚**：按每个用户在时隙内的实际服务中断比例处罚，完整中断一时隙最多 `-0.30`
- **切换失败**：固定 `-0.20`；成功切换不额外奖励，只计算实际切换中断
- **延迟发放**：MEC 任务完成后通过 `pending_rewards` 在后续步发放
- **全局奖励** = mean(所有用户奖励)

负载均衡、队列占用和切换次数只作为评价指标，不再重复进入 reward。完整公式和论文依据见
[Reward 函数设计](docs/REWARD_WEIGHT_CONFIG.md)。

### 竞争机制
- 多用户共享卫星 MEC 队列 → CPU 时间片均分 → 用户越多每人分到越少
- HAN 通过 User→Sat→User 元路径编码用户间资源竞争关系

---

## 快速开始

```bash
# 安装依赖
conda create -n satellite python=3.10 -y && conda activate satellite
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy matplotlib gymnasium pyyaml

# 快速验证
python scripts/train.py --total_timesteps 100000 --max_steps 512 --eval_episodes 1

# 标准训练（不传参数时也采用这组训练规模）
python scripts/train.py --total_timesteps 150000 --max_steps 512 --n_steps 1024 --batch_size 512

# 生成图表
python scripts/plot_training_artifacts.py --comparison-summary results/baseline_compare/<run_id> --output-dir results/baseline_compare/<run_id>/replot
```

详见 [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md)。

---

## 模型规模

| 组件 | 参数量 | 结构 |
|------|--------|------|
| HAN 编码器 | 571K | 2 层 × 4 头 × 3 元路径，64 维输出 |
| Actor | 68K | MLP(69→256→128) + 离散头 + 连续头 |
| Critic | 183K | 用户编码 + 卫星编码 + MLP → V(s) |
| **总计** | **~822K** | |

---

## 参考论文
1. 宋晓勤等，基于深度确定性策略梯度的星地融合网络可拆分任务卸载算法
2. 付一阳等，星地融合网络中基于异质图表征的多智能体协作切换方法
