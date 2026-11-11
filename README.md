# LEO 卫星网络切换与任务卸载联合优化（HAN+PDQN）

> 2026-11-11：统一入口升级为协议 v2，两方法共同默认 batch=128、每 4 环境步更新一次，并优化图打包、传输和计算复用。新增 `--benchmark` 短基准，每分钟报告训练速度与剩余时间估计。用户已暂停耗时较长的 v1，选择从零重训；v2 不读取 v1 恢复文件，正式新版结果仍待运行。

> 2026-11-06：新增 [统一协议入口](scripts/run_matched_pdqn_suite.py)，运行设置匹配的 PDQN 与从零联合训练 HAN+PDQN。默认 U30、五种子，验证/测试分离，支持完整回放断点恢复。已有论文数值仍来自冻结版；软件及小规模验证通过不代表正式补充实验完成。详见 [实验待办](docs/EXPERIMENT_TODO.md)。

> 2026-09-08 更新：多用户套件默认写入 `results/multiseed_0907/seed{seed}/u{num_users}`，主方法训练也在 `learned_baselines/han_pdqn` 下。`--with-sca` 可调用独立 SCA 评估并校验后汇总；旧 run-id 路径须用 `--legacy-layout`。完整命令见 [多种子目录说明](docs/MULTISEED_LAYOUT.md)。

**当前论文主方法：HAN+PDQN。** 用户已于 2026-09-07 确认。HAN+MAPPO 是历史主方法，现作为对比方法保留。

项目将异质图注意力网络（HAN）与参数化深度 Q 网络（PDQN）结合，联合决定用户的卫星切换和连续任务卸载比例。
后续会话先读 [项目状态](docs/PROJECT_STATUS.md)；助手的项目级约定在根目录 [AGENTS.md](AGENTS.md)。

## 方法与实现对应

| 方法 | 角色 | 训练/对比入口 |
|---|---|---|
| **HAN+PDQN** | **当前主方法** | `scripts/train.py --algorithm pdqn` → `HANPDQNTrainer`；结果键 `han_pdqn` |
| HAN+PDQN 联合训练版 | 新协议补充实验 | `scripts/run_matched_pdqn_suite.py`；`han_pdqn_joint`；无需预训练 |
| PDQN 匹配版 | 新协议无图对照 | 同一入口；`pdqn_matched` |
| PDQN | 无 HAN 基线 | `scripts/compare_system_baselines.py --baselines pdqn` |
| SCA | 复用主方法离散规则、优化连续比例的对照 | `scripts/evaluate_sca_baseline.py`；结果键 `han_pdqn_sca`；[说明](docs/SCA_BASELINE.md) |
| HAN+MAPPO | 历史主方法、当前基线 | `scripts/train.py --algorithm mappo`；对比键 `han_mappo` |
| MAPPO / MADDPG 等 | 对比方法 | 统一对比脚本中的对应 baseline |
| LOCR | 独立优化基线设计 | [算法说明](docs/基于Lyapunov的在线凸松弛基线算法说明.md) |

注意：训练入口的 `pdqn` 包含 HAN；对比列表的 `pdqn` 不包含 HAN，`han_pdqn` 才是主方法。

后续默认实验已停用 **Attn+MAPPO、Joint Greedy、Random**，实现保留用于显式调用和读取历史结果。
多用户/多种子套件默认比较 HAN+PDQN、HAN+MAPPO、MAPPO、MADDPG、PDQN、Min-Distance、Full-Local。
SCA 使用独立评估入口补充，不自动训练或混入旧结果。

## 当前算法流程

1. 环境生成用户、卫星、可见性、任务及资源竞争状态。
2. 构建异质图，用 HAN 提取用户表示并拼接任务与候选卫星信息。
3. PDQN 参数网络给出连续卸载参数，Q 网络评估并选择离散切换动作。
4. 原环境执行联合动作，计算任务时延、终端能耗、中断和奖励。
5. 使用 replay buffer 与 target networks 更新 PDQN。

**已有论文结果和旧入口使用预训练并冻结的 HAN。** 旧 replay 存储编码后的观测，旧结果不能描述为端到端联合训练。旧入口新训练需要包含 `han_state_dict` 且 schema 匹配的检查点，并记录预训练来源、种子及开销。

**新增联合训练入口从随机初始化开始。** replay 保存原始观测和图快照，抽样时重算 HAN；TD 损失与参数损失共同更新 HAN，配套目标 HAN。两种新方法共用 PDQN 更新、物理可行性掩码、奖励、截断和评估规则。新旧 checkpoint 格式与协议不同，使用各自入口，不直接混合统计。

## 运行入口

在项目根目录、已安装项目依赖的 Python 环境中执行。**当前补充实验使用新入口**，默认两种方法 × U30 × seed42–46，共 10 次训练，每次 150,000 步：

```bash
# 预览协议，不创建结果、不训练
python scripts/run_matched_pdqn_suite.py --dry-run
# 小规模真实环境验证，写入独立的 smoke_test 子目录
python scripts/run_matched_pdqn_suite.py --smoke-test --device cpu
# 先在实际服务器测速度，不启动正式实验、不保存训练权重
python scripts/run_matched_pdqn_suite.py --benchmark --device cuda
# 正式 GPU 实验，无需预训练权重
python scripts/run_matched_pdqn_suite.py --device cuda --output-dir results/multiseed_joint_20261111
# 中断后恢复完整 replay、优化器、目标网络和随机状态
python scripts/run_matched_pdqn_suite.py --device cuda --output-dir results/multiseed_joint_20261111 --resume
# 按预先规划扩展到五规模，校验并复用已完成的 U30
python scripts/run_matched_pdqn_suite.py --device cuda --output-dir results/multiseed_joint_20261111 --extend-suite --user-counts 20 25 30 35 40
```

默认路径仍为 `results/multiseed_joint_20261106/`，以上命令显式把 v2 另存 `results/multiseed_joint_20261111/`。两方法共同默认 batch=128、microbatch=128、每 4 环境步更新一次；PDQN 学习率=1e-3、HAN 学习率=1e-4、梯度裁剪=0.5。目标网络每 125 次更新同步，约对应 500 环境步；ε 衰减长度按更新频率换算，保持约 37,500 环境步的衰减尺度。验证仍为每 25k 步、5 个固定场景，独立测试仍为另 20 个场景。

`--benchmark` 测真实图样本的单次更新与完整环境步耗时，并报告 GPU 峰值显存；本地只能验证 CPU，服务器 3090 的实际速度需现场测量。显存不足可在基准和正式命令中同时设置 `--microbatch-size 32`，有效 batch 保持 128。降低 batch 和更新频率改变了学习预算，属于新实验协议，不保证性能与 v1 完全一致；只在相同设置下验证实现优化的数值一致性。恢复文件仍含原始图 replay，需留出内存与磁盘空间。

以下为**旧冻结版本入口**；预训练路径须替换为实际存在且版本匹配的检查点。旧检查点不含完整 replay，不是新入口的完整恢复格式。

```bash
# 当前主方法；省略 --algorithm 时也默认 pdqn
python scripts/train.py --algorithm pdqn --exp_name han_pdqn --pretrained_han_path results/han_encoder_pretrain/best_model.pt --pdqn_lr 0.001 --batch_size 512 --total_timesteps 150000 --save_path results/full_train_han_pdqn

# 主方法续训，编码器从该检查点本身恢复
python scripts/train.py --algorithm pdqn --load_path results/full_train_han_pdqn/final_model.pt --save_path results/full_train_han_pdqn --total_timesteps 300000

# 评估已有主方法，并运行基线
python scripts/compare_system_baselines.py --run-mode compare_only --system-run-dir results/full_train_han_pdqn --baselines pdqn han_mappo mappo_no_han maddpg min_distance full_local

# 新的多用户主方法实验；先预览命令
python scripts/run_multiuser_scaling_suite.py --run-id han_pdqn_multiuser --pretrained-han-path "results/han_pretrain_u{num_users}_seed{seed}/best_model.pt" --pdqn-lr 0.001 --dry-run
```

多用户脚本支持 `{num_users}` 和 `{seed}` 路径占位符；实际执行前需准备对应编码器检查点。
新训练结果目录采用 `full_train_han_pdqn_multiuser_u{用户数}_{run_id}`，避免与历史 MAPPO 目录混用。
若需用现有 MAPPO 训练器产生 HAN 预训练检查点，可显式运行 `--algorithm mappo --exp_name han_encoder_pretrain --save_path results/han_encoder_pretrain`，并单独计入预训练成本。

| 参数 | 含义 |
|---|---|
| `pdqn_lr` / `--pdqn-lr` | PDQN Q 网络与参数网络学习率，默认 `1e-3` |
| `batch_size` | replay 训练批大小，默认 `512` |
| `pretrained_han_path` | 新训练所需的预训练 HAN 检查点 |
| `learning_rate`、`n_steps`、`n_epochs` | PPO 路径参数，不用于替代 PDQN 学习率 |

## 历史结果与实验口径

SCA 已有独立评估入口：`python scripts/evaluate_sca_baseline.py --checkpoint <HAN+PDQN权重路径>`。
只创建新输出目录，配对评估原方法和 SCA，不训练或覆盖旧权重。默认评估 5 回合，测试 seeds 与统一对比脚本对齐；
不能直接把新结果并入不同测试场景的旧表。参数和模型近似见 [SCA 说明](docs/SCA_BASELINE.md)。

已有参考实验在 `results/baseline_compare/multiuser_scaling_multiuser_single_seed_150k_20260804/`。
读取当前主方法时选择汇总表的 **`method=han_pdqn`**，其训练历史位于 `u*/learned_baselines/han_pdqn/`。
这批文件的 `is_system=True` 是历史 HAN+MAPPO 运行角色；原始文件保持原样，不用于重新定义当前主方法。

环境/奖励以运行保存的 `env_config`、`training_history.json` 和代码版本为准。
当前默认时延/能耗权重为 `0.60/0.40`，可达 MEC 公平性奖励权重为 `0.05`。
状态、动作维度、奖励细节和历史机制见 [系统架构说明](docs/系统架构算法智能体与奖励函数完整说明.md)。
已有五种子主比较和冻结版超参数扫描已完成；新协议正式训练、独立测试、消融和泛化按 [实验待办](docs/EXPERIMENT_TODO.md) 推进。

## 项目文件

- `paper/`：仅保留计划插入论文的精选图片副本，图表安排及参考文献见 [论文插图说明](paper/README.md)；现有及后续结果保存路径保持原样，不批量复制结果目录或 CSV。
- `scripts/train.py`：当前主方法及其他 HAN 训练器。
- `scripts/run_matched_pdqn_suite.py`：新协议的匹配 PDQN、从零联合训练 HAN+PDQN、独立测试、汇总及绘图。
- `src/algorithm/pdqn.py`：PDQN Q 网络、连续参数网络、回放更新。
- `src/model/hetero_gnn.py`、`src/graph/`：HAN 与异质图构建。
- `src/environment/`：轨道、信道、队列、任务和环境。
- `scripts/compare_system_baselines.py`：统一评估及基线。
- `scripts/run_multiuser_scaling_suite.py`：多用户、多种子实验与汇总。
- `docs/COMPARE_SYSTEM_BASELINES_CLI.md`、`docs/MULTIUSER_AGGREGATE_PLOTTING.md`：命令说明。
- `tests/`：环境、算法与实验入口回归检查。
