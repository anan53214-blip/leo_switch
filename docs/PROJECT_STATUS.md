# 当前项目状态

> 2026-09-08 更新：多用户套件默认写入 `results/multiseed_0907/seed{seed}/u{num_users}`，主方法训练也在 `learned_baselines/han_pdqn` 下。`--with-sca` 可调用独立 SCA 评估并校验后汇总；旧 run-id 路径须用 `--legacy-layout`。完整命令见 [多种子目录说明](MULTISEED_LAYOUT.md)。

更新日期：2026-09-07。用户确认主方法为 **HAN+PDQN**。

## 方法身份与代码入口

| 角色 | 名称 | 入口/标识 |
|---|---|---|
| 当前论文主方法 | HAN+PDQN | `HANPDQNTrainer`；训练参数 `--algorithm pdqn`；结果键 `han_pdqn` |
| 无图消融/基线 | PDQN | 对比参数 `--baselines pdqn` |
| 连续求解对照（已实现） | SCA | `scripts/evaluate_sca_baseline.py`；结果键 `han_pdqn_sca` |
| 历史主方法、当前基线 | HAN+MAPPO | 训练参数 `--algorithm mappo`；对比键 `han_mappo` |
| 模型驱动独立基线设计 | LOCR | `基于Lyapunov的在线凸松弛基线算法说明.md`；不是主方法模块 |

核心实现：`scripts/train.py::HANPDQNTrainer`、`src/algorithm/pdqn.py`、`src/model/hetero_gnn.py`。
PDQN 的参数网络给出候选离散动作的连续卸载参数，Q 网络选择离散动作。
当前实现从检查点加载训练过的 HAN 并冻结，replay 存储编码后的观测；不能描述为端到端联合训练 HAN。
预训练开销、来源和种子划分需要在正式实验中记录。

## 已有结果如何读取

参考目录：`results/baseline_compare/multiuser_scaling_multiuser_single_seed_150k_20260804/`。
该目录覆盖 20/25/30/35/40 用户、训练种子 42、约 150k 步、每方法 5 个评估回合。
主方法结果读取 `multiuser_summary.csv` 中 `method=han_pdqn` 的行；训练历史位于
`u{用户数}/learned_baselines/han_pdqn/training_history.json`。
历史 `is_system=True` 指 HAN+MAPPO，保留原始记录，但不能据此认定当前论文主方法仍为 MAPPO。
仓库中是否存在可用权重，须实际检查，不能根据 Linux 历史路径推断本机权重存在。

## 下一阶段实验

后续默认训练、多用户和多种子实验已注释停用 `attn_mappo`、`joint_greedy`、`random`。
不删除历史结果或算法实现。多用户套件默认是 HAN+PDQN 加六个基线；SCA 另用独立入口配对评估。
SCA 与其他方法统一默认评估 5 回合，seed 规则为训练 seed + 1,000,000 + 回合索引。独立测试场景的划分仍需对所有方法一起实施。

SCA/MM 连续替换实现已完成，详见 [SCA_BASELINE.md](SCA_BASELINE.md)。使用独立配对评估入口，保留原离散规则和隐含本地/卸载模式。
2026-09-08 已使用用户补充的 `han_pdqn/best_model_u{20,25,30,35,40}.pt` 完成 seed42 的 SCA 配对评估（每组 5 回合）。五组原方法复现历史指标，SCA 已加入 `results/multiseed_0907/seed42` 并按用户要求覆盖现有图表。来源、权重哈希和核验见各组 `sca/sca_manifest.json`、`comparison_summary.json` 和根目录 `verification.json`。

1. 统一奖励/环境版本，拆开模型选择的验证场景与最终测试场景。
2. HAN+PDQN 及强基线的多训练种子实验；测试回合不能替代训练种子。
3. 优化类对比：固定离散策略、连续部分优化的替换实验，以及独立 LOCR 基线。
4. HAN 消融、切换与卸载联合决策消融、负载/资源压力、跨规模泛化及决策耗时。
5. PDQN 的 `batch_size` 与 `pdqn_lr` 敏感性和收敛性；PPO 超参数仅用于 PPO 基线。

上述是待办，不代表已经实施或完成。主方法确认不改变旧实验数值，也不自动解决验证/测试复用问题。
