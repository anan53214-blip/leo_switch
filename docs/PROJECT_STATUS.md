# 当前项目状态

2026-10-29 用户确认：U20/seed43、U30/seed45 的两次复跑用于修复断电导致的训练中断与数据异常。后续使用修复后的有效结果，不再将这两次复跑视为择优替换或提出同类质疑；此前自动审阅的相反推断已撤回。见根目录 `AGENTS.md` 和修订后的 `PAPER_REVIEW_20261029.md`。

> 2026-09-08 更新：多用户套件默认写入 `results/multiseed_0907/seed{seed}/u{num_users}`，主方法训练也在 `learned_baselines/han_pdqn` 下。`--with-sca` 可调用独立 SCA 评估并校验后汇总；旧 run-id 路径须用 `--legacy-layout`。完整命令见 [多种子目录说明](MULTISEED_LAYOUT.md)。

更新日期：2026-11-06。用户确认主方法为 **HAN+PDQN**。

## 当前执行重点（2026-11-06）

新增 `scripts/run_matched_pdqn_suite.py`，默认 U30、seed42–46，方法键为 `pdqn_matched`、`han_pdqn_joint`。两者统一 batch=512、PDQN 学习率=1e-3、梯度裁剪=0.5、25k 步验证、可行性掩码与时限截断规则。联合版保存原始图 replay，重算编码并更新 HAN，配套目标 HAN，不需要预训练。

新入口已实现，CPU 小规模训练/独立测试与关键回归检查通过；**尚未启动正式 150k 五种子训练**。新结果写入 `results/multiseed_joint_20261106/`。验证/测试场景独立；回合边界保存完整 replay、优化器、网络及随机状态。运行说明和分阶段清单见 [EXPERIMENT_TODO.md](EXPERIMENT_TODO.md)。

已核实旧 `multiseed_0907` 为 5 规模 × 7 方法 × 5 种子，共 175 条方法/种子记录；`dif_converge` 已完成冻结版 U30 的 7 配置 × 5 种子扫描，其中 30 次新运行、5 次默认配置复用。它们都不是联合训练结果。

2026-10-08：用户新建 `paper/`，仅收集计划插入论文的精选图片副本。参考 TWC、TMC、JSAC 的相关论文全文后，保留现有框架 PDF 及多种子收敛、核心性能、资源/切换指标三张汇总 PNG，删除此前复制的 322 张多余图片和来源 CSV。图表规划、文献依据及待补实验见 [论文插图说明](../paper/README.md)。现有及后续各种结果保存路径保持原样，不移动原文件或修改默认输出路径。

## 论文写作（2026-10-08）

用户指定的论文项目：[Overleaf](https://www.overleaf.com/project/6a95399953316e3d09098b2f)。远端访问受限是 10 月 8 日的历史记录；10 月 29 日已审阅本地导出的 12 页稿件，见 `PAPER_REVIEW_20261029.md`。当前六张精选 PDF 以 `paper/README.md` 为准。后续改稿先读取最新稿件、沿用模板与符号。本次新增代码和文档没有修改 Overleaf。

现稿和已有数值对应预训练冻结 HAN、PDQN 混合动作和编码观测 replay。联合版正式实验完成后再据实更新方法和图表，不能把新机制追溯套在旧数值上。待补消融、资源压力和泛化不写成已完成。

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
旧入口从检查点加载 HAN 并冻结，replay 存储编码后的观测；其历史结果不能描述为端到端联合训练。新入口实现从零联合训练，当前仅完成软件及小规模验证。
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
旧 SCA/比较入口默认评估 5 回合，seed 为训练 seed + 1,000,000 + 回合索引，验证/测试尚未分离。新入口已分离；历史基线仍需取回权重后统一补评，不能直接合并不同测试场景的数字。

SCA/MM 连续替换实现已完成，详见 [SCA_BASELINE.md](SCA_BASELINE.md)。使用独立配对评估入口，保留原离散规则和隐含本地/卸载模式。
2026-09-08 已使用用户补充的 `han_pdqn/best_model_u{20,25,30,35,40}.pt` 完成 seed42 的 SCA 配对评估（每组 5 回合）。五组原方法复现历史指标，SCA 已加入 `results/multiseed_0907/seed42` 并按用户要求覆盖现有图表。来源、权重哈希和核验见各组 `sca/sca_manifest.json`、`comparison_summary.json` 和根目录 `verification.json`。

1. 新协议已实现，正式匹配训练和独立测试待执行。
2. 旧 HAN+PDQN 及强基线五训练种子结果已存在；补新版本及受控对照，不把测试回合当成训练种子。
3. 优化类对比：固定离散策略、连续部分优化的替换实验，以及独立 LOCR 基线。
4. HAN 消融、切换与卸载联合决策消融、负载/资源压力、跨规模泛化及决策耗时。
5. 冻结版 `batch_size` 与 `pdqn_lr` 扫描已完成；如采用联合版主结果，再验证新版敏感性。PPO 超参数仅用于 PPO 基线。

完成状态以本页与 `EXPERIMENT_TODO.md` 为准；软件实现和短试跑不代表正式实验完成。主方法确认不改变旧数值，也不自动解决旧结果验证/测试复用问题。
