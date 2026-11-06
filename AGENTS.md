# 项目身份：主方法是 HAN+PDQN

用户于 2026-09-07 明确确认：本项目/论文的当前主方法（proposed method）是 **HAN+PDQN**，不是 HAN+MAPPO。

- 进入项目先读 `README.md` 和 `docs/PROJECT_STATUS.md`。分析结果、设计实验及撰写论文时，以 HAN+PDQN 为主方法。
- 用户于 2026-10-08 指定论文 Overleaf 项目为 `https://www.overleaf.com/project/6a95399953316e3d09098b2f`，当前工作重点转为论文写作。续写前先读取现有稿件，沿用其模板、符号和已有内容；不可把未完成实验写成已完成。
- `paper/` 只收集计划插入论文的精选图片副本，不批量复制实验目录、单种子重复图或 CSV。按 `paper/README.md` 的论文图表安排筛选。现有及后续各类结果和图片保存路径保持原样；需要用于论文的图片生成后额外复制一份到 `paper/`，不移动原文件、不修改默认输出路径。
- 训练入口 `scripts/train.py --algorithm pdqn` 对应 `HANPDQNTrainer`；结果方法键为 `han_pdqn`，图表名为 `HAN+PDQN`。
- 对比脚本中的 `pdqn` 是不带 HAN 的 PDQN 基线；不能与主方法 `han_pdqn` 混淆。HAN+MAPPO 是历史主方法、当前对比方法。
- 2026-11-06 起，统一协议的补充实验使用 `scripts/run_matched_pdqn_suite.py`：`han_pdqn_joint` 为从随机初始化开始的 HAN+PDQN 联合训练，`pdqn_matched` 为设置匹配的无 HAN 对照，均不需要预训练。默认 U30、seed42–46，新结果另存 `results/multiseed_joint_20261106/`。新机制已实现并完成小规模验证，正式实验完成情况以清单为准。
- 现有 `multiseed_0907`、`dif_converge` 的 HAN+PDQN 及旧入口 `scripts/train.py --algorithm pdqn` 使用预训练且冻结的 HAN、编码观测 replay；该旧入口新训练仍需要 `pretrained_han_path`。保留其 `han_pdqn` 身份，不得改名为联合训练结果。学习率为 `pdqn_lr`，不要误把 PPO 的 `learning_rate`、`n_steps`、`n_epochs` 当作 PDQN 参数。进度见 `docs/EXPERIMENT_TODO.md`。
- 历史结果的 `is_system` 记录当次运行角色，并不决定当前论文主方法。20260804 单种子目录中 HAN+PDQN 位于 `u*/learned_baselines/han_pdqn/`，应按 `method=han_pdqn` 读取其指标。
- 保留历史 CSV/JSON/checkpoint 和实验日志的真实算法身份；不要把 HAN+MAPPO 的数值或权重改名成 HAN+PDQN。重画历史图时另存输出。
- 用户于 2026-10-29 明确确认：U20/seed43、U30/seed45 的两次复跑，是断电导致训练中断、数据异常后的修复。后续以修复后的有效结果开展分析，不再质疑这两次复跑，不将其归类为低分择优替换或选择性复跑；不将归档异常数据恢复进正式统计来否定当前结果。此前自动审阅中相反的推断已撤回，历史原始文件保留。
- 主方法变更时，同步更新本文件、README、项目状态和实际默认入口；不要只改图例。
- 后续默认训练/多种子对比不运行 Attn+MAPPO、Joint Greedy、Random；它们已从两个 DEFAULT_BASELINES 中注释移除。保留实现和历史标识，除非用户显式要求，否则不要重新加入默认列表。SCA 通过独立评估入口运行。
