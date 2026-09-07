# 项目身份：主方法是 HAN+PDQN

用户于 2026-09-07 明确确认：本项目/论文的当前主方法（proposed method）是 **HAN+PDQN**，不是 HAN+MAPPO。

- 进入项目先读 `README.md` 和 `docs/PROJECT_STATUS.md`。分析结果、设计实验及撰写论文时，以 HAN+PDQN 为主方法。
- 训练入口 `scripts/train.py --algorithm pdqn` 对应 `HANPDQNTrainer`；结果方法键为 `han_pdqn`，图表名为 `HAN+PDQN`。
- 对比脚本中的 `pdqn` 是不带 HAN 的 PDQN 基线；不能与主方法 `han_pdqn` 混淆。HAN+MAPPO 是历史主方法、当前对比方法。
- 当前 HAN+PDQN 使用预训练且冻结的 HAN 和 replay buffer。新训练需要 `pretrained_han_path`，学习率为 `pdqn_lr`；不要误把 PPO 的 `learning_rate`、`n_steps`、`n_epochs` 当作 PDQN 参数。
- 历史结果的 `is_system` 记录当次运行角色，并不决定当前论文主方法。20260804 单种子目录中 HAN+PDQN 位于 `u*/learned_baselines/han_pdqn/`，应按 `method=han_pdqn` 读取其指标。
- 保留历史 CSV/JSON/checkpoint 和实验日志的真实算法身份；不要把 HAN+MAPPO 的数值或权重改名成 HAN+PDQN。重画历史图时另存输出。
- 主方法变更时，同步更新本文件、README、项目状态和实际默认入口；不要只改图例。
- 后续默认训练/多种子对比不运行 Attn+MAPPO、Joint Greedy、Random；它们已从两个 DEFAULT_BASELINES 中注释移除。保留实现和历史标识，除非用户显式要求，否则不要重新加入默认列表。SCA 通过独立评估入口运行。
