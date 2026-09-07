# 八方法对比结果（2026-09-08）

包含 HAN+PDQN（主方法）、PDQN、HAN+MAPPO、MAPPO、MADDPG、Min-Distance、Full-Local、SCA。
用户数为 20、25、30、35、40；训练种子为 42。

原七方法数值来自 `results/baseline_compare/multiuser_scaling_multiuser_single_seed_150k_20260804`，保持不变。
本次使用项目根目录 `han_pdqn/best_model_u{用户数}.pt`，在 CPU 上新评估 SCA。
每组对 HAN+PDQN 与 SCA 各评估 5 回合，每回合 512 步，测试种子 1000042 至 1000046。
五组配对原方法在相对/绝对容差 1e-6 内复现历史逐回合指标，环境配置完全一致。
`comparison_summary.json` 的 metadata_recovery 记录元数据来源与复现误差；checkpoint SHA256 见各组 sca/sca_manifest.json。

SCA 固定原策略的离散动作和本地/卸载模式，仅优化连续卸载比例。它不是独立训练的策略，没有训练收敛曲线。
在相同状态上固定离散规则不意味着两个闭环轨迹始终相同，连续动作会影响后续环境状态。

按用户要求，本次直接覆盖各用户目录和本目录的图表、CSV，共 34 张 PNG；没有另存带后缀的图。
原始七方法训练历史及模型权重未改动，验证结果见 verification.json。
训练收敛图仍只有五个学习方法；其余性能对比图包含八个方法。
单用户曲线使用 3 点滑动平均，跨用户收敛图使用原始回合均值。

重新绘图：

```bash
python scripts/run_multiuser_scaling_suite.py --results-root results/multiseed_0907 --seed 42 --aggregate-only
```

suite_manifest.json 记录绘图调用参数，实际训练参数查各算法的 training_history.json。
本组仍只有一个训练种子，五个测试回合不能替代多训练种子实验。
