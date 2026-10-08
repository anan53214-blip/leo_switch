# 30 用户 HAN+PDQN 超参数收敛实验

本目录包含服务器运行脚本；尚未训练新参数，尚无参数对比结果图。

## 推荐设计与论文依据

| 单因素实验 | 比较值 | 固定值 |
|---|---|---|
| PDQN 学习率 | 1e-4、3e-4、1e-3、3e-3 | batch size=512 |
| replay batch size | 128、256、512、1024 | pdqn_lr=1e-3 |

每组用 seed42–46。重复的默认组合只算一次，共 7 组、35 次运行。默认复用当前 `results/multiseed_0907/seed*/u30/learned_baselines/han_pdqn/training_history.json` 中的 5 条完整历史，包括已替换的 seed45/u30；新训练 30 次。使用 `--fresh-default` 可将默认组合也从头训练，共 35 次新训练。

这组值是围绕本项目默认 `(1e-3, 512)` 的建议，不是论文规定的唯一取值，不预设默认参数会最好。批大小是每次 replay 抽样的联合时隙转移数量，每条转移含 30 用户；不能直接与其他论文的单用户 batch 数字等同。

1. **Sensors 2023：Computation Offloading and Resource Allocation Based on P-DQN in LEO Satellite Edge Networks**。第 5.2 节、图 3/4 分别对学习率 `{1e-3,1e-4,1e-5}` 和批大小 `{32,64,128}` 绘制回合平均回报收敛曲线。它支持分别改变两个参数、观察学习速度和末期回报的设计；不意味着本项目最优参数也相同。来源：https://doi.org/10.3390/s23249885
2. **IJCAI 2019：Multi-Pass Q-Networks for Deep Reinforcement Learning with Parameterised Action Spaces**。第 5 节按数量级搜索 Q/参数网络学习率 `1e-1` 至 `1e-5`，批大小 `{32,64,128}`，参数搜索每组 5 个随机运行。Q 和参数网络可用不同学习率；本项目当前两个优化器共用 `pdqn_lr`，本实验保持该实现。来源：https://www.raillab.org/publication/bester-2019-multi/bester-2019-multi.pdf
3. **2025：Priority-aware task offloading for LEO satellite edge computing network: a multi-agent deep reinforcement learning-based approach**。第 5.2 节固定场景分别比较学习率、隐藏维度和 batch size，并使用统一滑动平均。其 Actor-Critic 参数不能直接替代 PDQN 的参数。来源：https://doi.org/10.1007/s44443-025-00160-w

## 在原 GPU 服务器上运行

将本目录两个文件放入完整仓库的 `results/dif_converge`，在仓库根目录激活原 `satellite.env` 环境后执行：

```bash
python -B results/dif_converge/run_dif_converge.py --dry-run
python -B results/dif_converge/run_dif_converge.py --device cuda
```

脚本优先读取现有训练配置记录的原预训练 HAN 文件。该文件不在当前服务器路径时，尝试仓库 `han_pdqn/best_model_u30.pt`，或显式指定：

```bash
python -B results/dif_converge/run_dif_converge.py --device cuda --pretrained-han-path /absolute/path/to/pretrained_han.pt
```

已有完整的新参数历史会核验后跳过。中断且不完整的运行会报错并保留现场，检查后移走该单次运行目录再重跑；不自动丢弃失败结果，不从没有 replay buffer 的检查点伪装成严格续训。

全部完成后，可只重画：

```bash
python -B results/dif_converge/run_dif_converge.py --plot-only
```

## 保持不变的条件

以当前 seed43/u30 的完整 `TrainConfig` 为模板，固定环境、奖励、HAN、网络结构、replay 容量、目标网络更新、探索日程和总步数，仅改变 `pdqn_lr` 或 `batch_size`。同一参数值在五个种子下使用同一冻结 HAN；每次 PDQN 从头初始化，绝不加载已训练控制器续训。其他方法不参与此图。

总步数 150000、每回合上限 512、warmup=1000、每 25000 步验证 5 回合。当前训练器要求 replay 至少有 `max(warmup_steps,batch_size)` 条转移，因此 B=1024 的首次更新比其他组晚 24 步；这一实际行为保持原样并在此记录。

主图纵轴为训练回合平均回报、横轴为环境训练步数；每个种子先用相同的 3 回合向后滑动平均，再计算五种子均值和 95% percentile bootstrap 区间（2000 次采样）。不完整末回合排除，不填补缺失种子，不用最佳 checkpoint 回报冒充整条收敛曲线。`--window 1` 可画未平滑曲线。

验证回报用于比较参数，沿用现有验证协议，不声称是独立测试表现。保存末 20 个完整训练回合平均回报和最终验证回报，用于辅助比较；不把单条曲线的平台期解释成理论收敛证明。

## 输出

- `convergence_u30.png`：双子图，左学习率、右 batch size。
- `convergence_u30.pdf`：同一图的论文矢量版。
- `convergence_seed_records.csv`：各参数/种子的原始训练与验证轨迹。
- `convergence_curves.csv`：绘图均值、置信区间与实际种子数。
- `convergence_summary.csv`：末期训练回报与最终验证回报。
- `experiment_manifest.json`：固定配置、HAN 权重指纹和每次运行状态。
- `lr*_bs*/seed*/`：必要的新训练历史、日志和 best/final 权重；不复制现有默认结果，不生成冗余中间 checkpoint。

当前已知原 GPU 记录中 30 用户、B=512 的 150000 步训练约需 2.1–2.2 小时；其他 batch size 的耗时须实际测量，不能直接用该时间保证总完成时间。
