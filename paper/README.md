# 论文精选插图

更新与核对日期：2026-10-25。当前论文主方法为 **HAN+PDQN**。

2026-11-06 补充：当前数值 PDF 仍对应预训练冻结 HAN 版本。新增 `scripts/run_matched_pdqn_suite.py` 用于匹配 PDQN 与从零联合训练 HAN+PDQN；正式补充实验尚未完成，本目录没有替换新版结果。新协议图先保存在新结果目录，确认用于论文后再额外复制精选文件到这里。见 [实验待办](../docs/EXPERIMENT_TODO.md)。

本目录当前包含 **6 张单页 PDF**、[论文大纲 OUTLINE.md](OUTLINE.md) 和本说明。只保存计划用于论文的精选插图与写作说明；实验 CSV、训练历史、PNG 预览、生成清单和临时渲染文件保留在原目录，不复制到这里。

## 1. 当前文件清单

表中的论文位置是建议位置，最终图号由 LaTeX 自动编号，文件名中的 figure1/figure2 不代表论文最终图号。

| 文件 | 内容与建议位置 | 实际画布尺寸（宽 × 高） | 插入方式 |
|---|---|---|---|
| [系统架构图.drawio.pdf](系统架构图.drawio.pdf) | 系统模型：LEO 卫星、地面用户、服务链路、星间链路与覆盖范围 | 66.717 × 66.717 mm | 建议单栏，按列宽等比例缩放 |
| [算法框架图10_23.pdf](算法框架图10_23.pdf) | 方法章节：环境、异质图/HAN、用户观测融合、PDQN 参数网络与 Q 网络、联合动作执行 | 252.222 × 125.984 mm | 建议通栏，按正文宽度等比例缩放并检查小字 |
| [algorithm_convergence_u30.pdf](algorithm_convergence_u30.pdf) | 性能评估：30 用户下不同学习算法的训练回报收敛 | 88.9 × 71.628 mm | IEEE 单栏 |
| [convergence_u30.pdf](convergence_u30.pdf) | 超参数分析：HAN+PDQN 的学习率与 replay batch size 敏感性；上下两个子图 | 88.9 × 121.92 mm | IEEE 单栏 |
| [figure1_line_metrics.pdf](figure1_line_metrics.pdf) | 核心任务性能：成功任务平均时延、截止时间违约率、每成功任务能耗；横排三个折线子图 | 181.864 × 69.088 mm | IEEE 双栏通栏 |
| [figure2_bar_metrics.pdf](figure2_bar_metrics.pdf) | 服务与资源代价：阻塞用户时间、总终端能耗、切换频率；横排三个柱状子图 | 181.864 × 69.088 mm | IEEE 双栏通栏 |

当前目录已不包含旧版 `算法框架图.pdf`、`multiuser_reward_convergence.png`、`multiuser_core_metrics.png`、`multiuser_resource_metrics.png`，不再把它们列为本目录保留文件。相关历史结果仍按原路径管理。

## 2. 数值结果图的内容与口径

2026-10-25：两张通栏性能图的七项顶部图例改为单行，图例框距子图顶边约 5 pt；保留子图实际尺寸与原有字号，将画布高度从 76.2 mm 缩短至 69.088 mm。均值、置信区间及源数据指纹与调整前一致。

### 2.1 核心性能折线图：figure1_line_metrics.pdf

- (a) `avg_success_delay`：仅对成功任务统计的平均时延，图中单位为 **ms**。
- (b) `deadline_violation_rate`：截止时间违约任务数占生成任务总数的比例，图中单位为 **%**。
- (c) `energy_per_successful_task`：全部已记账终端能耗除以成功任务数，图中单位为 **J/成功任务**；不是只对成功任务自身能耗求平均。
- 横轴为用户数 **20、25、30、35、40**。七种方法为 HAN+PDQN、HAN+MAPPO、MAPPO、MADDPG、PDQN、Min-Distance、Full-Local。
- 曲线为训练种子 **42–46** 的均值，阴影为训练种子层面的 **95% bootstrap 置信区间**。每个训练种子的评估结果先按其 5 个回合汇总。

### 2.2 服务与资源柱状图：figure2_bar_metrics.pdf

- (a) `blocked_user_seconds`：每评估回合的累计阻塞用户时间，单位 **user-s/episode**；按用户要求，仅比较除 Full-Local 以外的六种方法。
- (b) `total_energy`：每评估回合的总终端能耗；原始数据为 J，图中换算为 **kJ/episode**。
- (c) `handovers_per_user_minute`：每用户每分钟切换次数，单位 **handovers/user-minute**。
- (b)(c) 保留七种方法；Full-Local 不执行切换，因此 (c) 中为零。所有柱状纵轴从零起，误差棒为五训练种子的 95% bootstrap 区间。
- 已删除图底部额外小字。Full-Local 的省略范围和误差棒含义应在论文 caption 中说明，不再添加图内脚注。
- 累计阻塞用户时间会受用户数和观察时间影响；跨规模分析时不能把其增幅直接等同于阻塞比例的增幅。切换频率用于解释策略代价，不应要求它必然最低。

这两张图均来自 [当前多种子汇总数据](../results/multiseed_0907/multiuser_summary.csv)，逐训练种子记录见 [multiuser_seed_records.csv](../results/multiseed_0907/multiuser_seed_records.csv)。SCA 不在这两张五种子主对比图中。

### 2.3 不同算法收敛图：algorithm_convergence_u30.pdf

- 图例保留三列两行，按绘图区水平居中，图例框距绘图区顶边约 5 pt；绘图区实际尺寸和字号保持原样。
- 正文选用 **U30**，展示 HAN+PDQN、HAN+MAPPO、MAPPO、MADDPG、PDQN 五种学习方法。
- 横轴为 **Training episodes**，纵轴为 **Mean episode reward**。图中的“30 users”标题已删除，用户数量在正文或 caption 中说明。
- 每回合为 **512 环境步**。仅使用预算内完整回合：HAN+PDQN、PDQN、MADDPG 的位置为 1–292；PPO 类保持每两回合记录一次的原生位置 2、4、…、292，不插造逐回合值。
- 五个训练种子为 42–46；保留原始回合回报，不额外平滑。阴影为 2,000 次 percentile bootstrap 得到的逐点 95% 区间。
- 完整五规模收敛图适合补充材料，不作为正文单栏图。原始 steps 版本见 [多规模收敛原图](../results/multiseed_0907/multiuser_reward_convergence.png)，已有 episode 版本见 [五规模训练回报图](../figure/training/mean_reward.png)。本图本身不用于证明跨规模泛化。

### 2.4 超参数图：convergence_u30.pdf

- 同为 **U30、HAN+PDQN**；上、下两个子图均以 episode 为横轴、回合平均回报为纵轴。
- (a) 学习率为 **1e-4、3e-4、1e-3、3e-3**，固定 replay batch size=512。
- (b) replay batch size 为 **128、256、512、1024**，固定 `pdqn_lr=1e-3`。两图共同的默认组合只对应一组实验，共 7 种唯一配置、每组 5 个训练种子。
- 横轴由 `episode=steps/512` 换算，完整回合为 1–292；保留原来的 **3 回合向后移动平均**和五种子 **95% bootstrap 区间**。
- 固定预训练并冻结的 HAN，改变 PDQN 学习率或 replay batch size。这里的学习率是 `pdqn_lr`，不是 PPO 的 `learning_rate`。
- 数据和运行说明见 [dif_converge 目录](../results/dif_converge/README.md)及 [实验清单](../results/dif_converge/experiment_manifest.json)。该实验与成图已有数据，不再列为待完成实验。

两张收敛图说明经验训练动态与参数敏感性；图注应交代用户规模、训练种子、平滑方法和区间含义。它们不构成理论收敛证明，也不自动证明某个默认参数严格最优。

## 3. 来源、预览与复现

四张数值 PDF 的 PNG 预览、对应 PDF 和生成清单均保留在 [figure/paper_candidate](../figure/paper_candidate/README.md)，`paper/` 中只放精选 PDF 副本。

| 本目录 PDF | 对应生成脚本 | 来源与核对记录 |
|---|---|---|
| `figure1_line_metrics.pdf`、`figure2_bar_metrics.pdf` | [plot_paper_candidate_figures.py](../scripts/plot_paper_candidate_figures.py) | [figure_manifest.json](../figure/paper_candidate/figure_manifest.json)；数据在 `results/multiseed_0907/` |
| `algorithm_convergence_u30.pdf` | [plot_paper_algorithm_convergence.py](../scripts/plot_paper_algorithm_convergence.py) | [algorithm_convergence_u30_manifest.json](../figure/paper_candidate/algorithm_convergence_u30_manifest.json)；读取 U30 五方法、五种子的训练历史 |
| `convergence_u30.pdf` | [plot_paper_hyperparameter_convergence.py](../scripts/plot_paper_hyperparameter_convergence.py) | [convergence_u30_manifest.json](../figure/paper_candidate/convergence_u30_manifest.json)；原始曲线数据在 `results/dif_converge/` |

本次核对结果：六份 PDF 均可读取且各为单页；四张数值 PDF 与候选目录对应副本的 SHA256 一致，三个生成清单登记的源文件指纹也与当前文件一致。两张示意图以本目录现存 PDF 为当前版本，未在本目录保留对应的可编辑 `.drawio` 源文件。

在项目根目录、已安装项目绘图依赖的 Python 环境中，可分别运行：

```powershell
python scripts/plot_paper_candidate_figures.py
python scripts/plot_paper_algorithm_convergence.py
python scripts/plot_paper_hyperparameter_convergence.py
```

三条命令分别生成两张通栏性能图、U30 算法收敛图、U30 超参数图。第一条将通栏图输出到 `figure/paper_candidate/`，更新后需额外复制选定 PDF 到本目录；后两条会同时同步各自的精选 PDF。原始实验目录和原始图不移动、不改默认保存路径。

## 4. IEEEtran 插入方式

通栏图使用 `figure*` 与 `width=\textwidth`；单栏图使用 `figure` 与 `width=\columnwidth`。只设置宽度，保持纵横比，不额外指定 height 强制拉伸。

```latex
% 双栏通栏：也适用于 figure2_bar_metrics.pdf
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figure1_line_metrics.pdf}
    \caption{Task performance versus the number of users.}
    \label{fig:task_performance}
\end{figure*}

% 单栏：也适用于 convergence_u30.pdf
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{algorithm_convergence_u30.pdf}
    \caption{Training convergence of the learning algorithms with 30 users.}
    \label{fig:algorithm_convergence}
\end{figure}
```

上面只演示排版，正式 caption 还应补上各子图含义、五训练种子以及 95% 区间；超参数图还需说明 3 回合平滑。两张示意图分别可按单栏和通栏宽度缩放，最终字号需在实际论文页面中检查。

## 5. 示意图仍需核对的标注

本次只更新说明，不修改现有 PDF。

- `算法框架图10_23.pdf` 仍可见 **Heterogeous** 拼写，应改为 **Heterogeneous**；不要因文件名更新就认为旧拼写已修复。
- 当前框架图侧重在线编码、决策与动作执行，未显式标注 HAN“预训练并冻结”，也未完整画出经验回放和目标网络。方法文字/caption 应说明真实训练机制，不能把它描述为端到端联合训练。
- 图中卸载比例使用 `lambda` 符号；定稿时与正文统一。若正文采用 `rho`，应同步统一图内符号。
- `系统架构图.drawio.pdf` 已存在，展示网络与覆盖关系；不再记为“系统图待绘制”。

## 6. 本目录尚未收录的验证图

- 完整 HAN/关系设计消融与固定切换、固定卸载比例等联合决策对照。
- 任务负载、带宽、MEC 算力等资源敏感性，以及跨规模泛化、在线推理耗时。
- 任务级时延 CDF：需要逐任务时延样本，不能由几个种子的均值或 P95 反推。

以上表示本目录尚未收录相应图，不据此断言其他实验目录没有结果。准备写入论文前，先核对实际数据和评估设置；不把未核实实验写成已完成。HAN+PDQN 与 PDQN 的比较有助于检验 HAN，但不能替代全部联合决策消融。

## 7. 图表组织参考

既有图表安排参考了同领域论文，作为组织方式的参考，不是期刊规定的固定图数或图型：

- [Wei et al., IEEE TWC 2023](https://arxiv.org/pdf/2210.00268)：系统与算法框架、收敛、任务时延及联合决策对照分别展示。
- [Shen et al., IEEE TMC 2024](https://shen-hang.github.io/files/TMC2024.pdf)：同时使用完成率柱状图、随资源/负载变化的性能折线和时延分布图。
- [Huang et al., IEEE JSAC 2024](https://arxiv.org/pdf/2401.01140)：能耗和时延随用户数、任务及计算资源参数变化的折线。

后续只在确定图对应的论文论点后加入精选副本；不向本目录批量复制结果、单种子重复图、CSV、临时预览或生图中间文件。
