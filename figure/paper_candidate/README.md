# IEEE 双栏通栏候选图

两张图均为 **181.864 × 69.088 mm（7.16 × 2.72 in）**，每张横排三个子图；七个图例项单行排列，图例框距子图顶边约 5 pt。

| 图 | PNG 预览 | 矢量 PDF（论文优先使用） |
|---|---|---|
| 图 1：时延、违约率、每成功任务能耗 | [figure1_line_metrics.png](figure1_line_metrics.png) | [figure1_line_metrics.pdf](figure1_line_metrics.pdf) |
| 图 2：阻塞用户时间、总终端能耗、切换频率 | [figure2_bar_metrics.png](figure2_bar_metrics.png) | [figure2_bar_metrics.pdf](figure2_bar_metrics.pdf) |

- PNG：600 dpi，4296 × 1632 像素。PDF：直接矢量导出，嵌入 TrueType 字体，适合 Overleaf。
- Times New Roman；轴标签与子图标签 8.5 pt，刻度与共享图例 8 pt。
- 原有算法配色和标记；均值、95% CI 直接来自 `results/multiseed_0907/multiuser_summary.csv`。
- 五个训练种子 42–46；用户规模 20、25、30、35、40。保留原汇总中的两次复跑替换，不改变统计样本。
- 图 2(a) 沿用用户选择，排除 Full-Local；图 2(b)(c) 保留七种算法。所有柱状纵轴从零起。
- 时延为 ms，违约率为 %，每成功任务能耗为 J，累计阻塞时间为 user-s/episode，总终端能耗换算为 kJ/episode，切换频率为 handovers/user-minute。
- 每成功任务能耗的分子包含所有已记账终端能耗；五种子 CI 不表示逐任务分布。

## IEEEtran 插入示例

使用 `figure*` 与 `width=\textwidth`，不要另设 height 或使用 resizebox 强制改变纵横比。图号由 LaTeX 自动编号。

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figure1_line_metrics.pdf}
    \caption{Task-level performance versus the number of users:
    (a) average successful-task delay, (b) deadline violation rate,
    and (c) terminal energy per successful task.
    Shaded regions indicate 95\% bootstrap confidence intervals over five training seeds.}
    \label{fig:task_performance}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figure2_bar_metrics.pdf}
    \caption{Service and resource costs versus the number of users:
    (a) cumulative blocked user-time per episode, (b) total terminal energy per episode,
    and (c) handovers per user-minute. Full-Local is omitted from (a)
    and performs no handovers in (c).
    Error bars indicate 95\% bootstrap confidence intervals over five training seeds.}
    \label{fig:service_resource_costs}
\end{figure*}
```

## 复现

```powershell
python scripts/plot_paper_candidate_figures.py
```

原始数据、单指标图片和训练历史不变；本目录只保存候选成图与用途说明。
