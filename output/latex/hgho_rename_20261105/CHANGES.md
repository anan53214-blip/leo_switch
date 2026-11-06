# HGHO 文字改名清单

依据：用户于 2026-11-05 粘贴的当前 LaTeX。桌面 leo_switch.pdf 为此前编译版本，12 页，部分段落与当前源码不同；以下行号均对应本次粘贴源码。修改版保持原行数，便于逐处核对。

命名：HGHO = Heterogeneous Graph-based Joint Handover and Offloading。中文：基于异质图的联合切换与卸载框架。HAN 与 PDQN 继续作为组成模块名，单独的 PDQN 仍是基线。

## 已完成修改

- HAN--PDQN → HGHO：33 处。
- HAN+PDQN → HGHO：2 处。
- 第 54 行摘要首次出现、第 174–176 行正文首次出现：补全 HGHO 英文展开，并说明由 HAN 和 PDQN 组成。
- 第 54 行：将 networks.Simulations 改为 networks. Simulations，补上原来缺失的句间空格。

| 位置 | 原名称出现次数 | 源码行号 |
|---|---:|---|
| 摘要 | 3 | 54 |
| Main Contributions | 2 | 174, 221 |
| Problem Formulation | 1 | 797 |
| Proposed HAN--PDQN Framework | 1 | 799 |
| Framework Overview | 2 | 806, 836 |
| Training Procedure | 1 | 1371 |
| Experimental Setup and Evaluation Protocol | 4 | 1461, 1541, 1577, 1612 |
| Training Convergence and Hyperparameter Sensitivity | 6 | 1705, 1707, 1709, 1714, 1718, 1747 |
| Task Performance under User Scaling | 8 | 1777, 1782, 1785, 1795, 1799, 1806, 1811, 1831 |
| Service Continuity and Operating Costs | 5 | 1850, 1865, 1881, 1884, 1897 |
| Conclusion | 2 | 1914, 1924 |

## 关键替换

摘要首次定义：
```latex
To solve this problem, we propose a heterogeneous graph-based joint handover and offloading (HGHO) framework, which combines a heterogeneous graph attention network (HAN) with a parameterized deep Q-network (PDQN).
```

引言 Main Contributions 开头：
```latex
To address this problem, we develop a heterogeneous graph-based
joint handover and offloading (HGHO) framework, which integrates
a heterogeneous graph attention network (HAN) with a parameterized deep Q-network (PDQN).
```

第四章标题：
```latex
\section{Proposed HGHO Framework}
```

框架图图注（只改 LaTeX 图注，本轮未编辑图片）：
```latex
\caption{HGHO framework for joint handover and partial
    task offloading.}
```

算法标题：
```latex
\caption{HGHO Training}
```

训练参数表对应行：
```latex
        HGHO / PDQN replay batch size & $512/128$ \\
```

## 保留与可选项

- 原第 48 行标题未含旧方法名，本版保留。若希望在标题中显示系统名，可选用：
```latex
\title{HGHO: Heterogeneous Graph-Based Joint Handover and Partial Task Offloading in LEO Satellite Edge Networks}
```
- 关键词不含旧方法名，无需为改名调整。
- 单独的 HAN、PDQN，以及基线 HAN+MAPPO、MAPPO、MADDPG、Min-Distance、Full-Local 保留。
- 第四章的 PDQN-Based Control、PDQN-Based Decision 等模块标题保留。
- fig:han_pdqn_framework、alg:han_pdqn_training 以及相应引用是内部定位键，不会显示在正文，本版保留，以免无谓修改引用。
- 公式、符号、实验数值、文献引用、图像文件路径与插图参数全部保留。
- 数值图现有图注不含旧方法名，无需仅为改名修改；框架图图注已随正文统一。

## 下一轮图内文字

PDF 第 9、10、11 页的主方法图例仍写 HAN+PDQN，对应算法收敛图及两张性能图。这些文字位于独立图文件中，本轮未改。
框架图内部的 HAN、PDQN 如果指各自模块，应保留；只有指整套方法的旧名称才需要改为 HGHO。

## 校验

35 处显示名称均已统一为 HGHO，原行数保持；数值序列、公式、引用、内部标签及图像引用保持一致。
此目录只提供文字修改后的完整源码与清单，未编译新的 PDF，也未修改 Overleaf。

## 逐行对照

### 第 54 行

原文：
```latex
Integrating low Earth orbit (LEO) satellites with edge computing extends computation services to areas with limited terrestrial coverage and to terminals with constrained local resources. However, satellite motion continually changes link quality, satellite visibility, and remaining service time, while multiple users compete for limited onboard communication and computing resources. For divisible tasks, a user must select a serving satellite and determine a continuous offloading ratio matched to that satellite's link, resource, and contention conditions. We therefore formulate joint serving-satellite selection and partial task offloading as a sequential decision problem with a parameterized hybrid action space. To solve this problem, we propose HAN--PDQN, which combines a heterogeneous graph attention network (HAN) with a parameterized deep Q-network (PDQN). A heterogeneous graph distinguishes user and satellite nodes as well as serving and visibility relations. Relation-specific meta-path aggregation and hierarchical attention capture established resource sharing and potential contention among users with overlapping satellite alternatives. Conditioned on this relational representation, local task information, and candidate-satellite features, PDQN generates a continuous offloading ratio for each feasible satellite action and estimates the long-term value of the resulting satellite--ratio pair. The same contention context therefore affects both how much workload is assigned to each candidate and whether that candidate is selected. HAN--PDQN consequently supports coordinated serving-satellite selection and partial task offloading in dynamic multi-user LEO satellite edge networks.Simulations across five training seeds and network sizes of 20--40 users show that, compared with PDQN, HAN--PDQN reduces average successful-task delay, energy per successful task, and the deadline violation rate by 2.16\%, 5.72\%, and 21.26\%, respectively, while increasing the task success rate by 2.12\%.
```
修改后：
```latex
Integrating low Earth orbit (LEO) satellites with edge computing extends computation services to areas with limited terrestrial coverage and to terminals with constrained local resources. However, satellite motion continually changes link quality, satellite visibility, and remaining service time, while multiple users compete for limited onboard communication and computing resources. For divisible tasks, a user must select a serving satellite and determine a continuous offloading ratio matched to that satellite's link, resource, and contention conditions. We therefore formulate joint serving-satellite selection and partial task offloading as a sequential decision problem with a parameterized hybrid action space. To solve this problem, we propose a heterogeneous graph-based joint handover and offloading (HGHO) framework, which combines a heterogeneous graph attention network (HAN) with a parameterized deep Q-network (PDQN). A heterogeneous graph distinguishes user and satellite nodes as well as serving and visibility relations. Relation-specific meta-path aggregation and hierarchical attention capture established resource sharing and potential contention among users with overlapping satellite alternatives. Conditioned on this relational representation, local task information, and candidate-satellite features, PDQN generates a continuous offloading ratio for each feasible satellite action and estimates the long-term value of the resulting satellite--ratio pair. The same contention context therefore affects both how much workload is assigned to each candidate and whether that candidate is selected. HGHO consequently supports coordinated serving-satellite selection and partial task offloading in dynamic multi-user LEO satellite edge networks. Simulations across five training seeds and network sizes of 20--40 users show that, compared with PDQN, HGHO reduces average successful-task delay, energy per successful task, and the deadline violation rate by 2.16\%, 5.72\%, and 21.26\%, respectively, while increasing the task success rate by 2.12\%.
```

### 第 174 行

原文：
```latex
To address this problem, we develop HAN+PDQN, which
```
修改后：
```latex
To address this problem, we develop a heterogeneous graph-based
```

### 第 175 行

原文：
```latex
integrates a heterogeneous graph attention network (HAN)
```
修改后：
```latex
joint handover and offloading (HGHO) framework, which integrates
```

### 第 176 行

原文：
```latex
with parameterized reinforcement learning.
```
修改后：
```latex
a heterogeneous graph attention network (HAN) with a parameterized deep Q-network (PDQN).
```

### 第 221 行

原文：
```latex
HAN+PDQN framework and its training and execution
```
修改后：
```latex
HGHO framework and its training and execution
```

### 第 797 行

原文：
```latex
the HAN--PDQN framework developed next.
```
修改后：
```latex
the HGHO framework developed next.
```

### 第 799 行

原文：
```latex
\section{Proposed HAN--PDQN Framework}
```
修改后：
```latex
\section{Proposed HGHO Framework}
```

### 第 806 行

原文：
```latex
HAN--PDQN, which connects a representation of multi-user
```
修改后：
```latex
HGHO, which connects a representation of multi-user
```

### 第 836 行

原文：
```latex
    \caption{HAN--PDQN framework for joint handover and partial
```
修改后：
```latex
    \caption{HGHO framework for joint handover and partial
```

### 第 1371 行

原文：
```latex
\caption{HAN--PDQN Training}
```
修改后：
```latex
\caption{HGHO Training}
```

### 第 1461 行

原文：
```latex
We evaluate HAN--PDQN in a multi-user LEO satellite edge
```
修改后：
```latex
We evaluate HGHO in a multi-user LEO satellite edge
```

### 第 1541 行

原文：
```latex
        HAN--PDQN / PDQN replay batch size & $512/128$ \\
```
修改后：
```latex
        HGHO / PDQN replay batch size & $512/128$ \\
```

### 第 1577 行

原文：
```latex
We compare HAN--PDQN with six benchmarks:
```
修改后：
```latex
We compare HGHO with six benchmarks:
```

### 第 1612 行

原文：
```latex
HAN--PDQN and PDQN use replay batch sizes of 512 and 128,
```
修改后：
```latex
HGHO and PDQN use replay batch sizes of 512 and 128,
```

### 第 1705 行

原文：
```latex
levels. HAN--PDQN and PDQN exhibit a more pronounced
```
修改后：
```latex
levels. HGHO and PDQN exhibit a more pronounced
```

### 第 1707 行

原文：
```latex
HAN--PDQN maintains the highest mean return.
```
修改后：
```latex
HGHO maintains the highest mean return.
```

### 第 1709 行

原文：
```latex
HAN--PDQN achieves a return of 87.78, compared with
```
修改后：
```latex
HGHO achieves a return of 87.78, compared with
```

### 第 1714 行

原文：
```latex
The comparison therefore favors HAN--PDQN in the
```
修改后：
```latex
The comparison therefore favors HGHO in the
```

### 第 1718 行

原文：
```latex
We next examine the sensitivity of HAN--PDQN to its
```
修改后：
```latex
We next examine the sensitivity of HGHO to its
```

### 第 1747 行

原文：
```latex
for the subsequent HAN--PDQN comparisons.
```
修改后：
```latex
for the subsequent HGHO comparisons.
```

### 第 1777 行

原文：
```latex
HAN--PDQN achieves the lowest mean successful-task
```
修改后：
```latex
HGHO achieves the lowest mean successful-task
```

### 第 1782 行

原文：
```latex
successful-task delay of HAN--PDQN increases from
```
修改后：
```latex
successful-task delay of HGHO increases from
```

### 第 1785 行

原文：
```latex
At 30 users, HAN--PDQN achieves 1779.7~ms, compared
```
修改后：
```latex
At 30 users, HGHO achieves 1779.7~ms, compared
```

### 第 1795 行

原文：
```latex
for HAN--PDQN and 9.47\% for PDQN, a reduction of
```
修改后：
```latex
for HGHO and 9.47\% for PDQN, a reduction of
```

### 第 1799 行

原文：
```latex
At 40 users, HAN--PDQN maintains a violation rate
```
修改后：
```latex
At 40 users, HGHO maintains a violation rate
```

### 第 1806 行

原文：
```latex
HAN--PDQN uses less terminal energy per successful
```
修改后：
```latex
HGHO uses less terminal energy per successful
```

### 第 1811 行

原文：
```latex
At 30 users, HAN--PDQN requires 1.242~J per successful
```
修改后：
```latex
At 30 users, HGHO requires 1.242~J per successful
```

### 第 1831 行

原文：
```latex
Under this convention, HAN--PDQN reduces successful-task
```
修改后：
```latex
Under this convention, HGHO reduces successful-task
```

### 第 1850 行

原文：
```latex
In Fig.~\ref{fig:service_operating_costs}(a), HAN--PDQN
```
修改后：
```latex
In Fig.~\ref{fig:service_operating_costs}(a), HGHO
```

### 第 1865 行

原文：
```latex
HAN--PDQN also maintains the lowest mean total
```
修改后：
```latex
HGHO also maintains the lowest mean total
```

### 第 1881 行

原文：
```latex
At 30 users, HAN--PDQN performs 0.606 handovers per
```
修改后：
```latex
At 30 users, HGHO performs 0.606 handovers per
```

### 第 1884 行

原文：
```latex
Thus, HAN--PDQN obtains its task-performance advantage
```
修改后：
```latex
Thus, HGHO obtains its task-performance advantage
```

### 第 1897 行

原文：
```latex
HAN--PDQN reduces mean blocked user-time,
```
修改后：
```latex
HGHO reduces mean blocked user-time,
```

### 第 1914 行

原文：
```latex
HAN--PDQN to account for dynamic connectivity and shared
```
修改后：
```latex
HGHO to account for dynamic connectivity and shared
```

### 第 1924 行

原文：
```latex
of 20--40 showed that, compared with PDQN, HAN--PDQN
```
修改后：
```latex
of 20--40 showed that, compared with PDQN, HGHO
```
