# 基于 Lyapunov 的在线凸松弛基线算法说明

## 1. 文档目的

本文整理一个用于与 **HAN+PDQN** 独立比较的模型驱动优化基线：

> **LOCR（Lyapunov-based Online Convex Relaxation）**  
> 基于 Lyapunov 的在线凸松弛联合切换与部分卸载算法。

LOCR 不是 HAN+PDQN 的组成模块，也不参与 HAN+PDQN 的训练。两种方法在相同
环境状态、合法动作掩码、episode seed 和评价协议下分别产生动作，再交给同一个
环境执行。

本文只定义算法原理与后续实现边界，不修改当前代码。需要特别强调：现有联合
切换与卸载问题包含离散决策、队列准入和非凸 reward，LOCR 求解的是逐时隙的
**凸松弛代理问题**，不是原始环境问题的全局最优解。

## 2. 为什么需要优化类基线

HAN+PDQN 属于数据驱动方法，通过与环境交互学习状态到联合动作的映射。优化类
基线则根据已知物理模型，在每个时隙显式构造目标函数和约束并求解。二者对比能
回答以下问题：

1. HAN+PDQN 相比模型驱动的在线优化是否获得了更高的长期收益；
2. HAN 表达的多用户—多卫星关系是否带来了超出传统队列感知优化的收益；
3. 学习方法是否用更低的在线决策耗时逼近或超过迭代求解器；
4. 在用户数、负载和卫星动态性增加后，两类方法的性能与复杂度如何变化。

因此，LOCR 在实验中的身份应写成“优化基线”或“基于凸松弛的在线优化基线”，
而不是本文提出的学习方法。

## 3. 与当前环境保持一致的控制边界

当前 HAN+PDQN 只控制：

- 离散 handover：stay 或切换到合法候选卫星；
- 离散 execution mode：纯本地或卸载；
- 条件连续 offload ratio：卸载模式下取 `[0.05, 1]`。

因此 LOCR 也只能控制上述变量。以下资源继续由环境按原规则决定：

- 用户上行发射功率固定为 24 dBm；
- 同星同批次实际准入的上传用户等分 10 MHz OFDMA 总带宽；
- 每星 MEC 总算力为 `5 GHz × 2 cores = 10 Gcycle/s`；
- MEC 使用两个并行处理槽和有限 FCFS 队列；
- 用户本地 CPU 固定为 1 GHz；
- 切换和迁移必须通过现有 action mask 与环境二次校验。

不能直接照搬文献中“联合优化带宽、发射功率和 CPU 频率”的全部变量，否则
LOCR 会获得 HAN+PDQN 不具备的控制权限，比较不公平。

## 4. 原始问题为什么不是凸问题

### 4.1 离散卫星关联和切换

设用户 `i` 是否选择卫星 `s` 为 `x_is`。原始决策要求：

```text
x_is ∈ {0, 1},    Σ_s x_is = 1
```

这会产生混合整数问题。切换动作还会影响迁移容量、服务中断和后续关联状态。

### 4.2 零膨胀卸载动作

环境的有效动作集合不是普通区间 `[0,1]`，而是：

```text
λ_i ∈ {0} ∪ [λ_min, 1],    λ_min = 0.05
```

纯本地和卸载之间存在离散执行模式，不能直接视为单个连续凸变量。

### 4.3 OFDMA 人数耦合

若卫星 `s` 在当前时隙有 `N_s` 个实际准入上传用户，则每个用户带宽为：

```text
B_is = B_s / N_s
```

而 `N_s` 又取决于所有用户最终的卫星选择和 execution mode。于是数据率、上传
时延和能耗与离散用户数量相互耦合。

### 4.4 FCFS 队列和有限槽位

一个任务只要发生卸载就占用一个 MEC 队列槽。队列是否已满、哪些任务处于
processing，以及两处理槽如何推进，均包含离散状态变化。简单的“总 cycles 除以
总算力”只能作为等待时间代理，不能完全复现真实 FCFS 状态转移。

### 4.5 reward 中的非凸项

成功任务的能耗归一化为：

```text
g(E) = E / (E + E_ref)
```

`g(E)` 对非负能耗是递增凹函数。最小化“时延项 + g(E)”一般不是凸优化。
Jain 公平性同样不是负载变量上的凸目标。deadline 成功/失败的 `+1/-1` 跳变也
是不连续的。

因此，任何保留全部原始语义的算法都不能被严谨地称为“纯凸优化算法”。LOCR
需要依次进行在线分解、连续松弛、凸代理和动作恢复。

## 5. LOCR 的总体结构

LOCR 由四层组成：

1. **Lyapunov 在线分解**：把长期随机优化转成当前时隙的 drift-plus-penalty
   问题；
2. **凸松弛与凸代理**：松弛卫星关联和执行模式，使用 epigraph 表达最大分支
   时延，并以凸函数替代非凸 reward；
3. **离散动作恢复**：将连续关联结果投影为合法 handover 和 execution mode；
4. **固定点重求解**：根据恢复后的同星上传人数重新计算 OFDMA 速率，再求一次
   连续卸载比例。

最终交给环境的仍是每个用户一个合法的：

```text
(handover_action, offload_ratio)
```

## 6. Lyapunov 在线优化原理

### 6.1 队列演化

用 `Q_s(t)` 表示卫星 `s` 在时隙 `t` 的剩余计算工作量，单位可统一为 CPU
cycles。其抽象更新式为：

```text
Q_s(t+1) = [Q_s(t) - μ_s(t)]⁺ + A_s(t)
```

其中：

- `μ_s(t)` 是该时隙处理掉的工作量；
- `A_s(t)` 是新卸载到该卫星的工作量；
- `[a]⁺ = max(a, 0)`。

当前环境还有每用户串行本地 CPU 时间线。后续实现可为其定义本地虚拟队列
`Q_i^loc(t)`，用于表示继续安排本地计算造成的积压压力。

### 6.2 Lyapunov 函数与漂移

定义二次 Lyapunov 函数：

```text
L(t) = 1/2 × [Σ_s Q_s(t)² + Σ_i Q_i^loc(t)²]
```

单步条件漂移为：

```text
Δ(t) = E[L(t+1) - L(t) | Q(t)]
```

将队列更新代入并取上界后，与当前决策相关的主要项为：

```text
Σ_s Q_s(t) A_s(t)
+ Σ_i Q_i^loc(t) A_i^loc(t)
```

直观解释如下：

- 向积压大的卫星继续加入任务会产生更高代价；
- 本地 CPU 已经繁忙时，继续选择本地计算也会产生更高代价；
- 队列压力会自动引导任务流向空闲 MEC 或更合适的执行模式。

### 6.3 Drift-plus-penalty

Lyapunov 方法不是只最小化队列，而是在队列漂移上叠加即时 QoS 代价：

```text
minimize    drift_upper_bound + V × instantaneous_cost
```

`V > 0` 是性能—积压权衡参数：

- `V` 较大：更重视即时 reward、时延和能耗，但可能允许更长队列；
- `V` 较小：更积极地稳定队列，但可能牺牲即时任务成本。

在标准可稳定性假设下，Lyapunov 优化通常具有 `O(1/V)` 的长期代价差距与
`O(V)` 的平均队列量权衡。本文不应直接把该理论界照搬到真实 FCFS 环境；只有
在后续给出的抽象队列模型和有界到达假设成立时才能引用这一性质。

## 7. 逐时隙凸松弛模型

### 7.1 状态与符号

在时隙 `t`，对有待处理任务的用户 `i` 定义：

| 符号 | 含义 |
| --- | --- |
| `D_i` | 输入数据量，bits |
| `C_i` | 所需计算量，CPU cycles |
| `T_i^max` | 任务 deadline |
| `w_i` | 从创建到当前时隙已经等待的时间 |
| `f_i` | 用户本地 CPU 频率 |
| `S_i(t)` | stay 与合法候选卫星组成的集合 |
| `R_is` | 给定 OFDMA 人数估计后的用户—卫星数据率 |
| `F_s` | 卫星 MEC 总计算能力 |
| `q_s` | 根据当前剩余 cycles 得到的 MEC 等效等待时间 |
| `RVT_is` | 用户对卫星的剩余可见时间 |

所有候选必须先通过当前环境的可见性、仰角、SNR、RVT、MEC 存在性和迁移
容量检查。无效候选不进入优化变量，而不是交给求解器后再处罚。

### 7.2 松弛决策变量

定义：

```text
x_is ∈ [0,1]    用户 i 对卫星 s 的关联权重
y_is ∈ [0,1]    用户 i 经卫星 s 卸载的激活权重
z_is ∈ [0,1]    用户 i 经卫星 s 卸载的任务比例
λ_i = Σ_s z_is  用户 i 的总卸载比例
d_i ≥ 0         任务完成时延的 epigraph 变量
ξ_i ≥ 0         deadline 软约束松弛量
```

基本约束为：

```text
Σ_s x_is = 1
0 ≤ y_is ≤ x_is
λ_min y_is ≤ z_is ≤ y_is
0 ≤ λ_i = Σ_s z_is ≤ 1
```

`x_is`、`y_is` 在原始问题中应为 0/1。把它们放宽到 `[0,1]` 后得到连续凸
可行域，但求解结果可能同时关联多颗卫星，必须在后处理阶段恢复为离散动作。

### 7.3 本地计算代理

给定卸载比例 `λ_i`，本地工作量为：

```text
C_i^loc = (1 - λ_i) C_i
```

本地计算时长和动态能耗分别为：

```text
T_i^loc,comp = (1 - λ_i) C_i / f_i
E_i^loc = κ f_i² (1 - λ_i) C_i
```

在固定 CPU 频率下二者都是 `λ_i` 的仿射函数。当前环境中本地分支还包含已经
等待的时间和本地 CPU 预约积压。由于“只要本地比例大于零就产生固定等待，而
比例等于零时分支消失”是不连续条件，凸模型只能使用连续等待时间代理。最终
时延仍必须由真实环境结算。

### 7.4 卸载分支代理

给定数据率和等效 MEC 等待时间后，经卫星 `s` 的卸载分支可写成：

```text
T_is^off = q_s y_is
           + z_is D_i / R_is
           + z_is C_i / F_s
           + z_is D_i × result_ratio / R_is^down
```

终端上传能耗为：

```text
E_is^up = P_i^battery × z_is D_i / R_is
```

这里的 `P_i^battery` 必须采用环境中的终端实际取电功率，而不是仅使用天线
辐射功率。固定 `R_is` 后，上述表达式均为仿射函数。

### 7.5 并行分支总时延

split task 的完成时延是本地与卸载分支的最大值。使用 epigraph：

```text
d_i ≥ T_i^loc
d_i ≥ Σ_s T_is^off
```

最小化 `d_i` 等价于最小化两条分支最大值，而且这两个不等式保持凸性。

deadline 使用软约束：

```text
d_i ≤ T_i^max + ξ_i
ξ_i ≥ 0
```

如果直接使用硬 deadline，一旦当前状态下所有动作都无法按时完成，整个优化
问题会 infeasible。对 `ξ_i` 施加足够大的惩罚，可以优先满足 deadline，同时
保证每个时隙都有可返回的动作。

### 7.6 MEC 容量与队列约束

可使用两类凸约束：

```text
Σ_i y_is ≤ available_queue_slots_s
Σ_i C_i z_is ≤ admitted_cycle_budget_s
```

第一个约束是有限队列槽位的连续松弛；第二个约束限制新准入 workload。实际
环境仍按照稳定 user-id 顺序执行批量准入，因此优化器的松弛可行不保证环境中
绝对不会发生拒绝。后续必须记录 optimizer-to-environment admission mismatch。

### 7.7 能耗 reward 的凸替代

环境使用：

```text
g(E_i) = E_i / (E_i + E_ref)
```

为保持一次求解问题是凸的，推荐基线使用其简单仿射上界：

```text
g_tilde(E_i) = E_i / E_ref
```

因为对 `E_i ≥ 0` 有：

```text
E_i / (E_i + E_ref) ≤ E_i / E_ref
```

该替代保留“能耗越高，代价越大”的排序，但数值尺度与真实 reward 不完全相同。
LOCR 产生动作后，正式比较仍由环境使用原始 `g(E)` 计算 reward。

另一种选择是在参考能耗 `E_i^(k)` 处使用一阶切线：

```text
g(E_i) ≤ g(E_i^(k)) + g'(E_i^(k)) [E_i - E_i^(k)]
```

每次固定参考点后右侧是仿射函数，但需要迭代更新，算法应改称 SCA/MM 基线。
首版 LOCR 建议使用固定线性代理，减少超参数与收敛歧义。

### 7.8 Jain 公平性的凸替代

环境中的 Jain 指数不适合作为凸最小化项。推荐使用归一化 MEC 负载方差：

```text
Phi_load = (1 / |S|) Σ_s (L_s - L_bar)²
```

其中：

```text
L_s = L_s^current + Σ_i C_i z_is / (F_s Δt)
L_bar = (1 / |S|) Σ_s L_s
```

该函数是负载向量上的凸二次函数。最小化负载方差与提高 Jain 公平性的方向
一致，但两者数值不等价。环境最终仍报告真实 Jain 指数。

### 7.9 切换与 RVT 项

切换中断代理为：

```text
H_i = Σ_(s != current_i) x_is × handover_delay / slot_duration
```

这是 `x_is` 的线性函数。合法候选已经通过硬 RVT 阈值后，还可以增加一个较小
的长期连接风险项：

```text
P_i^rvt = Σ_s x_is × rvt_threshold / max(RVT_is, epsilon)
```

它鼓励选择剩余可见时间更长的卫星。该项属于模型驱动的前瞻代理，必须使用
HAN+PDQN 观测中同样可获得的 RVT，不能读取未来任务到达信息。

### 7.10 推荐的逐时隙目标

综合上述项，推荐的凸代理目标为：

```text
minimize
    V × Σ_i [
        0.60 × d_i / T_i^max
        + 0.40 × g_tilde(E_i)
        + 0.30 × H_i
        + M_deadline × ξ_i / T_i^max
    ]
    + α_queue × Σ_s Q_s A_s
    + α_local × Σ_i Q_i^loc (1 - λ_i) C_i
    + η_load × Phi_load
    + ρ_rvt × Σ_i P_i^rvt
```

其中：

```text
A_s = Σ_i C_i z_is
```

在当前时隙固定速率、等待时间估计和候选集合后：

- 时延、能耗、切换、RVT 和队列压力项为线性或分段线性；
- 负载方差是凸二次函数；
- 约束是线性不等式或仿射等式。

因此该子问题是标准凸 QP；关闭负载方差项后可以退化为 LP。

### 7.11 量纲归一化

上述公式中的 seconds、joules、CPU cycles 和队列长度量纲不同，不能把原始数值
直接加权相加。例如任务计算量可达到 `10^9` cycles，而单步 reward 通常为个位数。
实现时至少应采用以下无量纲量：

```text
d_i / T_i^max
E_i / E_ref
Q_s / (F_s × T_queue_ref)
A_s / (F_s × slot_duration)
Q_i^loc / (f_i × T_local_ref)
ξ_i / T_i^max
```

负载方差使用 `[0,1]` 范围内的归一化 utilization 计算。只有先完成量纲归一化，
`V`、`α_queue`、`α_local`、`η_load` 等参数才具有可比较、可复现的含义。正式
文档和实验配置中必须记录所有参考尺度，不能只报告最终权重。

## 8. OFDMA 固定点与动作恢复

### 8.1 为什么需要重求解

第一次求解前并不知道最终同星上传人数 `N_s`，但数据率依赖 `N_s`。因此不能
在不作处理的情况下声称整个联合问题是一次凸求解。

### 8.2 推荐流程

1. 用上一时隙实际上传人数或当前关联人数初始化 `N_s^(0)`；
2. 计算 `B_s/N_s^(0)` 和对应数据率；
3. 求解松弛凸问题；
4. 将每个用户投影到一个合法卫星；
5. 将 `λ_i < 0.05` 映射为 `0`，其余裁剪到 `[0.05,1]`；
6. 按投影结果重新统计 `N_s^(1)`；
7. 固定卫星关联，重新求解仅含卸载比例的凸问题；
8. 若上传激活集合变化，则最多重复 2--3 次；
9. 达到迭代上限仍未稳定时，使用最后一个可行结果。

每轮内部是凸问题，但“更新人数—重求解”的外层固定点过程不是单个全局凸
问题。这也是算法名称中必须保留“在线凸松弛”而不能写“全局凸最优”的原因。

### 8.3 关联变量取整

最简单的投影规则为：

```text
selected_sat_i = argmax_s x_is
```

但独立 argmax 可能导致多个用户同时挤入同一颗卫星。更稳健的恢复方式是：

1. 按剩余 deadline 从小到大排列用户；
2. 对每个用户按 `x_is` 从大到小尝试候选；
3. 只有目标卫星剩余队列槽和迁移容量可行时才提交；
4. 无可行目标时保持当前卫星；
5. 固定全部离散结果后再求一次连续卸载比例。

若后续希望减少顺序偏差，可以把恢复阶段写成容量受限的最小费用匹配，但首版
不应同时引入过多额外算法。

## 9. 算法伪代码

```text
Algorithm LOCR

Input:
    当前用户任务、合法 handover mask、链路/RVT、
    MEC 队列、本地 CPU 时间线、上一时隙上传人数

for each time slot t:
    1. 读取与 HAN+PDQN 相同的可观测状态
    2. 构造每个用户的合法候选集合
    3. 从真实 MEC 队列计算 Q_s、剩余槽位和等效等待时间
    4. 初始化每星上传人数 N_s

    for k = 1 ... K_max:
        5. 根据 N_s 计算 OFDMA 带宽和数据率
        6. 构造并求解逐时隙凸 QP
        7. 将松弛关联投影为合法卫星动作
        8. 将连续 λ 投影到 {0} ∪ [0.05, 1]
        9. 重新统计 N_s
       10. 若关联和上传激活集合稳定，则停止

   11. 固定离散动作，再求一次连续卸载比例凸问题
   12. 若求解失败，执行安全回退策略
   13. 将动作提交给原环境
   14. 环境按真实 OFDMA、FCFS、deadline 和 reward 规则更新
```

当前无任务且不处于预切换/重连窗口的用户应直接输出 stay 和 `offload_ratio=0`，
不进入任务卸载子问题；处于预切换或阻塞状态的无任务用户只参加关联恢复，不参加
任务时延和能耗项计算。

推荐安全回退策略：

- handover：stay；阻塞用户则选择 action mask 中 RVT 最大的合法候选；
- offload：若 MEC 无容量则本地，否则使用上一时隙可行动作；
- 不得因求解失败跳过环境动作或读取未来状态。

## 10. 与 HAN+PDQN 的公平比较协议

### 10.1 必须相同

- 环境 schema、物理参数和 reward；
- 用户数量、episode 长度和 task arrival seed；
- 星座起点、用户位置和任务序列；
- handover action mask 与候选卫星排序；
- OFDMA 等分、MEC FCFS、迁移和 deadline 结算规则；
- 最终评价指标和配对 episode seeds。

### 10.2 LOCR 不允许拥有的额外信息

- 未来任务到达；
- 未来信道或未包含在当前观测中的精确轨迹序列；
- 环境内部尚未暴露给 HAN+PDQN 的随机数状态；
- 可绕过 action mask 的目标卫星；
- 可自行改变的带宽、发射功率或 CPU 频率。

RVT 已经属于当前观测，因此 LOCR 可以使用当前 RVT，但不能额外获取未来任务。

### 10.3 参数选择

需要调节的核心参数为：

```text
V, M_deadline, α_queue, α_local, η_load, ρ_rvt
```

这些参数只能在训练/调参 seeds 上选择，不能根据正式评估 seeds 调参。建议：

1. 先固定 `M_deadline` 为足够大的可行性优先权重；
2. 调节 `V` 观察即时 QoS—队列积压权衡；
3. 再加入 `η_load`；
4. 最后决定是否保留较小的 `ρ_rvt`；
5. 报告完整参数和归一化尺度，避免隐藏调参优势。

### 10.4 除原有指标外还应报告

- 每时隙平均、P95 和最大求解时间；
- solver optimal、inaccurate、infeasible、timeout 比例；
- 离散取整前后的目标差；
- 取整后重新求解次数；
- optimizer 预测准入与环境实际准入不一致率；
- 安全回退次数；
- LOCR 的峰值内存占用。

这些指标用于说明优化基线在性能之外的在线计算代价。

## 11. 求解器与实现建议

后续实现可使用支持 disciplined convex programming 的建模库，例如 CVXPY，
由其调用适合 QP 或锥规划的数值求解器。库只负责数值求解，不会自动完成：

- 从环境状态提取数学系数；
- 判断表达式是否与真实 FCFS/OFDMA 语义一致；
- 设计非凸项的代理；
- 松弛变量取整；
- solver 失败回退；
- 与现有基线评估协议对接。

首版应优先保持模型小而稳定，不建议一开始同时优化功率、带宽和 CPU。可缓存
问题结构并使用参数化系数，避免每个时隙重新构建全部表达式。

## 12. 适用性、优点与局限

### 12.1 优点

- 不需要训练，便于复现；
- 显式利用队列、deadline、能耗和卫星负载；
- 能同时考虑多个用户，不像顺序 Joint Greedy 只做局部选择；
- 每个凸子问题具有明确的数值最优性状态；
- 可以作为检验 HAN+PDQN 性能是否超过传统模型驱动方法的强基线。

### 12.2 局限

- 松弛解不是原始混合整数问题的全局最优解；
- 等效 MEC 等待时间不能完全复制两个处理槽的 FCFS 顺序；
- OFDMA 通过固定点迭代处理，缺少全局凸性保证；
- 能耗和 Jain reward 使用代理函数，优化目标与环境 reward 不完全相同；
- 取整会破坏松弛解的最优性，并可能引入用户顺序偏差；
- 用户数增加时，每时隙求解成本会明显高于神经网络前向推理。

因此实验结论应写成“在统一环境评价下，HAN+PDQN 相比 LOCR 的表现”，不能
写成“HAN+PDQN 超过原问题全局最优解”。

## 13. 参考文献与借鉴关系

### 13.1 LEO/卫星场景的核心参考

1. X. Zhang, J. Liu, R. Zhang, et al., “Energy-Efficient Computation Peer
   Offloading in Satellite Edge Computing Networks,” *IEEE Transactions on
   Mobile Computing*, vol. 23, no. 4, pp. 3077–3091, 2024.  
   DOI: <https://doi.org/10.1109/TMC.2023.3269801>  
   **借鉴内容**：动态卫星网络中的 Lyapunov 框架、队列/负载约束、全局问题向
   在线子问题分解。该文主要研究卫星间多跳 peer offloading，不能直接复制其
   网络动作。

2. Y. Li, S. Zhu, T. Xiong, et al., “Computation Offloading in
   Delay-Sensitive Multisatellite Cooperative Edge Computing Systems,”
   *IEEE Internet of Things Journal*, vol. 13, no. 1, pp. 123–137, 2026.  
   DOI: <https://doi.org/10.1109/JIOT.2025.3580504>  
   **借鉴内容**：多卫星选择、离散变量松弛、Lagrange 乘子、迭代资源优化和
   使用 CVX 求解凸子问题。该文优化了当前环境并不开放的资源变量，LOCR 只能
   借鉴求解结构。

3. J. Kim and J. Kwak, “DCOOL: Dynamic Computation Offloading and Resource
   Allocation for LEO Satellite-Assisted Edge Computing in a Ground-Space
   Integrated Framework,” *ICT Express*, vol. 10, no. 6, pp. 1212–1219,
   2024.  
   DOI: <https://doi.org/10.1016/j.icte.2024.09.014>  
   **借鉴内容**：LEO-MEC 中使用 drift-plus-penalty 将长期队列稳定问题转为
   逐时隙优化。其场景和动作空间比当前系统简单，作为场景适配补充文献，而非
   唯一理论依据。

4. S. Bhandari, T. X. Vu, and S. Chatzinotas, “LEO-Based Edge Computing
   Service Platform for Challenging Geographical Terrain,” *IEEE Open Journal
   of the Communications Society*, 2025.  
   DOI: <https://doi.org/10.1109/OJCOMS.2025.3638149>  
   **借鉴内容**：交替优化、SCA、切换期间任务交付和凸资源分配。适合解释为何
   LEO 切换与卸载联合问题通常需要分解，而不是一次纯凸求解。

### 13.2 MEC 凸优化的理论与方法参考

5. Y. Mao, J. Zhang, and K. B. Letaief, “Dynamic Computation Offloading for
   Mobile-Edge Computing With Energy Harvesting Devices,” *IEEE Journal on
   Selected Areas in Communications*, vol. 34, no. 12, pp. 3590–3605, 2016.  
   DOI: <https://doi.org/10.1109/JSAC.2016.2611964>  
   **借鉴内容**：经典 Lyapunov drift-plus-penalty 在线卸载、只依赖当前状态、
   逐时隙确定性子问题、闭式解或二分搜索。

6. Y. Wang, M. Sheng, X. Wang, L. Wang, and J. Li, “Mobile-Edge Computing:
   Partial Computation Offloading Using Dynamic Voltage Scaling,” *IEEE
   Transactions on Communications*, vol. 64, no. 10, pp. 4268–4282, 2016.  
   DOI: <https://doi.org/10.1109/TCOMM.2016.2599530>  
   **借鉴内容**：部分卸载比例、变量替换、把能耗最小化问题重构为凸问题，以及
   本地计算与卸载的时延—能耗关系。

7. C. You, K. Huang, H. Chae, and B.-H. Kim, “Energy-Efficient Resource
   Allocation for Mobile-Edge Computation Offloading,” *IEEE Transactions on
   Wireless Communications*, vol. 16, no. 3, pp. 1397–1411, 2017.  
   DOI: <https://doi.org/10.1109/TWC.2016.2633522>  
   **借鉴内容**：多用户部分卸载、有限 MEC 容量、Lagrange 对偶、KKT 和卸载
   优先级。该文同时说明 OFDMA 联合资源分配会成为混合整数非凸问题，支持本文
   使用固定点与松弛，而非宣称一次全局凸求解。

8. S. Sardellitti, G. Scutari, and S. Barbarossa, “Joint Optimization of
   Radio and Computational Resources for Multicell Mobile-Edge Computing,”
   *IEEE Transactions on Signal and Information Processing over Networks*,
   vol. 1, no. 2, pp. 89–103, 2015.  
   DOI: <https://doi.org/10.1109/TSIPN.2015.2448520>  
   **借鉴内容**：successive convex approximation、内凸化和对偶分解。若以后
   LOCR 保留非凸能耗归一化或进一步优化无线资源，可将其扩展为 SCA 版本。

9. C. You and K. Huang, “Multiuser Resource Allocation for Mobile-Edge
   Computation Offloading,” in *Proc. IEEE GLOBECOM*, 2016.  
   DOI: <https://doi.org/10.1109/GLOCOM.2016.7842016>  
   **借鉴内容**：多用户凸资源分配、时延约束和基于卸载优先级的低复杂度策略；
   是上述 TWC 工作的会议阶段参考。

## 14. 推荐的论文表述

方法介绍可采用以下表述：

> 为提供无需训练的模型驱动对照，本文设计了基于 Lyapunov 的在线凸松弛基线
> LOCR。该方法利用 drift-plus-penalty 将长期随机队列优化转化为逐时隙问题，
> 对离散卫星关联和卸载激活变量进行连续松弛，并通过 epigraph 和仿射能耗代理
> 构造凸二次子问题。求解松弛问题后，算法将卫星关联投影为合法切换动作，依据
> 实际同星上传用户数更新 OFDMA 速率，并在固定关联下重新优化部分卸载比例。
> LOCR 与 HAN+PDQN 使用相同状态、动作可行域和环境资源规则；其输出由原环境
> 按真实 FCFS、OFDMA 和 reward 结算。

结果分析中应采用以下限定：

> 由于原始问题包含离散切换、零膨胀卸载、OFDMA 用户数量耦合和非凸奖励项，
> LOCR 是原问题的在线凸松弛近似，而非全局最优求解器。
