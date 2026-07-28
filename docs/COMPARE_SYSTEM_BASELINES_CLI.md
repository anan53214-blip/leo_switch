# compare_system_baselines.py 命令行参数说明

本文档记录当前版本 `scripts/compare_system_baselines.py` 的 `--` 参数用法。

推荐使用项目 Conda 环境执行：

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe scripts\compare_system_baselines.py
```

这个脚本用于训练或加载一个系统算法，然后评估指定基线算法，最后输出
`JSON`、`CSV` 和对比图像。

## 常用命令

### 单用户轻量对比

用于 `num_users = 1`，系统算法与简单基线以及 `MAPPO(no-HAN)` 消融进行快速对比：

```powershell
& 'C:\Users\19704\.conda\envs\satellite.env\python.exe' scripts\compare_system_baselines.py `
  --run-mode train_compare `
  --objective multi_objective `
  --num-users 1 `
  --total-timesteps 50000 `
  --episodes 3 `
  --max-steps 600 `
  --device cpu `
  --best-model-metric avg_delay `
  --compare-ranking-metric avg_delay `
  --baselines random min_distance full_local joint_greedy mappo_no_han `
  --system-run-dir results\single_user_system_mappo `
  --output-dir results\baseline_compare\single_user_mappo_vs_baselines
```

### 单用户完整默认基线对比

`--baselines all` 会包含学习型基线，因此会明显更慢：

```powershell
& 'C:\Users\19704\.conda\envs\satellite.env\python.exe' scripts\compare_system_baselines.py `
  --run-mode train_compare `
  --objective multi_objective `
  --num-users 1 `
  --total-timesteps 50000 `
  --episodes 3 `
  --max-steps 600 `
  --device cpu `
  --best-model-metric avg_delay `
  --compare-ranking-metric avg_delay `
  --baselines all `
  --system-run-dir results\single_user_system_all `
  --output-dir results\baseline_compare\single_user_all_baselines
```

### 只用已有系统模型重新对比

当 `--system-run-dir` 里已经有 `best_model.pt` 或 `final_model.pt`，并且有运行配置或训练历史时，可以使用 `compare_only`：

```powershell
& 'C:\Users\19704\.conda\envs\satellite.env\python.exe' scripts\compare_system_baselines.py `
  --run-mode compare_only `
  --system-run-dir results\single_user_system_mappo `
  --episodes 3 `
  --max-steps 600 `
  --device cpu `
  --compare-ranking-metric avg_delay `
  --baselines random min_distance full_local joint_greedy `
  --output-dir results\baseline_compare\single_user_recompare
```

## 运行模式与系统模型路径

| 参数 | 取值 / 默认值 | 用法 |
| --- | --- | --- |
| `--run-mode` | `train_compare` 或 `compare_only`，默认 `train_compare` | `train_compare` 会先训练系统算法再做基线对比；`compare_only` 只读取已有系统模型做对比。 |
| `--system-run-dir` | 默认 `results/full_train_latency_priority` | 系统算法训练输出目录，也可以作为 `compare_only` 的已有模型目录。 |
| `--system-checkpoint` | checkpoint 路径，默认无 | 显式指定系统算法的 `best_model.pt` 或 `final_model.pt`。 |
| `--resume-system` | 开关参数 | 从已有系统 checkpoint 继续训练。 |
| `--overwrite-system-run-dir` | 开关参数 | 允许新的 `train_compare` 直接写入已有系统目录。默认会保护已有目录，必要时创建带时间戳的兄弟目录。 |
| `--exp-name` | 默认 `han_mappo_latency_priority` | 从统一对比脚本训练系统算法时使用的实验名。 |

## 训练与评估规模

| 参数 | 默认值 | 用法 |
| --- | --- | --- |
| `--episodes` | `3` | 每个方法评估的 episode 数。 |
| `--max-steps` | 默认不覆盖 | 覆盖训练、评估、基线测试时每个 episode 的最大步数。 |
| `--total-timesteps` | `300000` | 系统算法训练步数。学习型基线如果没有单独设置步数，也会使用这个值。 |
| `--early-stop-patience` | `0` | MAPPO 类训练的早停耐心值，`0` 表示禁用早停。 |
| `--seed` | `42` | 基线评估和默认配置使用的随机种子。 |
| `--device` | `auto`，可选 `auto`、`cpu`、`cuda` | 训练和评估设备。本机快速验证建议用 `cpu`。 |
| `--objective` | `multi_objective` | 当前保留的目标函数。旧的 `delay_only` 和 `energy_only` 分支已经删除。 |
| `--num-users` | `10` | 没有已有系统 run 时使用的用户数量。单用户实验设为 `--num-users 1`。 |

## 指标选择

`--best-model-metric` 和 `--compare-ranking-metric` 都支持下面这些指标：

```text
reward
avg_delay
avg_success_delay
p95_success_delay
total_energy
service_continuity_rate
service_availability_rate
handover_failure_rate
load_balance_coefficient
load_balance_variance
mec_load_fairness
jain_mec_load_fairness
avg_load_balance_score
energy_per_successful_task
task_completion_rate
task_success_rate
task_failure_rate
task_settlement_rate
```

| 参数 | 默认值 | 用法 |
| --- | --- | --- |
| `--best-model-metric` | `avg_delay` | 系统训练时用于保存 `best_model.pt` 的指标，也用于从历史记录中选最好的一条。 |
| `--compare-ranking-metric` | `avg_delay` | 用于选择启发式基线的最优卸载比例，并给对比结果排序或标注。 |

`avg_delay` 仅为兼容旧实验保留。论文主结果推荐使用
`avg_success_delay`，并同时报告 `p95_success_delay`；负载公平性正文使用
`jain_mec_load_fairness`。

脚本会自动处理指标方向。比如 `avg_delay`、`total_energy`、`handover_failure_rate`、
`avg_success_delay`、`p95_success_delay`、`load_balance_variance`、
`task_failure_rate` 和 `energy_per_successful_task`
都是越低越好。

## 基线算法选择

| 参数 | 默认值 | 用法 |
| --- | --- | --- |
| `--baselines` | `all` | 指定要评估的基线算法。快速实验建议显式列出基线，而不是用 `all`。 |
| `--fixed-offload-grid` | `0.0 0.5 1.0` | 简单启发式基线使用的卸载比例候选，例如 `random`、`min_distance`、`full_local`。 |
| `--joint-offload-grid` | `0.0 0.25 0.5 0.75 1.0` | `joint_greedy` 使用的卸载比例搜索网格。 |
| `--dqn-offload-grid` | `0.0 0.5 1.0` | DQN 基线使用的离散卸载比例网格。 |

当前 `--baselines all` 会展开为：

```text
random
min_distance
full_local
joint_greedy
maddpg
pdqn
han_mappo
mappo_no_han
attn_mappo
han_attn
han_maddpg
han_pdqn
```

推荐的轻量基线组合：

```powershell
--baselines random min_distance full_local joint_greedy mappo_no_han
```

学习型基线名称：

```text
dqn
maddpg
pdqn
mappo_no_han
han_mappo
attn_mappo
han_attn
han_maddpg
han_pdqn
```

## 学习型基线训练步数

这些参数只有在选择了对应学习型基线时才有意义。

| 参数 | 默认值 | 用法 |
| --- | --- | --- |
| `--dqn-timesteps` | 默认使用 `--total-timesteps` | DQN 基线训练步数。 |
| `--maddpg-timesteps` | 默认使用 `--total-timesteps` | MADDPG 和 HAN+MADDPG 训练步数。 |
| `--pdqn-timesteps` | 默认使用 `--total-timesteps` | PDQN 和 HAN+PDQN 训练步数。 |
| `--no-han-total-timesteps` | 默认使用 `--total-timesteps` | MAPPO(no-HAN) 消融训练步数。 |
| `--attn-mappo-timesteps` | 默认使用 `--total-timesteps` | Attn+MAPPO 基线训练步数。 |

## 复用与跳过选项

| 参数 | 默认值 | 用法 |
| --- | --- | --- |
| `--skip-system-eval` | 开关参数 | 跳过系统 checkpoint 评估，只在有历史摘要时使用历史摘要。 |
| `--reuse-methods-from` | 可传 0 个或多个路径 | 从已有 `baseline_compare` 目录或 `comparison_summary.json` 中复用方法结果。 |
| `--reuse-methods` | 可传 0 个或多个方法名 | 指定从 `--reuse-methods-from` 中复用哪些方法。为空时复用所有非系统方法。 |
| `--reuse-learned-checkpoints` | 开关参数 | 对学习型基线，优先评估 `--output-dir/learned_baselines/<method>` 下已有 checkpoint，而不是重新训练。 |

## 绘图与输出

| 参数 | 默认值 | 用法 |
| --- | --- | --- |
| `--plot-window` | `5` | reward 曲线和按 step 记录指标曲线的平滑窗口。 |
| `--output-dir` | 默认在 `results/baseline_compare` 下创建时间戳目录 | JSON、CSV 和图像输出目录。论文实验建议显式指定。 |

脚本一定会写出：

```text
comparison_summary.json
comparison_summary.csv
method_comparison.png
paper_baseline_dashboard.png
```

如果有训练历史和 episode 级指标，还可能写出：

```text
episode_metrics.csv
reward_curve_vs_baselines.png
training_qos_metrics_vs_steps.png
reward_components_vs_steps.png
reward_components_per_task_vs_steps.png
additional_metrics_episode_comparison.png
delay_energy_tradeoff.png
success_continuity_tradeoff.png
performance_radar.png
reward_distribution.png
```

学习型基线的 checkpoint 和训练历史会放在：

```text
<output-dir>\learned_baselines\<method>\
```

## 实用建议

- 论文实验建议总是显式设置 `--output-dir`，避免结果散落到时间戳目录。
- 快速验证或单用户实验建议显式写 `--baselines`，不要直接用 `all`。
- 改用户数量、算法、奖励权重、观测维度或指标协议时，建议使用新的 `--system-run-dir`。
- 本机快速验证建议使用 `--device cpu`，除非已经确认当前 PyTorch/CUDA 支持本机 GPU。
