# 多用户与训练产物绘图命令说明

本文档说明如何基于已有结果重新生成对比图。这里的命令默认不重新训练、不重新评估，只读取已有结果文件。

涉及两个脚本：

```text
scripts\run_multiuser_scaling_suite.py
scripts\plot_training_artifacts.py
```

## 1. 两个脚本的区别

| 场景 | 使用脚本 |
| --- | --- |
| 从 `u20/u25/u30/.../comparison_summary.csv` 生成多用户扩展图 | `run_multiuser_scaling_suite.py --aggregate-only` |
| 从几个 `training_history.json` 生成算法训练曲线和方法对比图 | `plot_training_artifacts.py --history ...` |
| 从一个已有 `comparison_summary.json` 重画普通 baseline 对比图 | `plot_training_artifacts.py --comparison-summary ...` |

一句话区分：

- 要画 `multiuser_core_metrics.png`：用 `run_multiuser_scaling_suite.py --aggregate-only`。
- 要画几个算法的训练曲线和普通方法对比图：用 `plot_training_artifacts.py`。

## 2. 多用户聚合绘图：run_multiuser_scaling_suite.py

`--aggregate-only` 模式只读取已有的每个用户数目录下的 `comparison_summary.csv`：

```text
results\baseline_compare\multiuser_scaling_<run_id>\u20\comparison_summary.csv
results\baseline_compare\multiuser_scaling_<run_id>\u25\comparison_summary.csv
results\baseline_compare\multiuser_scaling_<run_id>\u30\comparison_summary.csv
...
```

例如已有目录：

```text
results\baseline_compare\multiuser_scaling_multiuser_6_7
```

对应的 `run_id` 是：

```text
multiuser_6_7
```

因为脚本会自动拼接目录名：

```text
results\baseline_compare\multiuser_scaling_<run_id>
```

### 2.1 重新生成全部算法的多用户图

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe scripts\run_multiuser_scaling_suite.py `
  --run-id multiuser_6_7 `
  --user-counts 20 25 30 35 40 `
  --aggregate-only
```

这不会重新训练，也不会重新评估。它会重新生成：

```text
multiuser_summary.csv
multiuser_core_metrics.png
multiuser_resource_metrics.png
multiuser_reward_convergence.png
suite_manifest.json
```

注意：不加后缀时会覆盖这些默认输出文件。

### 2.2 只生成某几个算法的对比图

使用 `--include-methods` 过滤要画的算法。

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe scripts\run_multiuser_scaling_suite.py `
  --run-id multiuser_6_7 `
  --user-counts 20 25 30 35 40 `
  --aggregate-only `
  --include-methods han_mappo mappo_no_han random min_distance full_local joint_greedy
```

常用方法名：

```text
han_mappo
attn_mappo
mappo_no_han
maddpg
pdqn
han_maddpg
han_pdqn
random
min_distance
full_local
joint_greedy
```

`HAN+MAPPO` 在多用户聚合 CSV 中一般对应：

```text
han_mappo
```

`MAPPO` 或 `MAPPO(no-HAN)` 对应：

```text
mappo_no_han
```

### 2.3 不覆盖原图：使用输出后缀

如果不想覆盖原来的默认图，添加 `--output-suffix`。

推荐命令：

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe scripts\run_multiuser_scaling_suite.py `
  --run-id multiuser_6_7 `
  --user-counts 20 25 30 35 40 `
  --aggregate-only `
  --include-methods han_mappo mappo_no_han random min_distance full_local joint_greedy `
  --output-suffix selected
```

输出文件会变成：

```text
multiuser_summary_selected.csv
multiuser_core_metrics_selected.png
multiuser_resource_metrics_selected.png
multiuser_reward_convergence_selected.png
suite_manifest_selected.json
```

原始默认文件不会被覆盖：

```text
multiuser_summary.csv
multiuser_core_metrics.png
multiuser_resource_metrics.png
multiuser_reward_convergence.png
suite_manifest.json
```

### 2.4 只选择部分用户数

如果只想画 `20/30/40` 用户：

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe scripts\run_multiuser_scaling_suite.py `
  --run-id multiuser_6_7 `
  --user-counts 20 30 40 `
  --aggregate-only `
  --include-methods han_mappo mappo_no_han random min_distance `
  --output-suffix u20_u30_u40_selected
```

这要求以下文件存在：

```text
results\baseline_compare\multiuser_scaling_multiuser_6_7\u20\comparison_summary.csv
results\baseline_compare\multiuser_scaling_multiuser_6_7\u30\comparison_summary.csv
results\baseline_compare\multiuser_scaling_multiuser_6_7\u40\comparison_summary.csv
```

### 2.5 多用户聚合参数说明

| 参数 | 作用 |
| --- | --- |
| `--run-id` | 指定多用户 suite 的运行 ID；目录为 `results\baseline_compare\multiuser_scaling_<run_id>` |
| `--user-counts` | 指定要聚合的用户数目录，例如 `20 25 30 35 40` |
| `--aggregate-only` | 只读取已有 CSV 并重新生成聚合图，不训练、不评估 |
| `--include-methods` | 只保留指定算法的方法名 |
| `--output-suffix` | 给输出文件增加后缀，避免覆盖默认图 |

## 3. 单次训练产物绘图：plot_training_artifacts.py

`plot_training_artifacts.py` 是纯绘图脚本，用于从已有 `training_history.json` 或 `comparison_summary.json` 生成普通算法对比图。它不会重新训练、不会重新评估、不会加载 checkpoint。

它适合以下场景：

- 对比几个训练 run 的奖励曲线
- 从多个 `training_history.json` 生成方法对比图
- 从已有 `comparison_summary.json` 重新生成普通 baseline 对比图

它不适合生成 `multiuser_core_metrics.png` 这种跨用户数扩展图。多用户扩展图仍然使用：

```powershell
scripts\run_multiuser_scaling_suite.py --aggregate-only
```

### 3.1 从多个 training_history.json 生成算法对比图

示例：

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe scripts\plot_training_artifacts.py `
  --history HAN+MAPPO=results\full_train_xxx\training_history.json `
  --history MAPPO=results\baseline_compare\xxx\learned_baselines\mappo_no_han\training_history.json `
  --history HAN+MADDPG=results\baseline_compare\xxx\learned_baselines\han_maddpg\training_history.json `
  --history HAN+PDQN=results\baseline_compare\xxx\learned_baselines\han_pdqn\training_history.json `
  --output-dir results\plot_only_comparison\selected_methods `
  --selection-metric avg_delay `
  --plot-window 5
```

说明：

- `LABEL=PATH` 中的 `LABEL` 是图例显示名。
- 第一条 `--history` 默认作为系统方法显示。
- `--selection-metric` 用来从历史记录里选择代表性指标记录。
- `--plot-window` 控制训练曲线平滑窗口。

### 3.2 从已有 comparison_summary.json 重新生成普通对比图

示例：

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe scripts\plot_training_artifacts.py `
  --comparison-summary results\baseline_compare\xxx\comparison_summary.json `
  --output-dir results\baseline_compare\xxx\replot `
  --selection-metric avg_delay `
  --plot-window 5
```

也可以传入目录，脚本会自动读取目录下的 `comparison_summary.json`：

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe scripts\plot_training_artifacts.py `
  --comparison-summary results\baseline_compare\xxx `
  --output-dir results\baseline_compare\xxx\replot
```

### 3.3 plot_training_artifacts.py 输出文件

`plot_training_artifacts.py` 会在 `--output-dir` 中生成：

```text
comparison_summary.json
comparison_summary.csv
plot_manifest.json
method_comparison.png
reward_curve_vs_baselines.png
training_qos_metrics_vs_steps.png
reward_components_vs_steps.png
delay_energy_tradeoff.png
success_continuity_tradeoff.png
performance_radar.png
paper_baseline_dashboard.png
```

如果输入包含 episode metrics，还可能生成：

```text
additional_metrics_episode_comparison.png
reward_distribution.png
```

### 3.4 plot_training_artifacts.py 参数说明

| 参数 | 作用 |
| --- | --- |
| `--history` | 指定一个 `training_history.json`，可以写成 `显示名=路径`，可重复传入 |
| `--comparison-summary` | 指定已有 `comparison_summary.json` 或其所在目录 |
| `--output-dir` | 指定输出目录 |
| `--selection-metric` | 指定用于选择代表记录和排序的指标，例如 `avg_delay` |
| `--plot-window` | 指定训练曲线平滑窗口 |

## 4. 常见问题

### 会不会重新训练？

不会。多用户聚合使用 `--aggregate-only` 时只读取已有 CSV；`plot_training_artifacts.py` 本身也是 plot-only 脚本。

### 多用户聚合不加 `--output-suffix` 会发生什么？

会覆盖默认聚合输出：

```text
multiuser_summary.csv
multiuser_core_metrics.png
multiuser_resource_metrics.png
multiuser_reward_convergence.png
suite_manifest.json
```

### 加 `--output-suffix selected` 会发生什么？

会写出带 `_selected` 的新文件，不覆盖默认图：

```text
multiuser_core_metrics_selected.png
```

### 筛选方法名写错会怎样？

如果 `--include-methods` 没有匹配到任何方法，脚本会报错：

```text
No methods matched --include-methods
```

这时检查每个 `u*/comparison_summary.csv` 中的 `method` 列，按里面的名字填写。
