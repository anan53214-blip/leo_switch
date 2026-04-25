# 当前对比与绘图说明

本文档记录当前基线集合与 reward 曲线绘图方式，便于复现实验和核对论文图。

## 1. 当前基线集合

`scripts/compare_system_baselines.py --baselines all` 当前评估：

- `random`
- `min_distance`
- `full_local`
- `joint_greedy`
- `dqn`
- `mappo_no_han`

已移除的旧基线名称：

- `stay`
- `max_elev`
- `max_rvt`
- `threshold_rvt`

这些旧策略主要是早期单规则切换检查。当前集合更适合论文对比，因为它覆盖：

- 随机策略下界
- 几何规则切换
- 全本地计算下界
- 一步联合贪心强基线
- 离散化值函数 RL 基线
- 去掉 HAN 图编码器的 MAPPO 消融

## 2. 学习式基线

### 2.1 DQN

DQN 将混合动作空间离散化：

```text
q_action = (handover_action, discrete_offload_ratio)
```

卸载比例网格由以下参数控制：

```powershell
--dqn-offload-grid 0.0 0.5 1.0
```

训练长度默认使用 `--total-timesteps`，也可以单独覆盖：

```powershell
--dqn-timesteps <steps>
```

### 2.2 MAPPO 无 HAN

`mappo_no_han` 消融保留 MAPPO，但移除 HAN 编码器。Actor 和 Critic 直接接收每个用户的原始环境观测。

训练长度默认使用 `--total-timesteps`，也可以单独覆盖：

```powershell
--no-han-total-timesteps <steps>
```

## 3. 推荐命令

完整默认对比：

```powershell
python scripts\compare_system_baselines.py
```

对比已有系统训练目录：

```powershell
python scripts\compare_system_baselines.py `
  --run-mode compare_only `
  --system-run-dir results\full_train_latency_priority
```

快速 smoke 对比：

```powershell
python scripts\compare_system_baselines.py `
  --run-mode compare_only `
  --system-run-dir results\full_train_latency_priority `
  --device cpu `
  --episodes 1 `
  --max-steps 50 `
  --dqn-timesteps 500 `
  --no-han-total-timesteps 2048
```

自定义基线子集：

```powershell
python scripts\compare_system_baselines.py `
  --run-mode compare_only `
  --system-run-dir results\full_train_latency_priority `
  --baselines full-local joint-greedy dqn mappo-no-han
```

## 4. Reward 曲线样式

Reward 图现在使用：

- 原始 `mean_reward` 作为半透明波动背景。
- 轻度平滑后的 reward 作为前景实线趋势。
- 当旧历史文件没有 `mean_reward` 时，才回退使用 `recent_mean_reward`。

平滑窗口已做上限限制，避免曲线过于平滑。图中仍会保留适当震荡，使训练过程更接近真实 reward 波动。

横轴、纵轴、标题和 tick 格式保持项目原有绘图风格，不套用外部参考图的坐标轴。

## 5. 测试产物清理

smoke 运行后优先使用清理脚本：

```powershell
python scripts\cleanup_old_results.py
python scripts\cleanup_old_results.py --apply
```

未加 `--apply` 时只打印将要清理的目标。脚本会清理临时输出目录，例如：

```text
results/baseline_smoke_*
results/plot_*_smoke
results/test_han_integration
results/profile_tmp
pytest-cache-files-*
```

也应清理验证过程生成的 Python 缓存：

```text
__pycache__/
.pytest_cache/
```
