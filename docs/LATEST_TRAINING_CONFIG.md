# 当前训练配置摘要

本文档记录当前代码中主要训练配置，便于复现实验和检查参数是否与论文描述一致。以源码为准，重点参考 `scripts/train.py`、`src/environment/gym_env.py`、`src/environment/mec.py` 和 `scripts/compare_system_baselines.py`。

## 1. 实验基础配置

| 参数 | 当前默认值 |
| --- | --- |
| `exp_name` | `han_mappo_delay_focus_fast` |
| `seed` | `42` |
| `device` | `cuda` 可用时使用 CUDA，否则使用 CPU |
| `save_path` | `results/full_train_delay_focus` |
| `log_path` | `results/logs` |
| `best_model_metric` | `reward` |

统一基线对比脚本的默认关注指标是：

```text
latency_priority_score
```

## 2. 星座与环境参数

| 参数 | 当前默认值 |
| --- | --- |
| `num_planes` | `6` |
| `sats_per_plane` | `11` |
| 卫星总数 | `66` |
| `altitude_km` | `550.0` |
| `inclination_deg` | `53.0` |
| `num_users` | `10` |
| `max_steps` | `2000` |
| `time_step_sec` | `1.0` |
| `max_visible_sats` | `10` |
| `task_arrival_prob` | `0.45` |

## 3. Reward 权重

| 参数 | 当前默认值 | 含义 |
| --- | --- | --- |
| `reward_delay_weight` | `1.4` | 时延项权重 |
| `reward_energy_weight` | `0.4` | 能耗项权重 |
| `reward_handover_weight` | `0.3` | 切换收益与切换代价权重 |
| `reward_load_balance_weight` | `0.1` | 负载均衡权重 |
| `reward_qos_weight` | `0.4` | QoS 奖励权重 |
| `reward_enqueue_bonus` | `0.02` | 成功入队奖励 |
| `reward_queue_full_penalty` | `0.3` | 队列满惩罚 |

## 4. MEC 参数

| 参数 | 当前默认值 |
| --- | --- |
| `satellite_cpu_freq_ghz` | `5.0` |
| `satellite_max_cpu_freq_ghz` | `8.0` |
| `max_queue_size` | `20` |
| `user_cpu_freq_ghz` | `0.5` |
| `user_max_cpu_freq_ghz` | `1.0` |

这些设置会让卫星 MEC 队列存在竞争，同时保留本地计算能力较弱的对比场景。

## 5. HAN 与 MAPPO 参数

| 参数 | 当前默认值 |
| --- | --- |
| `han_hidden_dim` | `64` |
| `han_out_dim` | `64` |
| `han_num_heads` | `4` |
| `han_num_layers` | `2` |
| `han_dropout` | `0.1` |
| `actor_hidden_dims` | `[256, 128]` |
| `critic_hidden_dims` | `[256, 256, 128]` |
| `learning_rate` | `3e-4` |
| `gamma` | `0.99` |
| `gae_lambda` | `0.95` |
| `clip_range` | `0.2` |
| `value_loss_coef` | `0.5` |
| `entropy_coef` | `0.01` |
| `max_grad_norm` | `0.5` |
| `n_epochs` | `4` |
| `batch_size` | `256` |

## 6. 训练长度与评估

| 参数 | 当前默认值 |
| --- | --- |
| `total_timesteps` | `1_200_000` |
| `n_steps` | `2048` |
| `eval_interval` | `100_000` |
| `eval_episodes` | `3` |
| `graph_update_interval` | `100` |
| `save_interval` | `200_000` |
| `log_interval` | `1` |
| `early_stop_patience` | `0` |

## 7. 当前基线对比配置

`--baselines all` 默认包含：

- `random`
- `min_distance`
- `full_local`
- `joint_greedy`
- `dqn`
- `mappo_no_han`

已移除：

- `stay`
- `max_elev`
- `max_rvt`
- `threshold_rvt`

DQN 相关参数：

```powershell
--dqn-offload-grid 0.0 0.5 1.0
--dqn-timesteps <steps>
```

MAPPO 无 HAN 消融相关参数：

```powershell
--no-han-total-timesteps <steps>
```

## 8. 推荐复现实验命令

训练主方法：

```powershell
python scripts\train.py `
  --num_users 10 `
  --max_steps 2000 `
  --total_timesteps 1200000 `
  --save_path results\full_train_delay_focus
```

按时延优先指标保存最优模型：

```powershell
python scripts\train.py `
  --best-model-metric latency_priority_score `
  --save_path results\full_train_latency_priority
```

对比已有系统结果：

```powershell
python scripts\compare_system_baselines.py `
  --run-mode compare_only `
  --system-run-dir results\full_train_latency_priority `
  --episodes 5 `
  --compare-ranking-metric latency_priority_score
```
