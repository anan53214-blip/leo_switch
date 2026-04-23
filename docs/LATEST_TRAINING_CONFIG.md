# 当前默认训练配置

> 更新日期：2026-04-15
> 权威来源：`results/full_train_delay_focus/training_history.json` 的 `config` 字段
> 已同步入口：`scripts/train.py`、`scripts/run_server_training.py`、`scripts/compare_system_baselines.py`

本文档记录当前应作为论文实验、复现实验和基线对比统一使用的训练参数。当前多目标默认权重已在训练入口、服务器批量入口和统一对比入口之间保持一致。旧版 `v1/v2/v3/v4` 过程性实验记录已从 `docs/` 移除，避免继续引用过时配置。

## 1. 实验标识

| 参数 | 当前值 |
|------|--------|
| `exp_name` | `han_mappo_delay_focus_fast` |
| `seed` | `42` |
| `device` | `cuda` |
| `save_path` | `results/full_train_delay_focus` |
| `log_path` | `results/logs` |

## 2. 环境与奖励权重

| 参数 | 当前值 |
|------|--------|
| `num_planes` | `6` |
| `sats_per_plane` | `11` |
| `altitude_km` | `550.0` |
| `inclination_deg` | `53.0` |
| `num_users` | `10` |
| `max_steps` | `2000` |
| `time_step_sec` | `1.0` |
| `max_visible_sats` | `10` |
| `reward_delay_weight` | `1.4` |
| `reward_energy_weight` | `0.4` |
| `reward_handover_weight` | `0.3` |
| `reward_load_balance_weight` | `0.1` |
| `reward_qos_weight` | `0.4` |

## 3. 网络结构

| 参数 | 当前值 |
|------|--------|
| `han_hidden_dim` | `64` |
| `han_out_dim` | `64` |
| `han_num_heads` | `4` |
| `han_num_layers` | `2` |
| `han_dropout` | `0.1` |
| `actor_hidden_dims` | `[256, 128]` |
| `critic_hidden_dims` | `[256, 256, 128]` |

当前 Actor 的连续卸载比例头使用 `Beta(alpha, beta)` 分布，不再使用早期文档中的 `Normal + clamp` 方案。

## 4. MAPPO 参数

| 参数 | 当前值 |
|------|--------|
| `learning_rate` | `3e-4` |
| `gamma` | `0.99` |
| `gae_lambda` | `0.95` |
| `clip_range` | `0.2` |
| `clip_range_vf` | `0.2` |
| `value_loss_coef` | `0.5` |
| `value_loss_type` | `huber` |
| `normalize_returns` | `true` |
| `value_huber_beta` | `10.0` |
| `entropy_coef` | `0.01` |
| `max_grad_norm` | `0.5` |
| `n_epochs` | `4` |
| `batch_size` | `256` |

## 5. 训练调度

| 参数 | 当前值 |
|------|--------|
| `total_timesteps` | `1_000_000` |
| `n_steps` | `2048` |
| `eval_interval` | `100_000` |
| `eval_episodes` | `3` |
| `graph_update_interval` | `100` |
| `save_interval` | `200_000` |
| `log_interval` | `1` |
| `early_stop_patience` | `30` |

## 6. 推荐命令

```bash
# 使用默认参数复现实验
python scripts/train.py

# 服务器训练入口，standard 已等价于当前默认实验参数
python scripts/run_server_training.py --plan standard

# 仅重新生成图表
python scripts/run_server_training.py --plot_only results/full_train_delay_focus

# 评估已训练模型
python scripts/train.py --load_path results/full_train_delay_focus/best_model.pt --eval_only
```

## 7. 一致性说明

当前 `scripts/train.py` 的默认命令行参数、`TrainConfig` 默认值，以及 `scripts/run_server_training.py` 的 `STANDARD_CONFIG` 已与 `full_train_delay_focus` 历史配置逐字段校验一致。`actor_hidden_dims` 和 `critic_hidden_dims` 在 Python 内部是 tuple，写入 JSON 后显示为 list，这是序列化形式差异，不是参数差异。
