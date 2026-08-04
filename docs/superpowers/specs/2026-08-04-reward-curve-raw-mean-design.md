# Reward 曲线统一使用原始 Mean Reward 设计

日期：2026-08-04

## 目标

所有训练过程绘图统一读取训练历史中的原始 `mean_reward`，不再读取或回退到
`recent_mean_reward`。所有按 step 绘图的默认平滑窗口从 5 调整为 3，并确保
每条曲线只执行一次绘图层移动平均。

## 范围

修改以下三个绘图入口：

- `scripts/compare_system_baselines.py`
- `scripts/run_multiuser_scaling_suite.py`
- `scripts/plot_training_artifacts.py`

窗口调整覆盖 reward 收敛图、dashboard reward 面板、QoS 指标图、
reward component 图及多用户聚合训练曲线。

训练器仍可继续把 `recent_mean_reward` 写入 history，供训练日志、
early stopping 和旧数据诊断使用；本次只取消绘图对该字段的使用，避免改变训练行为
或破坏旧 checkpoint/history 兼容性。

## 数据流

绘图数据统一采用：

1. 过滤 `partial_episode=True` 的不完整记录；
2. 从每条训练记录读取 `mean_reward`；
3. 旧格式缺少 `mean_reward` 时，只允许回退到原始语义字段
   `eval_mean_reward` 或 `reward`，不回退到 `recent_mean_reward`；
4. 按 `total_steps` 排序；
5. 执行一次窗口为 3 的移动平均；
6. 绘制曲线。

训练 history 中已有的 `recent_mean_reward` 不参与上述流程。

## 默认参数

所有公开绘图入口和多用户实验配置中的 `plot_window` 默认值统一为 3。
显式传入其他正整数时仍尊重调用方参数，以保留命令行可配置性。

## 测试

采用回归测试验证：

- 同一记录同时包含不同的 `mean_reward` 和 `recent_mean_reward` 时，
  reward 曲线必须返回 `mean_reward`；
- 多用户聚合曲线加载器必须返回 `mean_reward`；
- 三个绘图入口的默认窗口均为 3；
- 3 点移动平均输出符合预期；
- 现有相关测试继续通过。

## 产物验证

使用已有 U20 `training_history.json` 和 `comparison_summary.json` 重新生成：

- `reward_curve_vs_baselines.png`
- `paper_baseline_dashboard.png`
- `training_qos_metrics_vs_steps.png`
- `reward_components_vs_steps.png`
- `reward_components_per_task_vs_steps.png`

不重新训练、不重新评估模型。验证主 reward 图仍包含五种学习算法，并目视确认
曲线保留真实训练波动。
