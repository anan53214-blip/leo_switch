# 单目标训练改动说明（时延最小化 / 能耗最小化）

> 日期：2026-03-27  
> 仓库：`leo_switch`  
> 分支：`master`

## 1. 改动背景

当前系统原始优化目标为：

$$
\min_{\pi} \; \mathbb{E}\left[ \sum_{t=0}^{T} \left( w_1 \cdot T_{delay} + w_2 \cdot E_{energy} + w_3 \cdot C_{handover} - w_4 \cdot R_{QoS} \right) \right]
$$

本次改动目标：新增两个训练入口，分别进行**单目标优化**：

1. 时延最小化：
$$
\min_{\pi} \; \mathbb{E}\left[\sum_{t=0}^{T} T_{delay}\right]
$$

2. 能耗最小化：
$$
\min_{\pi} \; \mathbb{E}\left[\sum_{t=0}^{T} E_{energy}\right]
$$

---

## 2. 新增文件

### `scripts/train_delay_only.py`

- 作用：仅针对**时延最小化**训练。
- 方式：复用现有 `HANMAPPOTrainer` 训练链路，使用 `DelayOnlyEnv` 覆盖奖励。
- 奖励定义：
  - 记录 step 前 `total_delay`
  - 调用父类 `step`
  - 新奖励 `objective_reward = -(total_delay_after - total_delay_before)`

即每步最大化负时延增量，等价于最小化时延累计。

### `scripts/train_energy_only.py`

- 作用：仅针对**能耗最小化**训练。
- 方式：复用现有 `HANMAPPOTrainer` 训练链路，使用 `EnergyOnlyEnv` 覆盖奖励。
- 奖励定义：
  - 记录 step 前 `total_energy`
  - 调用父类 `step`
  - 新奖励 `objective_reward = -(total_energy_after - total_energy_before)`

即每步最大化负能耗增量，等价于最小化能耗累计。

---

## 3. 兼容性与实现说明

- 未修改原始 `scripts/train.py` 主训练脚本。
- 未修改环境动力学（切换、可见性、卸载、队列处理逻辑保持原样）。
- 新脚本通过子类方式覆盖奖励输出，降低对现有代码路径的侵入性。
- 两个新入口均支持与原训练脚本一致的常用参数（如 `--num_users`、`--total_timesteps`、`--eval_only` 等）。

---

## 4. 验证记录

已执行并通过：

1. 语法编译检查：
   - `scripts/train_delay_only.py`
   - `scripts/train_energy_only.py`

2. CLI 启动检查（`--help`）：
   - 两个脚本均能正常输出参数说明。

3. 小规模烟测（`eval_only`）：
   - 时延单目标脚本可初始化与评估。
   - 能耗单目标脚本可初始化与评估。

---

## 5. 运行示例（PowerShell）

### 时延最小化训练

```powershell
C:/Users/19704/.conda/envs/satellite.env/python.exe scripts/train_delay_only.py --total_timesteps 500000 --num_users 5
```

### 能耗最小化训练

```powershell
C:/Users/19704/.conda/envs/satellite.env/python.exe scripts/train_energy_only.py --total_timesteps 500000 --num_users 5
```

### 快速验证（仅评估）

```powershell
C:/Users/19704/.conda/envs/satellite.env/python.exe scripts/train_delay_only.py --eval_only --num_users 2 --max_steps 5 --device cpu
C:/Users/19704/.conda/envs/satellite.env/python.exe scripts/train_energy_only.py --eval_only --num_users 2 --max_steps 5 --device cpu
```

---

## 6. 已知问题（与本次功能无直接耦合）

- 部分测试文件存在 fixture 缺失（例如 `tests/test_channel.py`、`tests/test_user_task.py` 的若干用例）。
- 编辑器侧可能出现 `torch/numpy/gymnasium` 导入解析告警，但在已选 conda 环境下脚本可运行。

---

## 7. 后续建议

- 如需和原多目标训练做严格可比实验，建议统一：
  - 随机种子
  - `num_users` / `max_steps` / `total_timesteps`
  - 评估间隔与评估 episode 数
- 可扩展参数化版本（如 `--objective delay|energy`），减少脚本维护成本。

---

## 8. 二次增量更新（2026-03-27）

本节记录在上述基础上新增的能力，确保服务器训练流程与 `run_server_training.py` 的体验一致。

### 8.1 训练结果目录策略升级（每次运行独立目录）

已在以下文件生效：

- `scripts/train_delay_only.py`
- `scripts/train_energy_only.py`

新增参数：

- `--save_path`：显式指定最终保存目录（优先级最高）
- `--save_root`：当未指定 `save_path` 时的结果根目录
- `--run_name`：运行名后缀（未指定时默认使用时间戳）

默认行为：

- 若未指定 `--save_path`，每次训练都会自动生成独立目录：
  - Delay-only: `results/delay_only_train/<exp_name>_<timestamp>`
  - Energy-only: `results/energy_only_train/<exp_name>_<timestamp>`

这样可避免不同实验互相覆盖，并便于后续批量对比。

### 8.2 自动可视化增强

两个单目标脚本均支持训练结束后自动调用绘图：

- 默认：自动生成图表
- `--no_plot`：关闭自动绘图
- `--plot_window`：指定平滑窗口大小

图表输出目录固定为：`<save_path>/figures`。

### 8.3 `plot_results.py` 单目标语义适配

已在 `scripts/plot_results.py` 增加目标识别与图表语义调整：

- 自动识别 `delay_only` / `energy_only` / `multi_objective`
- 在单目标场景下：
  - 奖励图标题与纵轴改为目标奖励语义（如 `-ΔDelay` / `-ΔEnergy`）
  - Delay/Energy 双图中，主优化指标作为主展示，另一项作为参考指标
  - Dashboard 总标题和关键子图标题按目标类型切换
- 在 `plot_info.json` 中新增 `objective` 字段

### 8.4 图表标题与坐标轴语言规范

按当前要求，图内文本（`set_xlabel` / `set_ylabel` / `set_title` / `suptitle`）统一使用英文。

说明：中文主要保留在注释、文档字符串和终端提示，不影响图表论文展示。

---

## 9. 更新后的运行示例

### 9.1 自动创建独立结果目录（推荐）

```powershell
C:/Users/19704/.conda/envs/satellite.env/python.exe scripts/train_delay_only.py --total_timesteps 500000 --num_users 10
C:/Users/19704/.conda/envs/satellite.env/python.exe scripts/train_energy_only.py --total_timesteps 500000 --num_users 10
```

### 9.2 指定结果根目录 + 自定义运行名

```powershell
C:/Users/19704/.conda/envs/satellite.env/python.exe scripts/train_delay_only.py --save_root results/delay_only_train --run_name server_run_01
C:/Users/19704/.conda/envs/satellite.env/python.exe scripts/train_energy_only.py --save_root results/energy_only_train --run_name server_run_01
```

### 9.3 显式指定最终目录（手动管理）

```powershell
C:/Users/19704/.conda/envs/satellite.env/python.exe scripts/train_delay_only.py --save_path results/my_delay_exp
C:/Users/19704/.conda/envs/satellite.env/python.exe scripts/train_energy_only.py --save_path results/my_energy_exp
```
