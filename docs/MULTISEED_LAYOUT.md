# 多种子实验目录（2026-09-08）

默认根目录为 `results/multiseed_0907`，可用 `--results-root` 更换。
`--run-id` 现在仅记录实验标识；用 `--legacy-layout` 才恢复旧版按 run-id 分目录的行为。
单种子和多种子均使用相同结构：

```text
results/multiseed_0907/
  seed42/
    logs/
    u20/                         # 另有 u25、u30、u35、u40
      learned_baselines/
        han_pdqn/                # 主方法：权重、training_history.json
        han_mappo/
        mappo_no_han/
        maddpg/
        pdqn/                    # 无 HAN 基线
      sca/                       # 可选：独立配对评估、诊断、来源清单
      comparison_summary.json    # 原始统一评估数据及协议
      comparison_summary.csv
      comparison_seed_records.csv
      *.png                      # 各用户数对比图
    multiuser_summary.csv
    multiuser_seed_records.csv
    multiuser_*.png               # 4 张跨用户数汇总图
    suite_manifest.json
  seed43/
  ...                            # 多种子时根目录另生成跨种子统计与图表
```

规则方法没有训练目录或训练收敛曲线；SCA 也没有训练曲线。默认七种方法保留不变。
主方法的 `scripts/train.py` 通过套件传入 `--save_path` 写入上述目录；单独运行该脚本仍可自选保存位置。

## 新训练

在已安装依赖的 Python 环境运行。预训练路径必须替换为真实且 schema 匹配的 HAN 权重，可使用用户数和种子占位符。
现有 seed42 是历史筛选结果，不要直接覆盖；示例使用新种子。

```bash
python scripts/run_multiuser_scaling_suite.py --seeds 43 44 --pretrained-han-path "results/han_pretrain_u{num_users}_seed{seed}/best_model.pt" --pdqn-lr 0.001 --dry-run
```

确认命令后去掉 `--dry-run` 执行。指定 `--with-sca` 可在每组训练与基线对比后调用独立 SCA 评估入口。
HAN+PDQN 使用冻结的预训练 HAN、replay buffer 和 `pdqn_lr`；PPO 参数仍供 PPO 基线使用。
已有完整且配置匹配的主方法训练自动复用；`--reuse-learned-checkpoints` 复用学习基线权重。
不完整或配置不匹配的训练目录会报错，应换结果根目录，或明确使用 `--force-system-train` 重启主方法训练。

## 后补 SCA

SCA 不需要训练，它固定 HAN+PDQN 的离散规则并优化连续卸载比例。

```bash
python scripts/evaluate_sca_baseline.py --checkpoint results/multiseed_0907/seed43/u20/learned_baselines/han_pdqn/best_model.pt --output-dir results/multiseed_0907/seed43/u20/sca --episodes 5 --max-steps 512
python scripts/run_multiuser_scaling_suite.py --seeds 43 44 --aggregate-only
```

对各用户数、种子分别评估；SCA 输出目录必须是新目录。已有 SCA 清单时套件不重复执行求解。
汇总自动读取 `u*/sca` 中的 `han_pdqn_sca` 行，不重复加入其配对 HAN+PDQN 对照。
合并会核对完整状态、环境 schema、环境配置、显式测试种子和主方法 checkpoint SHA256；不一致时拒绝合并。
失败的 SCA 目录应保留诊断，修复后将旧目录移开再运行。没有 SCA 目录时正常画七方法图；只有部分组有 SCA 时仅展示已有组，不补值。

## 重绘已有 seed42

现有筛选版仅含 CSV，缺少原始 comparison_summary.json。显式允许 CSV 重绘，并另存以保留历史结果：

```bash
python scripts/run_multiuser_scaling_suite.py --seed 42 --aggregate-only --allow-csv-only --aggregate-output-dir results/replot_0907
```

生成 `results/replot_0907/seed42` 下的 34 张图。历史训练路径优先解析到原 seed42 下各算法目录。
`source_is_system` 保留旧运行角色，重绘的 `is_system` 默认对应 `han_pdqn`。
`--allow-csv-only` 不能用于绕过 SCA 的可比性检查：历史七方法结果需要补齐真实评估协议，通常应使用对应权重重新统一评估，不能凭空补写元数据。
重绘旧版 run-id 目录时追加 `--legacy-layout`，并继续指定原 `--run-id`。
