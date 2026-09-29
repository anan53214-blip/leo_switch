# 0907 多种子聚合与复跑核查

`results/multiseed_0907/` 现包含训练 seed 43–46 的四种子结果；历史 seed42 没有进入这组统计。2026-09-28 按用户要求，将 U20/seed43 和 U30/seed45 的复跑模型及统一评估结果写回原目录，并覆盖多种子图。原始低分实验的 CSV、JSON、权重和日志保存在 `results/multiseed_0907/archive_pre_rerun_20260928/`；两处临时目录已在核验后删除。

## 原始四种子结果与复跑

| 场景 | 原始 HAN+PDQN 奖励 | 原始最强对手 | 复跑 HAN+PDQN 奖励 | 复跑最强对手 |
|---|---:|---|---:|---|
| U20/seed43 | 60.12 | HAN+MAPPO 99.39 | 107.63 | HAN+MAPPO 99.39 |
| U30/seed45 | 33.20 | PDQN 80.35 | 94.05 | PDQN 80.35 |

原始训练日志显示，两组均从预训练 HAN 全新开始，连续完成 150,000 步，没有 `load_path` 续训或步数回退。最长相邻更新间隔分别约 181 秒和 70 秒，包含评估时间；没有发现断电后恢复错乱的直接证据。原始最佳 checkpoint 正确用于当时比较。原始四种子结果中，HAN+PDQN 在 20 个配对场景对当次奖励最高的对手胜出 18 次。

两次复跑使用原训练 seed、参数和预训练 HAN，均完整完成 150,000 步。新 checkpoint 又在各自原有的 5 个测试 seed 和相同环境配置下评估，其他六种方法的奖励与原结果逐一核对相同。来源、模型 SHA256 和替换清单见 `results/multiseed_0907/rerun_promotion_20260928.json`，训练命令及结果见 `rerun_status_20260928.json` 和 `rerun_script_20260928.py`。

当前四种子均值中，HAN+PDQN 在 U20、U25、U30、U35、U40 的**平均奖励**均为最高；逐 seed 对当次最强对手的奖励胜出数为 20/20。`multiuser_paired_reward_advantage.png` 展示全部 20 个配对点，明细在同名 CSV。红线为中位差值，只作分布描述。

当前聚合可在项目根目录复现：

```bash
/home/pjpjq/miniconda3/envs/satellite.env/bin/python scripts/run_multiuser_scaling_suite.py \
  --run-id promoted_rerun_20260928 --results-root results/multiseed_0907 \
  --seeds 43 44 45 46 --aggregate-only --proposed-method han_pdqn
```

**解释限制：**原始低分运行已正常完成，尚无证据证明它们无效。只替换两次低分训练会使统计均值向上偏；论文中应同时报告本次结果与归档的原始结果，或预先固定规则对所有 seed 一致复跑后重新统计。seed42 是历史筛选版且含单独补评的 SCA，本次未追加。

seed43–46 的原始主方法键为 `han_pdqn_multiuser_uXX`（训练实验名）；聚合读取时规范为 `han_pdqn`，原键写入 `source_method`。历史 HAN+MAPPO 和其他方法的身份不变。

## 图表复核（删除后补做）

用户追加要求逐图核对时，两个临时目录已经在上一轮删除。因此在主结果目录上补做独立审计，记录于 `results/multiseed_0907/chart_audit_20260928.json`。325 张 PNG 全部完整解码；171 张聚合图重新绘制后 SHA256 完全一致，61 份聚合 CSV 完全一致；两组被替换运行的 12 张原始比较图用 checkpoint 重新评估绘制后也逐字节一致。另核对了 140 个方法结果及其 700 条测试回合、385 个聚合指标均值和置信区间、100 份训练历史及 220 个可操作文件路径，未发现异常。剩余 108 张未改动的原始比较图与 34 张 seed42 历史图通过完整解码和底层结果核对，未逐张重绘。
