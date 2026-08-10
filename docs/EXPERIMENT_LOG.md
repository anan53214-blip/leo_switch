# Experiment Log

## 2026-08-08 — Restore HAN+MAPPO in fixed-user dashboard convergence panels

- Run directory: `results/baseline_compare/multiuser_scaling_multiuser_single_seed_150k_20260804`
- Mode: plot-only repair from existing training histories and comparison summaries; no training or evaluation was rerun.
- Command/config: regenerated `paper_baseline_dashboard.png` for u20, u25, u30, u35, and u40 with seed 42 and smoothing window 5.
- Code change: when an aggregated CSV contains a copied server-side training-history path, rebase its `results/...` suffix onto the current checkout before drawing fixed-user reward curves.
- Expected metric effect: none; the change only restores the HAN+MAPPO training curve and legend entry in the dashboard convergence panel.
- Reward components: unchanged.
- Diagnosis: aggregate-only plotting passed `history_path=None`; learned baselines were recovered from `uXX/learned_baselines`, but the system history remained an unavailable `/home/pjpjq/...` path and was skipped.
- Verification: the path-rebasing regression test and reward-curve tests pass; visual inspection of u20 and u40 confirms all five learned methods, including HAN+MAPPO, appear in the convergence panel.
- Follow-up decision: retain path rebasing so copied Linux experiment artifacts remain plot-compatible on Windows checkouts.

## 2026-08-08 — Complete u20–u40 baseline figure rebuild

- Run directory: `results/baseline_compare/multiuser_scaling_multiuser_single_seed_150k_20260804`
- Mode: plot-only aggregation from existing comparison CSV/JSON artifacts; no training or evaluation was rerun.
- Command: `C:\Users\19704\.conda\envs\satellite.env\python.exe scripts/run_multiuser_scaling_suite.py --run-id multiuser_single_seed_150k_20260804 --user-counts 20 25 30 35 40 --aggregate-only`
- Code changes:
  - Treat a blank `successful_task_throughput` CSV value as missing and derive it as `60 * completed_tasks / total_user_seconds`.
  - Remove the obsolete unsuffixed `success_continuity_tradeoff.png` when rebuilding fixed-user plots.
  - Prefer raw scalar methods from `comparison_summary.json`, then seed records, and sanitize already-aggregated CSV fallback fields so repeated in-place aggregation is idempotent.
  - Reapply the current throughput metric set and `hspace=0.38` multi-row layout by regenerating all fixed-user and multi-user figures.
- Key metrics: all 45 multi-user summary rows have positive throughput. Across the five user counts and nine methods, throughput ranges from `10.5641` to `20.0566` tasks per user-minute.
- Reward components/configuration: unchanged; this run reused existing experiment summaries and did not execute environment episodes, training, checkpoint selection, or reward calculation.
- Diagnosis: the completed long-running suite produced final artifacts with plotting code loaded before the partial-result plotting changes. Its CSVs also contained a `successful_task_throughput` header with blank values; `dict.setdefault()` treated the blank field as present, so aggregation preserved blanks and the scatter plot converted them to zero.
- Verification:
  - `40 passed` in `tests/test_paper_metrics.py`, `tests/test_reward_curve_plotting.py`, and `tests/test_config_default_consistency.py`.
  - u20–u40 each contain one regenerated `success_throughput_tradeoff.png`; no `success_continuity_tradeoff.png` remains.
  - All five fixed-user summaries and the 45-row multi-user summary contain zero recursively nested confidence-interval columns.
  - Visual inspection confirmed the fixed-user throughput data, the multi-user throughput panels, and non-overlapping subplot rows.
- Follow-up decision: keep the derived-metric normalization and legacy-artifact cleanup in the aggregate path so future plot-only rebuilds remain compatible with older or partially migrated CSV schemas.
