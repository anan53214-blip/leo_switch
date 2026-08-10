# Complete Multi-User Plot Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the completed u20-u40 comparison artifacts consistently use successful-task throughput and the corrected subplot spacing.

**Architecture:** Normalize derived paper metrics while reading comparison CSV rows so a present-but-empty derived field is treated as missing, then regenerate all fixed-user and scaling plots from the existing experiment data. Remove the obsolete continuity trade-off artifact during fixed-user regeneration so output directories match the current metric set.

**Tech Stack:** Python 3.10, csv, Matplotlib, pytest

---

### Task 1: Reproduce empty throughput normalization

**Files:**
- Modify: `tests/test_paper_metrics.py`
- Modify: `scripts/paper_metrics.py`

- [x] **Step 1: Write the failing test**

```python
def test_derived_throughput_replaces_blank_csv_value():
    result = derive_paper_metrics(
        {
            "successful_task_throughput": "",
            "completed_tasks": "8",
            "total_user_seconds": "40",
        }
    )

    assert result["successful_task_throughput"] == pytest.approx(12.0)
```

- [x] **Step 2: Run test to verify it fails**

Run: `C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests/test_paper_metrics.py::test_derived_throughput_replaces_blank_csv_value -v`

Expected: FAIL because the returned value is the original empty string.

- [x] **Step 3: Write minimal implementation**

Replace only missing or blank `successful_task_throughput` values with `60 * completed_tasks / total_user_seconds`; preserve explicit numeric values, including zero.

- [x] **Step 4: Run test to verify it passes**

Run the same targeted pytest command.

Expected: PASS.

### Task 2: Remove the obsolete fixed-user continuity plot

**Files:**
- Modify: `tests/test_paper_metrics.py`
- Modify: `scripts/run_multiuser_scaling_suite.py`

- [x] **Step 1: Write the failing test**

Add a fixed-user aggregate test that creates `success_continuity_tradeoff.png`, invokes `_plot_fixed_user_seed_summary`, and asserts the obsolete file no longer exists.

- [x] **Step 2: Run test to verify it fails**

Run: `C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests/test_paper_metrics.py::test_fixed_user_aggregate_removes_obsolete_continuity_plot -v`

Expected: FAIL because the legacy image remains.

- [x] **Step 3: Write minimal implementation**

Before generating the current fixed-user plots without an output suffix, unlink `success_continuity_tradeoff.png` if it exists. Do not remove suffixed or unrelated artifacts.

- [x] **Step 4: Run test to verify it passes**

Run the same targeted pytest command.

Expected: PASS.

### Task 3: Rebuild and verify the completed experiment artifacts

**Files:**
- Modify: `results/baseline_compare/multiuser_scaling_multiuser_single_seed_150k_20260804/**`
- Create or modify: `docs/EXPERIMENT_LOG.md`

- [x] **Step 1: Run focused and related tests**

Run: `C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests/test_paper_metrics.py tests/test_reward_curve_plotting.py tests/test_config_default_consistency.py -q`

Expected: all tests pass.

- [x] **Step 2: Regenerate from existing experiment CSV/JSON data**

Run: `C:\Users\19704\.conda\envs\satellite.env\python.exe scripts/run_multiuser_scaling_suite.py --run-id multiuser_single_seed_150k_20260804 --user-counts 20 25 30 35 40 --aggregate-only`

Expected: u20-u40 fixed-user plots and all multi-user aggregate plots are regenerated without training.

- [x] **Step 3: Validate data and image outputs**

Assert every regenerated user summary has a positive `successful_task_throughput`, `multiuser_core_metrics.png` contains the throughput panel data, no `success_continuity_tradeoff.png` remains, and rendered plots have no title/axis-label overlap.

- [x] **Step 4: Record the plotting experiment**

Document the run directory, aggregate-only command, metric normalization fix, layout behavior, generated artifacts, and verification outcome in `docs/EXPERIMENT_LOG.md`.

### Task 4: Keep in-place aggregation idempotent

**Files:**
- Modify: `tests/test_paper_metrics.py`
- Modify: `scripts/run_multiuser_scaling_suite.py`

- [x] **Step 1: Write failing repeated-aggregation tests**

Run the same in-place aggregation twice with raw JSON methods present and absent; assert headers remain stable and contain no recursively nested CI fields.

- [x] **Step 2: Verify the legacy CSV fallback fails before the fix**

Run: `C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest 'tests/test_paper_metrics.py::test_in_place_aggregate_is_idempotent[False]' -v`

Expected before the fix: FAIL with `_ci_low_ci_low` in the second header.

- [x] **Step 3: Implement stable raw-source selection**

Prefer scalar method summaries from JSON, then an existing `comparison_seed_records.csv`; if only an aggregated summary CSV exists, drop prior CI and derived sample-count fields before aggregation.

- [x] **Step 4: Verify both source variants pass**

Run: `C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests/test_paper_metrics.py::test_in_place_aggregate_is_idempotent -v`

Expected: both parameterized cases PASS.
