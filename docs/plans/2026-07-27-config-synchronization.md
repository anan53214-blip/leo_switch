# Configuration Synchronization Implementation Plan

> **For Codex:** Follow the repository's test-driven workflow and complete each
> task in order.

**Goal:** Align all behavioral defaults with
`multiuser_scaling_multiuser_6_7` and prevent future drift between environment,
training, CLI, comparison, and suite configurations.

**Architecture:** Keep `EnvConfig` as the source of truth for environment and
reward defaults. Keep `TrainConfig` as the source of truth for training and
evaluation cadence. Derive CLI and comparison defaults from those classes,
while leaving run-specific names, paths, devices, and user counts configurable.

**Tech Stack:** Python 3.10, dataclasses, argparse, pytest.

---

## Task 1: Lock the reference configuration with failing tests

**Files:**

- Create: `tests/test_config_default_consistency.py`
- Reference: `tests/test_reward_function.py`
- Reference: `tests/test_han_integration.py`

### Step 1: Add the canonical expected values

Define expected mappings for:

- environment/reward behavior;
- training and evaluation cadence;
- model-selection metric.

The mappings must match the five original training histories referenced by the
target suite.

### Step 2: Add consistency tests

Test that:

- `EnvConfig` matches the environment/reward reference;
- `TrainConfig` matches the complete behavioral reference;
- `train.parse_args()` with no arguments matches `TrainConfig`;
- `build_default_train_config()` preserves canonical defaults;
- `MultiUserConfig` and the latency-priority `SuiteConfig` match the canonical
  training cadence and selection metric.

### Step 3: Run the new tests and confirm RED

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests/test_config_default_consistency.py -q
```

Expected: failures for the stale reward, duration, cadence,
`graph_update_interval`, and model-selection defaults.

## Task 2: Establish canonical environment and training defaults

**Files:**

- Modify: `src/environment/gym_env.py`
- Modify: `scripts/train.py`
- Test: `tests/test_config_default_consistency.py`
- Test: `tests/test_reward_function.py`
- Test: `tests/test_han_integration.py`

### Step 1: Update `EnvConfig`

Set `max_steps=600`. Keep the already-correct delay-priority reward values:

```python
reward_delay_weight = 0.35
reward_energy_weight = 0.05
reward_handover_weight = 0.10
reward_load_balance_weight = 0.05
reward_qos_weight = 0.40
reward_service_continuity_weight = 0.15
reward_deadline_slack_weight = 0.25
reward_enqueue_bonus = 0.0
reward_failed_handover_penalty = 0.3
reward_deadline_penalty = 1.0
reward_failed_task_penalty = 0.8
```

### Step 2: Make `TrainConfig` reuse environment defaults

Reference the corresponding `EnvConfig` class defaults for environment and
reward fields, then set:

```python
total_timesteps = 300_000
eval_interval = 50_000
eval_episodes = 3
graph_update_interval = 1
save_interval = 100_000
log_interval = 1
best_model_metric = "avg_delay"
```

Keep the existing `n_steps=1024`, `n_epochs=6`, `batch_size=256`,
`entropy_coef=0.005`, and `target_update_interval=500`.

### Step 3: Derive training CLI defaults from `TrainConfig`

Create a default `TrainConfig` inside `parse_args()` and use its fields for the
behavioral CLI defaults. Preserve `device=auto` as the CLI default.

### Step 4: Run focused tests

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests/test_config_default_consistency.py tests/test_reward_function.py tests/test_han_integration.py -q
```

Expected: all tests pass.

## Task 3: Remove baseline-comparison overrides

**Files:**

- Modify: `scripts/compare_system_baselines.py`
- Test: `tests/test_config_default_consistency.py`
- Test: `tests/test_compare_system_baselines_config.py`
- Test: `tests/test_offpolicy_evaluation.py`

### Step 1: Derive comparison constants

Set the comparison training budget and default selection metric from
`TrainConfig`.

### Step 2: Remove duplicate reward assignments

Keep `build_default_train_config()` responsible only for contextual overrides:
seed, maximum steps, user count, output path, experiment name, and the explicit
selection metric. Let all reward and cadence fields come from `TrainConfig`.

### Step 3: Run comparison tests

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests/test_config_default_consistency.py tests/test_compare_system_baselines_config.py tests/test_offpolicy_evaluation.py -q
```

Expected: all tests pass.

## Task 4: Synchronize current documentation

**Files:**

- Modify: `README.md`
- Modify: `docs/COMPARE_SYSTEM_BASELINES_CLI.md`
- Modify: `docs/REWARD_WEIGHT_CONFIG.md`
- Modify: `docs/EXPERIMENT_LOG.md`

### Step 1: Update current defaults and examples

Document the 300k training budget, 600-step episodes, evaluation/checkpoint/
graph-refresh cadence, and `avg_delay` model selection.

### Step 2: Correct the reward configuration record

Mark the 2026-07-06 multi-objective values as superseded and document the
reference suite's delay-priority weights as the active defaults.

### Step 3: Preserve historical records

Do not edit old experiment entries. Append a dated entry describing:

- the configuration source used;
- the drift root cause;
- the synchronized files;
- the expected effect on future training and comparisons;
- that no experiment was run as part of the configuration-only change.

## Task 5: Final verification

**Files:**

- Verify all modified files.

### Step 1: Check for stale active defaults

Search active code and current documentation for the superseded values and
cadence:

```powershell
rg -n --glob '!results/**' "1_200_000|1200000|graph_update_interval.*100|eval_interval.*100_000|save_interval.*200_000|reward_energy_weight.*0\.30|reward_deadline_penalty.*0\.70" scripts src README.md docs
```

Any remaining match must be either an explicitly marked historical record or
an unrelated algorithm-specific setting.

### Step 2: Run the complete test suite

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest -q
```

Expected: all tests pass.

### Step 3: Review the final diff

Run:

```powershell
git diff --check
git status --short
git diff --stat
```

Confirm that no result artifacts, run-specific paths, device settings, or
historical experiment records were unintentionally changed.

