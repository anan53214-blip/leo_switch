# HAN MAPPO Raw Plus HAN Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change HAN+MAPPO from using `HAN embedding + rvt/task` observations to using `raw environment observation + HAN embedding + rvt/task`, so the experiment can test whether HAN provides incremental graph information instead of replacing the raw state signal.

**Architecture:** Keep the current frozen/cached HAN encoder path and MAPPO algorithm unchanged for this first diagnostic change. Update only the observation construction in `HANMAPPOTrainer`, simplify the duplicated HAN+PDQN observation override so it does not prepend raw observations twice, and add regression tests that compare HAN+MAPPO against the existing HAN+PDQN raw-plus-HAN behavior.

**Tech Stack:** Python, NumPy, PyTorch, Gymnasium environment in `src/environment/gym_env.py`, HAN+MAPPO trainer in `scripts/train.py`, pytest.

---

## File Structure

- Modify `scripts/train.py`
  - `HANMAPPOTrainer._init_environment`: set MAPPO observation dimension to `raw_obs_dim + han_out_dim + 5`.
  - `HANMAPPOTrainer._encode_graph_state`: concatenate raw observations, HAN user embeddings, and light `rvt/task` features.
  - `HANPDQNTrainer._init_environment` and `_encode_graph_state`: remove the old raw-prepending override or make it delegate to the new base behavior.
- Modify `tests/test_han_integration.py`
  - Add a failing regression test for HAN+MAPPO observation layout.
  - Update the existing HAN+PDQN observation test to assert it uses the same layout without double raw observations.
- Modify `docs/EXPERIMENT_LOG.md`
  - Add a code-change intent entry after implementation, before new experiment results are interpreted.
- Do not modify reward weights in this plan.
- Do not make HAN encoder trainable in this plan.
- Do not change `graph_update_interval` in this plan.

## Success Criteria

- `HANMAPPOTrainer.obs_dim == raw_obs_dim + han_out_dim + 5`.
- `HANMAPPOTrainer._encode_graph_state()` returns observations where:
  - columns `[0:raw_obs_dim]` equal `env._get_observation()`;
  - columns `[raw_obs_dim:raw_obs_dim + han_out_dim]` are HAN user embeddings;
  - final 5 columns are `rvt_warning + task_features`.
- `HANPDQNTrainer` still has the same layout and does not become `raw + raw + HAN + light`.
- Existing MAPPO action sampling, candidate masks, candidate satellite embeddings, rollout buffer storage, and checkpoint save/load still work for newly trained models.
- The g1 diagnostic suite can be launched under a new run id without reusing old checkpoints.

---

### Task 1: Add Failing Observation Layout Tests

**Files:**
- Modify: `tests/test_han_integration.py`

- [ ] **Step 1: Add a HAN+MAPPO raw-plus-HAN regression test**

Add this test immediately after `test_mappo_act_rejects_misaligned_candidate_mask_shape`:

```python
def test_han_mappo_observation_includes_raw_obs_han_and_light_features():
    run_id = uuid4().hex
    save_path = f"results/han_mappo_obs_{run_id}"
    log_path = f"results/han_mappo_obs_logs_{run_id}"
    config = TrainConfig(
        num_users=2,
        max_steps=10,
        total_timesteps=64,
        n_steps=16,
        batch_size=16,
        eval_episodes=0,
        device="cpu",
        save_path=save_path,
        log_path=log_path,
    )
    trainer_obj = HANMAPPOTrainer(config)
    try:
        trainer_obj.env.reset(seed=trainer_obj.config.seed)
        observations, _, _, _ = trainer_obj._encode_graph_state()
        raw_obs = trainer_obj.env._get_observation()

        expected_dim = trainer_obj.raw_obs_dim + trainer_obj.config.han_out_dim + 5
        assert trainer_obj.obs_dim == expected_dim
        assert observations.shape == (trainer_obj.num_agents, expected_dim)
        assert np.allclose(observations[:, : trainer_obj.raw_obs_dim], raw_obs)
    finally:
        trainer_obj.env.close()
        shutil.rmtree(save_path, ignore_errors=True)
        shutil.rmtree(log_path, ignore_errors=True)
```

- [ ] **Step 2: Strengthen the existing HAN+PDQN layout test**

In `test_han_pdqn_observation_includes_raw_obs_han_and_light_features`, keep the existing assertions and add this check after the `np.allclose` assertion:

```python
        han_start = trainer_obj.raw_obs_dim
        han_end = han_start + trainer_obj.config.han_out_dim
        light_features = observations[:, han_end:]
        assert light_features.shape == (trainer_obj.num_agents, 5)
```

- [ ] **Step 3: Run the new failing test**

Run:

```powershell
& 'C:\Users\19704\.conda\envs\satellite.env\python.exe' -m pytest tests\test_han_integration.py::test_han_mappo_observation_includes_raw_obs_han_and_light_features -q
```

Expected before implementation:

```text
FAILED
assert trainer_obj.obs_dim == expected_dim
```

- [ ] **Step 4: Commit the failing test**

```powershell
git add tests/test_han_integration.py
git commit -m "test: require raw observations in han mappo inputs"
```

---

### Task 2: Update HAN+MAPPO Observation Construction

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Update `HANMAPPOTrainer._init_environment` dimensions**

Find this block in `HANMAPPOTrainer._init_environment`:

```python
self.raw_obs_dim = self.env.user_obs_dim
self.han_out_dim = self.config.han_out_dim
# 最终观察 = HAN嵌入(64) + rvt_warning(1) + task_features(4) = 69
self.obs_dim = self.han_out_dim + 5
```

Replace it with:

```python
self.raw_obs_dim = self.env.user_obs_dim
self.han_out_dim = self.config.han_out_dim
# Final observation = raw env obs + HAN embedding + rvt_warning(1) + task_features(4).
self.obs_dim = self.raw_obs_dim + self.han_out_dim + 5
```

Find the log line:

```python
self.logger.info(f"  - 拼接后观测维度: {self.obs_dim} (HAN {self.han_out_dim} + rvt_warning 1 + task 4)")
```

Replace it with:

```python
self.logger.info(
    f"  - 拼接后观测维度: {self.obs_dim} "
    f"(raw {self.raw_obs_dim} + HAN {self.han_out_dim} + rvt_warning 1 + task 4)"
)
```

- [ ] **Step 2: Add a small raw-observation helper to `HANMAPPOTrainer`**

Add this method above `_encode_graph_state`:

```python
def _raw_policy_observations(self) -> np.ndarray:
    raw_observations = np.asarray(self.env._get_observation(), dtype=np.float32)
    if raw_observations.ndim == 1:
        raw_observations = raw_observations.reshape(1, -1)
    expected_shape = (self.num_agents, self.raw_obs_dim)
    if raw_observations.shape != expected_shape:
        padded = np.zeros(expected_shape, dtype=np.float32)
        copy_rows = min(raw_observations.shape[0], self.num_agents)
        copy_cols = min(raw_observations.shape[1], self.raw_obs_dim)
        padded[:copy_rows, :copy_cols] = raw_observations[:copy_rows, :copy_cols]
        raw_observations = padded
    return raw_observations.astype(np.float32, copy=False)
```

- [ ] **Step 3: Concatenate raw observations in `_encode_graph_state`**

Find the final observation construction:

```python
user_embeddings = np.concatenate([self._cached_han_user_embed, rvt_warning, task_features], axis=1)
return user_embeddings, self._cached_sat_embed, available_actions, candidate_sat_ids
```

Replace it with:

```python
raw_observations = self._raw_policy_observations()
light_features = np.concatenate([rvt_warning, task_features], axis=1)
user_embeddings = np.concatenate(
    [raw_observations, self._cached_han_user_embed, light_features],
    axis=1,
).astype(np.float32, copy=False)
return user_embeddings, self._cached_sat_embed, available_actions, candidate_sat_ids
```

- [ ] **Step 4: Run the HAN+MAPPO layout test**

Run:

```powershell
& 'C:\Users\19704\.conda\envs\satellite.env\python.exe' -m pytest tests\test_han_integration.py::test_han_mappo_observation_includes_raw_obs_han_and_light_features -q
```

Expected after implementation:

```text
1 passed
```

- [ ] **Step 5: Commit the HAN+MAPPO observation change**

```powershell
git add scripts/train.py tests/test_han_integration.py
git commit -m "feat: include raw observations in han mappo inputs"
```

---

### Task 3: Prevent HAN+PDQN From Double-Prepending Raw Observations

**Files:**
- Modify: `scripts/train.py`
- Modify: `tests/test_han_integration.py`

- [ ] **Step 1: Simplify `HANPDQNTrainer._init_environment`**

Find `HANPDQNTrainer._init_environment`. Replace its body with:

```python
def _init_environment(self):
    super()._init_environment()
    self.logger.info(
        f"  - HAN+PDQN observation dim: {self.obs_dim} "
        f"(raw {self.raw_obs_dim} + HAN {self.han_out_dim} + rvt/task 5)"
    )
```

- [ ] **Step 2: Simplify `HANPDQNTrainer._encode_graph_state`**

Find `HANPDQNTrainer._encode_graph_state`. Replace its body with:

```python
def _encode_graph_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return super()._encode_graph_state()
```

This keeps HAN+PDQN behavior aligned with the new base layout and avoids `raw + raw + HAN + light`.

- [ ] **Step 3: Run focused HAN integration tests**

Run:

```powershell
& 'C:\Users\19704\.conda\envs\satellite.env\python.exe' -m pytest tests\test_han_integration.py::test_han_mappo_observation_includes_raw_obs_han_and_light_features tests\test_han_integration.py::test_han_pdqn_observation_includes_raw_obs_han_and_light_features tests\test_han_integration.py::test_han_encoder_and_policy_share_consistent_shapes -q
```

Expected:

```text
3 passed
```

- [ ] **Step 4: Commit the HAN+PDQN alignment**

```powershell
git add scripts/train.py tests/test_han_integration.py
git commit -m "refactor: share raw han observation layout across han trainers"
```

---

### Task 4: Run Regression Tests for Training and Evaluation Paths

**Files:**
- No code files modified in this task.

- [ ] **Step 1: Run HAN integration tests**

Run:

```powershell
& 'C:\Users\19704\.conda\envs\satellite.env\python.exe' -m pytest tests\test_han_integration.py -q
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 2: Run MAPPO entropy and baseline plotting tests**

Run:

```powershell
& 'C:\Users\19704\.conda\envs\satellite.env\python.exe' -m pytest tests\test_mappo_entropy.py tests\test_baseline_plotting.py -q
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 3: Run a CPU smoke training job**

Use a new temporary output directory:

```powershell
& 'C:\Users\19704\.conda\envs\satellite.env\python.exe' scripts\train.py --exp_name han_mappo_raw_han_smoke --algorithm mappo --seed 42 --device cpu --num_users 2 --total_timesteps 64 --max_steps 10 --n_steps 16 --batch_size 16 --eval_interval 32 --eval_episodes 1 --graph_update_interval 1 --save_path results\han_mappo_raw_han_smoke --log_path results\logs --best-model-metric effective_latency_score
```

Expected:

```text
training exits with code 0
results\han_mappo_raw_han_smoke\training_history.json exists
```

- [ ] **Step 4: Inspect smoke history config**

Run:

```powershell
& 'C:\Users\19704\.conda\envs\satellite.env\python.exe' -c "import json; p='results/han_mappo_raw_han_smoke/training_history.json'; d=json.load(open(p,encoding='utf-8')); print(d['summary']['total_steps']); print(d['config']['best_model_metric'])"
```

Expected:

```text
64
effective_latency_score
```

- [ ] **Step 5: Commit verification-only cleanup if needed**

If the smoke run creates only disposable artifacts and the repository normally keeps smoke results out of version control, do not add them to git. If tracked files changed unexpectedly, inspect them with:

```powershell
git status --short
```

Commit only source, test, and docs changes from this plan.

---

### Task 5: Add Experiment Log Entry for the Code Change

**Files:**
- Modify: `docs/EXPERIMENT_LOG.md`

- [ ] **Step 1: Append a code-change entry**

Append this entry to the bottom of `docs/EXPERIMENT_LOG.md`:

```markdown
## 2026-05-29 - HAN+MAPPO Raw-Plus-HAN Observation Path

**Intent:** Prepare a diagnostic rerun that tests whether HAN provides
incremental graph information to MAPPO instead of replacing the raw environment
state with a frozen cached embedding.

**Code changes made:**

- `HANMAPPOTrainer` now builds policy observations as
  `raw_observation + HAN_user_embedding + rvt_warning/task_features`.
- `HANPDQNTrainer` reuses the shared raw-plus-HAN observation layout so it does
  not prepend raw observations twice.
- Regression tests cover the HAN+MAPPO and HAN+PDQN observation dimensions and
  raw-observation prefix.

**Expected metric effect:** This should reduce the risk that HAN+MAPPO
underperforms MAPPO(no-HAN) because direct raw state features were removed. The
first success criterion is a measurable improvement over MAPPO(no-HAN) on
`effective_latency_score` without losing the existing energy advantage. This
does not yet train the HAN encoder end-to-end and should be treated as an
information-path ablation, not a final architecture claim.
```

- [ ] **Step 2: Run a markdown sanity check**

Run:

```powershell
Get-Content -Path 'docs/EXPERIMENT_LOG.md' -Tail 45
```

Expected:

```text
The new "HAN+MAPPO Raw-Plus-HAN Observation Path" entry appears at the end.
```

- [ ] **Step 3: Commit documentation**

```powershell
git add docs/EXPERIMENT_LOG.md
git commit -m "docs: log han mappo raw plus han observation change"
```

---

### Task 6: Launch the Diagnostic Experiment

**Files:**
- No source files modified in this task.
- New artifacts under `results/full_train_latency_priority_g1_300k_600s_u10_rawhan_<run_id>`.
- New artifacts under `results/baseline_compare/g1_300k_600s_u10_rawhan_<run_id>`.

- [ ] **Step 1: Choose a run id**

Use:

```text
20260529_rawhan
```

If that directory already exists, use:

```text
20260529_rawhan_2
```

- [ ] **Step 2: Run the g1 diagnostic suite with a distinct run id**

Run:

```powershell
& 'C:\Users\19704\.conda\envs\satellite.env\python.exe' scripts\run_latency_priority_g1_300k_600s_u10_suite.py --run-id 20260529_rawhan --python-executable 'C:\Users\19704\.conda\envs\satellite.env\python.exe' --device cuda --graph-update-interval 1 --total-timesteps 300000 --max-steps 600 --num-users 10 --compare-episodes 3 --best-model-metric effective_latency_score --compare-ranking-metric effective_latency_score
```

Expected:

```text
results/full_train_latency_priority_g1_300k_600s_u10_20260529_rawhan/training_history.json exists
results/baseline_compare/g1_300k_600s_u10_20260529_rawhan/comparison_summary.json exists
```

- [ ] **Step 3: Summarize the comparison**

Run:

```powershell
$summary = Get-Content -Path 'results/baseline_compare/g1_300k_600s_u10_20260529_rawhan/comparison_summary.json' -Raw | ConvertFrom-Json
$summary.methods | Sort-Object selection_score -Descending | Select-Object display_name,selection_score,avg_delay,task_completion_rate,deadline_violation_rate,total_energy,energy_per_resolved_task,service_continuity_rate | Format-Table -AutoSize
```

Expected:

```text
HAN+MAPPO appears in the ranked table with selection_score, delay, task, deadline, energy, and continuity metrics.
```

- [ ] **Step 4: Compare against the previous g1 run**

Use these reference files:

```text
results/full_train_latency_priority_g1_300k_600s_u10_20260528_235135/training_history.json
results/baseline_compare/g1_300k_600s_u10_20260528_235135/comparison_summary.json
```

Run:

```powershell
$old = Get-Content -Path 'results/baseline_compare/g1_300k_600s_u10_20260528_235135/comparison_summary.json' -Raw | ConvertFrom-Json
$new = Get-Content -Path 'results/baseline_compare/g1_300k_600s_u10_20260529_rawhan/comparison_summary.json' -Raw | ConvertFrom-Json
$oldHan = $old.methods | Where-Object is_system
$newHan = $new.methods | Where-Object is_system
[PSCustomObject]@{
  old_selection_score = $oldHan.selection_score
  new_selection_score = $newHan.selection_score
  old_avg_delay = $oldHan.avg_delay
  new_avg_delay = $newHan.avg_delay
  old_task_completion = $oldHan.task_completion_rate
  new_task_completion = $newHan.task_completion_rate
  old_deadline_violation = $oldHan.deadline_violation_rate
  new_deadline_violation = $newHan.deadline_violation_rate
  old_energy_per_resolved_task = $oldHan.energy_per_resolved_task
  new_energy_per_resolved_task = $newHan.energy_per_resolved_task
} | Format-List
```

Expected:

```text
The output shows whether raw-plus-HAN improved effective latency, task completion, deadline violation, and energy per resolved task.
```

- [ ] **Step 5: Append final experiment result to `docs/EXPERIMENT_LOG.md`**

Run this command after `comparison_summary.json` exists. It computes the
HAN+MAPPO, MAPPO(no-HAN), and Min-Distance rows and appends a concrete entry:

```powershell
$newPath = 'results/baseline_compare/g1_300k_600s_u10_20260529_rawhan/comparison_summary.json'
$oldPath = 'results/baseline_compare/g1_300k_600s_u10_20260528_235135/comparison_summary.json'
$new = Get-Content -Path $newPath -Raw | ConvertFrom-Json
$old = Get-Content -Path $oldPath -Raw | ConvertFrom-Json
$han = $new.methods | Where-Object is_system
$nohan = $new.methods | Where-Object { $_.method -eq 'mappo_no_han' }
$mindist = $new.methods | Where-Object { $_.method -eq 'min_distance' }
$oldHan = $old.methods | Where-Object is_system
$deltaVsOld = (($han.selection_score - $oldHan.selection_score) / [Math]::Abs($oldHan.selection_score)) * 100.0
$deltaVsNoHan = (($han.selection_score - $nohan.selection_score) / [Math]::Abs($nohan.selection_score)) * 100.0
$decision = if ($deltaVsNoHan -ge 1.0) {
  'Raw-plus-HAN cleared the first diagnostic gate against MAPPO(no-HAN). Next run should test trainable HAN or a deadline/task-success reward-pressure ablation.'
} else {
  'Raw-plus-HAN did not clear the 1% diagnostic gate against MAPPO(no-HAN). Next work should prioritize trainable HAN or auxiliary HAN objectives before more graph-update-interval experiments.'
}
$entry = @"

## 2026-05-29 - Raw-Plus-HAN g1 300k/600s/u10 Diagnostic Result

**Experiment directories:**

- System training: ``results/full_train_latency_priority_g1_300k_600s_u10_20260529_rawhan``
- Baseline comparison: ``results/baseline_compare/g1_300k_600s_u10_20260529_rawhan``

**Configuration:** ``graph_update_interval=1``, ``total_timesteps=300000``,
``max_steps=600``, ``num_users=10``, ``best_model_metric=effective_latency_score``,
and HAN+MAPPO observations set to ``raw + HAN + rvt/task``.

**Main result:**

| Method | Effective Latency | Avg Delay | Task Completion | Deadline Violation | Energy / Resolved Task | Service Continuity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HAN+MAPPO | $($han.selection_score.ToString('0.000000')) | $($han.avg_delay.ToString('0.000000')) | $($han.task_completion_rate.ToString('0.000000')) | $($han.deadline_violation_rate.ToString('0.000000')) | $($han.energy_per_resolved_task.ToString('0.000000')) | $($han.service_continuity_rate.ToString('0.000000')) |
| MAPPO (no HAN) | $($nohan.selection_score.ToString('0.000000')) | $($nohan.avg_delay.ToString('0.000000')) | $($nohan.task_completion_rate.ToString('0.000000')) | $($nohan.deadline_violation_rate.ToString('0.000000')) | $($nohan.energy_per_resolved_task.ToString('0.000000')) | $($nohan.service_continuity_rate.ToString('0.000000')) |
| Min-Distance | $($mindist.selection_score.ToString('0.000000')) | $($mindist.avg_delay.ToString('0.000000')) | $($mindist.task_completion_rate.ToString('0.000000')) | $($mindist.deadline_violation_rate.ToString('0.000000')) | $($mindist.energy_per_resolved_task.ToString('0.000000')) | $($mindist.service_continuity_rate.ToString('0.000000')) |

**Diagnosis:** Raw-plus-HAN changed HAN+MAPPO effective-latency score by
$($deltaVsOld.ToString('+0.00;-0.00;0.00'))% relative to the previous g1 run
and by $($deltaVsNoHan.ToString('+0.00;-0.00;0.00'))% relative to MAPPO(no-HAN)
in this comparison.

**Follow-up decision:** $decision
"@
Add-Content -Path 'docs/EXPERIMENT_LOG.md' -Value $entry -Encoding UTF8
```

- [ ] **Step 6: Commit the experiment log result**

```powershell
git add docs/EXPERIMENT_LOG.md
git commit -m "docs: record raw plus han g1 diagnostic result"
```

---

## Self-Review Checklist

- [ ] The plan starts with the required writing-plans header.
- [ ] The plan keeps scope to the raw-plus-HAN observation path.
- [ ] The plan does not change reward weights, trainable HAN behavior, or graph update interval.
- [ ] The plan includes a failing test before implementation.
- [ ] Every code-editing step includes concrete replacement code.
- [ ] Verification commands use `C:\Users\19704\.conda\envs\satellite.env\python.exe`.
- [ ] New experiment artifacts use a distinct run id and do not overwrite `20260528_235135`.
