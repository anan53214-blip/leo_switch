# Configuration Synchronization Design

**Date:** 2026-07-27

## Goal

Synchronize the repository's behavioral defaults with the completed training
suite at
`results/baseline_compare/multiuser_scaling_multiuser_6_7`, while preserving
portable and run-specific settings such as device selection, experiment names,
output paths, and user counts.

## Evidence and Root Cause

The five source training histories referenced by the suite, for 20, 25, 30,
35, and 40 users, use the same behavioral configuration:

- `max_steps=600`
- `total_timesteps=300000`
- `n_steps=1024`
- `eval_interval=50000`
- `eval_episodes=3`
- `save_interval=100000`
- `graph_update_interval=1`
- `log_interval=1`
- `best_model_metric=avg_delay`
- delay-priority reward weights:
  `0.35/0.05/0.10/0.05/0.40/0.15/0.25`
- `reward_enqueue_bonus=0.0`
- `reward_failed_handover_penalty=0.3`
- `reward_deadline_penalty=1.0`
- `reward_failed_task_penalty=0.8`

The repository currently duplicates these defaults in `EnvConfig`,
`TrainConfig`, the training CLI, the baseline comparison script, suite
runners, tests, and documentation. A 2026-07-06 multi-objective reward change
updated only part of those locations, leaving incompatible defaults.

## Design

Use the existing configuration classes as the sources of truth instead of
adding another configuration layer:

1. `EnvConfig` owns environment-duration and reward defaults.
2. `TrainConfig` reuses the relevant `EnvConfig` defaults and owns the
   training, evaluation, checkpoint, graph-refresh, and model-selection
   defaults.
3. Training CLI argument defaults are derived from one `TrainConfig` instance.
4. Baseline comparison defaults are derived from `TrainConfig`; the comparison
   builder must not overwrite reward values with separate literals.
5. Suite runners retain their explicit reproducibility settings and are
   covered by consistency tests against the canonical defaults.

## Scope

Update:

- environment and reward defaults;
- training duration and cadence defaults;
- CLI defaults and help text;
- baseline comparison defaults;
- current configuration documentation;
- regression tests and a new experiment-log entry.

Do not update:

- `device=auto`, because hardware availability is runtime-specific;
- experiment names and output paths, because they identify runs rather than
  model behavior;
- default user count, because the reference suite intentionally varies it;
- historical experiment-log records;
- algorithm-specific baseline parameters that are intentionally different.

## Verification

Add tests that compare the canonical expected configuration with:

- `EnvConfig`;
- `TrainConfig`;
- training CLI defaults;
- baseline comparison configuration;
- both latency-priority suite configuration classes.

Run those tests first in the failing state, implement the synchronization, then
run the focused configuration/reward tests followed by the complete test suite.

