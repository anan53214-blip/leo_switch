# LEO_switch Experiment Log

This file records experiment results, diagnosis, and code changes that affect
training or evaluation behavior. Add a new entry for every experiment run,
including failed or inconclusive runs.

## Recording Checklist

- Experiment directory and command
- Code version or short description of local changes
- Objective, selection metric, seed, timesteps, episodes, max steps, user count
- Main comparison table: reward, effective latency score, delay, continuity,
  task success, deadline violation, energy
- Reward component diagnosis: delay, energy, QoS, continuity, handover,
  enqueue, deadline, queue-full, failed-handover, handover-cost
- Interpretation: what improved, what regressed, and likely root cause
- Follow-up decision: keep, revert, tune, or rerun

## 2026-05-10 01:07:57 - Baseline Compare, Latency Priority

**Experiment directory:** `results/baseline_compare/20260510_010757`

**Command summary:** `scripts/compare_system_baselines.py --run-mode train_compare --objective multi_objective --total-timesteps 1200000 --episodes 10 --max-steps 2000 --seed 42 --num-users 20 --best-model-metric effective_latency_score --compare-ranking-metric effective_latency_score --baselines all`

**Code state:** pre-fix reward and training defaults. Reward did not include an
explicit service-continuity term. MAPPO entropy coefficient used a hard-coded
linear decay to 10% by about 600k steps. Training CLI defaulted to
`n_epochs=4`, `batch_size=256`.

**Main result:**

| Method | Mean Reward | Effective Latency | Avg Delay | Continuity | Task Success | Deadline Violation | Energy / Resolved Task |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MADDPG | 952.221 | 0.297095 | 2.113535 | 0.988585 | 0.935667 | 0.064211 | 2.073609 |
| Min-Distance | 951.742 | 0.296882 | 2.113866 | 0.987993 | 0.935659 | 0.064219 | 2.073682 |
| DQN | 907.117 | 0.279776 | 2.186189 | 0.974788 | 0.914433 | 0.085330 | 1.702983 |
| HAN+MAPPO | 795.113 | 0.273059 | 2.218435 | 0.987705 | 0.889758 | 0.110106 | 1.194593 |
| MAPPO (no HAN) | 792.147 | 0.272036 | 2.233144 | 0.988277 | 0.889959 | 0.109919 | 1.245349 |

**Diagnosis:**

HAN+MAPPO was energy-efficient but lost on task success and deadline
violations. The earlier suspicion that service continuity was around 0.488 was
incorrect; that value was load-balance score. Evaluation continuity was already
high at 0.987705. The reward/metric mismatch still matters because
`effective_latency_score = 1 / (1 + delay) * continuity * task_success`, while
the reward optimized additive delay, energy, QoS, handover, and load-balance
terms.

Training curves showed exploration collapse after about 600k steps and weak
handover success improvement. HAN provided almost no measured gain over MAPPO
without HAN in this run.

**Code changes started from this diagnosis on 2026-05-12:**

- Added step-level `reward_service_continuity` to align reward with the
  continuity factor used by `effective_latency_score`.
- Lowered default `reward_failed_handover_penalty` from 0.6 to 0.3.
- Changed training defaults to `n_epochs=10`, `batch_size=64`,
  `entropy_coef=0.005`, and `entropy_schedule=constant`.
- Added configurable MAPPO entropy schedules: `constant` and `linear`.
- Added reward breakdown fields to baseline comparison summaries and
  per-episode CSV output.

**Follow-up experiment:**

Rerun the same suite with a new output directory and compare reward components
for every method, not only HAN+MAPPO training history. Primary success criteria:
HAN+MAPPO should improve `task_success_rate`, lower `deadline_violation_rate`,
and raise `effective_latency_score` without destroying energy efficiency.

## 2026-05-16 - HAN+MADDPG / HAN+PDQN Implementation Smoke

**Code version:** `b43ecfe` plus local implementation changes for
`src/algorithm/replay_buffer.py`, `src/algorithm/maddpg.py`,
`src/algorithm/pdqn.py`, `scripts/train.py`, and
`scripts/compare_system_baselines.py`.

**Current setting:** post-2026-05-12 environment/reward code is canonical.
Old 2026-05-10 learned-method results are historical reference only and should
not be mixed into strict current-code comparison tables.

**Commands and artifacts:**

- `python scripts/train.py --algorithm maddpg --total_timesteps 2000 --max_steps 100 --eval_episodes 1 --save_path results/smoke_han_maddpg --log_path results/smoke_logs --device cpu`
  - wrote `results/smoke_han_maddpg/training_history.json`
  - wrote `results/smoke_han_maddpg/final_model.pt`
- `python scripts/train.py --algorithm pdqn --total_timesteps 2000 --max_steps 100 --eval_episodes 1 --save_path results/smoke_han_pdqn --log_path results/smoke_logs --device cpu`
  - wrote `results/smoke_han_pdqn/training_history.json`
  - wrote `results/smoke_han_pdqn/final_model.pt`
- `python scripts/compare_system_baselines.py --baselines random maddpg pdqn han_maddpg han_pdqn --total-timesteps 2000 --maddpg-timesteps 2000 --pdqn-timesteps 2000 --episodes 1 --max-steps 100 --device cpu`
  - wrote `results/baseline_compare/20260516_153817/comparison_summary.json`
  - wrote `results/baseline_compare/20260516_153817/comparison_summary.csv`

**Smoke comparison snapshot:** 2000 timesteps, 1 evaluation episode, max 100
steps, CPU, seed 42. This is only a wiring smoke, not a paper-quality ranking.

| Method | Mean Reward | Effective Latency | Avg Delay | Continuity | Task Success | Deadline Violation | Energy / Resolved Task |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MADDPG | 98.752 | 0.319489 | 1.955349 | 0.999400 | 0.944767 | 0.055233 | 1.955349 |
| HAN+PDQN | 89.720 | 0.283957 | 2.161456 | 0.999400 | 0.898256 | 0.101744 | 0.783904 |
| PDQN | 88.287 | 0.275211 | 2.235813 | 0.991400 | 0.898256 | 0.101744 | 1.130091 |
| HAN+MADDPG | 89.545 | 0.274444 | 2.260448 | 0.999400 | 0.895349 | 0.104651 | 0.058514 |
| Random | 58.879 | 0.095314 | 2.417965 | 0.396000 | 0.822674 | 0.171512 | 1.912117 |
| HAN+MAPPO smoke system | 37.526 | 0.035208 | 2.632047 | 0.166000 | 0.770349 | 0.223837 | 1.004960 |

**Verification:** `pytest tests/ -v` passed with 77 passed and 4 skipped.

**Follow-up decision:** keep the implementation and rerun strict comparisons
under current code for `han_mappo`, `mappo_no_han`, `maddpg`, `han_maddpg`,
`pdqn`, and `han_pdqn`. Heuristic methods can be re-evaluated cheaply.
