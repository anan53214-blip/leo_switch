# Off-Policy Integration Round 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Low-cost credibility pass for the baseline comparison stack: seed off-policy RNG, use per-agent rewards in MAPPO rollout storage, normalize action diagnostics in summaries, and reduce MADDPG/PDQN implementation drift without expanding end-to-end HAN training.

**Architecture:** Keep `scripts/compare_system_baselines.py` as the only orchestration, evaluation, summary, dashboard, and plotting entrypoint. Keep HAN off-policy trainers on the current cached/no-grad encoder path. Reuse `src/algorithm/maddpg.py` and `src/algorithm/pdqn.py` as the shared off-policy action/update core; avoid reward-function changes and avoid broad rewrites of the compare script.

**Tech Stack:** Python, PyTorch, NumPy, Gymnasium-style `LEOSatelliteEnv`, `scripts/train.py`, `scripts/compare_system_baselines.py`, `src/algorithm/maddpg.py`, `src/algorithm/pdqn.py`, pytest with `C:\Users\19704\.conda\envs\satellite.env\python.exe`.

---

## Handoff Prompt For The Execution Session

```text
We are in D:\python_code\LEO_switch. Implement docs/superpowers/plans/2026-05-22-offpolicy-integration-round1.md task-by-task.

Use the required skills before coding:
1. superpowers:using-superpowers
2. leo-switch-system-map
3. superpowers:test-driven-development
4. superpowers:verification-before-completion

Constraints:
- Do not implement true end-to-end HAN for off-policy algorithms.
- Do not change the reward function.
- Do not refactor the whole compare script.
- Keep scripts/compare_system_baselines.py as the single experiment orchestration and reporting entrypoint.
- Preserve existing off-policy evaluation isolation behavior.
- Use C:\Users\19704\.conda\envs\satellite.env\python.exe for tests.
```

---

## Files And Responsibilities

- Modify: `src/algorithm/maddpg.py`
  - Add `seed` to `MADDPGConfig`.
  - Seed `MADDPGAlgorithm.rng` from config.
  - Keep `random_actions()`, `act()`, and `update()` as the shared MADDPG core.

- Modify: `src/algorithm/pdqn.py`
  - Add `seed` to `PDQNConfig`.
  - Seed `PDQNAlgorithm.rng` from config.
  - Keep `random_actions()`, `act()`, and `update()` as the shared PDQN core.

- Modify: `scripts/train.py`
  - Pass `TrainConfig.seed` into `MADDPGConfig` and `PDQNConfig` for HAN+MADDPG/HAN+PDQN.
  - Change `HANMAPPOTrainer.collect_rollouts()` so rollout-buffer rewards prefer `env.last_user_rewards` or `info["user_rewards"]`, while reward statistics continue to use the scalar/mean reward scale.

- Modify: `scripts/compare_system_baselines.py`
  - Pass experiment seed into raw PDQN and raw MADDPG algorithm configs.
  - Migrate raw MADDPG training/evaluation to `MADDPGAlgorithm` where practical, matching current raw hyperparameters.
  - Add a small summary-schema normalizer so `handover_action_rate`, `local_compute_rate`, and `mean_offload_ratio` exist in all JSON/CSV method rows, including old or reused artifacts.
  - Keep action collection in all evaluation functions: `evaluate_system_checkpoint()`, `evaluate_mappo_checkpoint_with_trainer()`, `evaluate_han_offpolicy_checkpoint()`, `evaluate_maddpg_policy()`, `evaluate_pdqn_policy()`, and `evaluate_policy()`.

- Modify: `tests/test_offpolicy_evaluation.py`
  - Add seeded MADDPG and PDQN early random-action reproducibility tests.
  - Keep the existing off-policy evaluation isolation test passing.

- Modify: `tests/test_han_integration.py`
  - Add a lightweight MAPPO rollout test proving per-agent rewards are stored in the rollout buffer instead of scalar reward copies.

- Modify: `tests/test_baseline_plotting.py`
  - Add a summary JSON/CSV schema test for action diagnostics.

---

## Task 1: Add Off-Policy RNG Reproducibility Tests

**Files:**
- Modify: `tests/test_offpolicy_evaluation.py`
- Test target before implementation: `src/algorithm/maddpg.py`, `src/algorithm/pdqn.py`

- [ ] **Step 1: Write failing tests for MADDPG and PDQN seeded random actions**

Append this to `tests/test_offpolicy_evaluation.py`:

```python
def test_maddpg_random_actions_are_reproducible_with_same_seed():
    masks = np.array(
        [
            [True, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )
    config = MADDPGConfig(
        num_agents=2,
        obs_dim=3,
        max_candidates=2,
        actor_hidden_dims=(8,),
        critic_hidden_dims=(16,),
        seed=123,
        device="cpu",
    )

    first = MADDPGAlgorithm(config).random_actions(masks)
    second = MADDPGAlgorithm(config).random_actions(masks)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.array_equal(first[2], second[2])


def test_pdqn_random_actions_are_reproducible_with_same_seed():
    masks = np.array(
        [
            [True, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )
    config = PDQNConfig(
        num_agents=2,
        obs_dim=3,
        max_candidates=2,
        q_hidden_dims=(8,),
        param_hidden_dims=(8,),
        seed=456,
        device="cpu",
    )

    first = PDQNAlgorithm(config).random_actions(masks)
    second = PDQNAlgorithm(config).random_actions(masks)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.array_equal(first[2], second[2])
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests\test_offpolicy_evaluation.py::test_maddpg_random_actions_are_reproducible_with_same_seed tests\test_offpolicy_evaluation.py::test_pdqn_random_actions_are_reproducible_with_same_seed -q
```

Expected: both tests fail with `TypeError: __init__() got an unexpected keyword argument 'seed'` or equivalent dataclass constructor failure.

- [ ] **Step 3: Add seed fields and seed the NumPy generators**

In `src/algorithm/maddpg.py`, update `MADDPGConfig` and `MADDPGAlgorithm.__init__()`:

```python
@dataclass
class MADDPGConfig:
    num_agents: int = 20
    obs_dim: int = 69
    max_candidates: int = 10
    sat_embed_dim: int = 64
    actor_hidden_dims: tuple = (256, 128)
    critic_hidden_dims: tuple = (512, 256, 128)
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.01
    noise_start: float = 0.35
    noise_final: float = 0.05
    noise_decay_steps: int = 100_000
    batch_size: int = 128
    replay_size: int = 50_000
    warmup_steps: int = 1_000
    grad_clip_norm: float = 1.0
    seed: int | None = None
    device: str = "cpu"
```

```python
self.rng = np.random.default_rng(config.seed)
```

In `src/algorithm/pdqn.py`, update `PDQNConfig` and `PDQNAlgorithm.__init__()`:

```python
@dataclass
class PDQNConfig:
    num_agents: int = 20
    obs_dim: int = 69
    max_candidates: int = 10
    q_hidden_dims: tuple = (256, 128)
    param_hidden_dims: tuple = (128, 64)
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 128
    replay_size: int = 50_000
    warmup_steps: int = 1_000
    target_update_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 100_000
    grad_clip_norm: float = 1.0
    param_loss_coef: float = 0.1
    bc_loss_coef: float = 0.01
    seed: int | None = None
    device: str = "cpu"
```

```python
self.rng = np.random.default_rng(config.seed)
```

- [ ] **Step 4: Pass seeds from trainers and compare script**

In `scripts/train.py`, add `seed=self.config.seed` to both `MADDPGConfig(...)` and `PDQNConfig(...)` in `HANMADDPGTrainer._init_mappo()` and `HANPDQNTrainer._init_mappo()`.

In `scripts/compare_system_baselines.py`, add `seed=int(seed)` to the raw `PDQNConfig(...)` construction in `train_and_evaluate_pdqn_baseline()`.

- [ ] **Step 5: Run the seed tests and existing off-policy update tests**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests\test_offpolicy_evaluation.py -q
```

Expected: all tests in `tests/test_offpolicy_evaluation.py` pass.

---

## Task 2: Store Per-Agent Rewards In MAPPO Rollout Buffer

**Files:**
- Modify: `tests/test_han_integration.py`
- Modify: `scripts/train.py`

- [ ] **Step 1: Add imports for lightweight stubs**

At the top of `tests/test_han_integration.py`, keep existing imports and add:

```python
from types import SimpleNamespace
import torch
```

The file already uses `shutil.rmtree(...)` in fixture teardown; if `shutil` is not imported, add:

```python
import shutil
```

- [ ] **Step 2: Write a failing MAPPO rollout reward test**

Append this to `tests/test_han_integration.py`:

```python
class _RecordingRolloutBuffer:
    def __init__(self):
        self.rewards = []

    def reset(self):
        self.rewards.clear()

    def add(self, **kwargs):
        self.rewards.append(np.asarray(kwargs["rewards"], dtype=np.float32).copy())

    def compute_returns_and_advantages(self, last_value, last_done):
        self.last_value = float(last_value)
        self.last_done = bool(last_done)


class _RolloutMode:
    def eval(self):
        return None


class _OneStepMAPPO:
    def __init__(self, trainer):
        self.actor = _RolloutMode()
        self.critic = _RolloutMode()
        self.trainer = trainer

    def act(self, observations, available_actions, satellite_embeddings=None):
        return (
            {
                "handover": np.zeros(self.trainer.num_agents, dtype=np.int64),
                "offload": np.zeros(self.trainer.num_agents, dtype=np.float32),
            },
            np.zeros(self.trainer.num_agents, dtype=np.float32),
            0.0,
        )

    def get_value(self, observations, satellite_embeddings=None):
        return 0.0


class _PerAgentRewardEnv:
    def __init__(self):
        self.last_user_rewards = np.array([1.0, -2.0], dtype=np.float32)
        self.stats = {}

    def reset(self):
        return None, {}

    def step(self, actions, return_observation=False, return_info=False):
        info = {"user_rewards": self.last_user_rewards.copy()}
        return None, float(np.mean(self.last_user_rewards)), True, False, info

    def get_stats_summary(self):
        return {
            "total_tasks": 0,
            "completed_tasks": 0,
            "resolved_tasks": 0,
            "total_energy": 0.0,
            "total_user_seconds": 0.0,
            "service_continuity_rate": 0.0,
            "service_availability_rate": 0.0,
            "task_completion_rate": 0.0,
            "task_success_rate": 0.0,
            "task_resolution_rate": 0.0,
            "avg_delay": 0.0,
            "effective_latency_score": 0.0,
        }


def test_collect_rollouts_stores_per_agent_rewards_from_env_metadata():
    trainer_obj = HANMAPPOTrainer.__new__(HANMAPPOTrainer)
    trainer_obj.config = SimpleNamespace(n_steps=1)
    trainer_obj.num_agents = 2
    trainer_obj.total_steps = 0
    trainer_obj.episodes = 0
    trainer_obj.recent_rewards = []
    trainer_obj.env = _PerAgentRewardEnv()
    trainer_obj.buffer = _RecordingRolloutBuffer()
    trainer_obj.mappo = _OneStepMAPPO(trainer_obj)
    trainer_obj.han_encoder = _RolloutMode()
    trainer_obj._encode_graph_state = lambda: (
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    )
    trainer_obj._process_actions = lambda actions: np.column_stack(
        [actions["handover"], actions["offload"]]
    ).astype(np.float32)
    trainer_obj._empty_env_stats = lambda: {}
    trainer_obj._accumulate_env_stats = lambda target, source: target.update(source) or target

    stats = trainer_obj.collect_rollouts()

    assert np.array_equal(
        trainer_obj.buffer.rewards[0],
        np.array([1.0, -2.0], dtype=np.float32),
    )
    assert stats["rollout_mean_reward"] == pytest.approx(-0.5)
```

- [ ] **Step 3: Run the new test and verify it fails**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests\test_han_integration.py::test_collect_rollouts_stores_per_agent_rewards_from_env_metadata -q
```

Expected: failure showing the rollout buffer received `[-0.5, -0.5]` instead of `[1.0, -2.0]`.

- [ ] **Step 4: Capture info from env.step and prefer per-agent rewards for buffer storage**

In `scripts/train.py`, inside `HANMAPPOTrainer.collect_rollouts()`, replace the current env step unpacking:

```python
_, rewards, terminated, truncated, _ = self.env.step(
    env_actions,
    return_observation=False,
    return_info=False
)
```

with:

```python
_, rewards, terminated, truncated, info = self.env.step(
    env_actions,
    return_observation=False,
    return_info=True,
)
info = info or {}
```

Then replace the reward conversion block with:

```python
if hasattr(self.env, "last_user_rewards"):
    agent_rewards = np.asarray(self.env.last_user_rewards, dtype=np.float32)
elif isinstance(info, dict) and "user_rewards" in info:
    agent_rewards = np.asarray(info["user_rewards"], dtype=np.float32)
elif isinstance(rewards, (int, float)):
    shared_reward = float(rewards)
    agent_rewards = np.full(self.num_agents, shared_reward, dtype=np.float32)
elif isinstance(rewards, dict):
    agent_rewards = np.array(
        [rewards.get(f"user_{i}", 0.0) for i in range(self.num_agents)],
        dtype=np.float32,
    )
else:
    agent_rewards = np.asarray(rewards, dtype=np.float32)

if agent_rewards.shape != (self.num_agents,):
    agent_rewards = np.resize(agent_rewards, self.num_agents).astype(np.float32)

reward = scalar_reward_value(rewards)
```

This keeps episode reward statistics on the scalar reward path while storing per-agent values in the rollout buffer.

- [ ] **Step 5: Run the rollout test and environment per-agent reward test**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests\test_han_integration.py::test_collect_rollouts_stores_per_agent_rewards_from_env_metadata tests\test_env_metrics.py::test_step_exposes_per_agent_rewards_matching_scalar_mean -q
```

Expected: both tests pass.

---

## Task 3: Normalize Action Diagnostics In Summary JSON/CSV

**Files:**
- Modify: `tests/test_baseline_plotting.py`
- Modify: `scripts/compare_system_baselines.py`

- [ ] **Step 1: Add imports for summary save helpers**

In `tests/test_baseline_plotting.py`, extend the import from `scripts.compare_system_baselines`:

```python
from scripts.compare_system_baselines import (
    action_diagnostics,
    annotate_priority_metrics,
    load_training_curve_from_path,
    reward_component_step_metrics_for_history,
    save_results_csv,
    save_results_json,
)
```

- [ ] **Step 2: Write a failing legacy-summary schema test**

Append this to `tests/test_baseline_plotting.py`:

```python
def test_comparison_summary_outputs_action_diagnostics_for_legacy_methods(tmp_path):
    methods = annotate_priority_metrics(
        [
            {
                "method": "legacy_method",
                "display_name": "Legacy Method",
                "is_system": False,
                "episodes": 1,
                "mean_reward": 1.0,
                "std_reward": 0.0,
                "avg_delay": 1.0,
                "service_continuity_rate": 1.0,
                "task_completion_rate": 1.0,
                "avg_load_balance_score": 0.0,
                "total_energy": 1.0,
                "resolved_tasks": 1.0,
            }
        ],
        "latency_priority_score",
    )

    json_path = save_results_json(tmp_path, {"methods": methods})
    csv_path = save_results_csv(tmp_path, methods)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    method = payload["methods"][0]
    assert method["handover_action_rate"] == pytest.approx(0.0)
    assert method["local_compute_rate"] == pytest.approx(0.0)
    assert method["mean_offload_ratio"] == pytest.approx(0.0)

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "handover_action_rate" in csv_text.splitlines()[0]
    assert ",0.0,0.0,0.0," in csv_text
```

- [ ] **Step 3: Run the new schema test and verify it fails**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests\test_baseline_plotting.py::test_comparison_summary_outputs_action_diagnostics_for_legacy_methods -q
```

Expected: failure because legacy method rows do not contain the diagnostics keys in JSON.

- [ ] **Step 4: Add a small summary normalizer**

In `scripts/compare_system_baselines.py`, after `action_diagnostics(...)`, add:

```python
def ensure_action_diagnostic_fields(method: Dict) -> Dict:
    normalized = dict(method)
    for key in ACTION_DIAGNOSTIC_KEYS:
        try:
            normalized[key] = float(normalized.get(key, 0.0))
        except (TypeError, ValueError):
            normalized[key] = 0.0
    return normalized
```

- [ ] **Step 5: Use the normalizer in summary annotation and JSON saving**

In `annotate_priority_metrics(...)`, replace:

```python
annotated = [dict(method) for method in methods]
```

with:

```python
annotated = [ensure_action_diagnostic_fields(method) for method in methods]
```

In `save_results_json(...)`, replace:

```python
json.dump(payload, handle, ensure_ascii=False, indent=2)
```

with:

```python
payload_to_save = dict(payload)
if "methods" in payload_to_save:
    payload_to_save["methods"] = [
        ensure_action_diagnostic_fields(method)
        for method in payload_to_save.get("methods", [])
    ]
json.dump(payload_to_save, handle, ensure_ascii=False, indent=2)
```

- [ ] **Step 6: Run plotting/summary tests**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests\test_baseline_plotting.py -q
```

Expected: all `tests/test_baseline_plotting.py` tests pass.

---

## Task 4: Route Raw MADDPG Through The Shared Algorithm Core

**Files:**
- Modify: `scripts/compare_system_baselines.py`
- Test: `tests/test_offpolicy_evaluation.py`

- [ ] **Step 1: Change `evaluate_maddpg_policy()` to accept `MADDPGAlgorithm`**

In `scripts/compare_system_baselines.py`, change the function signature:

```python
def evaluate_maddpg_policy(
    algorithm: MADDPGAlgorithm,
    objective: str,
    config: Dict,
    episodes: int,
    seed: int,
    max_steps: Optional[int],
) -> Dict:
```

Inside the loop, replace the call to `select_maddpg_env_actions(...)` with:

```python
env_actions, _, _ = algorithm.act(observations, masks, deterministic=True)
```

Remove the local `rng` and `actor.eval()` lines from that function.

- [ ] **Step 2: Replace raw MADDPG local networks with `MADDPGAlgorithm`**

In `train_and_evaluate_maddpg_baseline(...)`, replace the actor/critic construction, optimizers, and scalar hyperparameter locals with:

```python
maddpg_config = MADDPGConfig(
    num_agents=num_agents,
    obs_dim=obs_dim,
    max_candidates=handover_dim - 1,
    actor_hidden_dims=(256, 128),
    critic_hidden_dims=(512, 256, 128),
    actor_lr=5e-4,
    critic_lr=1e-3,
    gamma=0.99,
    tau=0.01,
    noise_start=0.35,
    noise_final=0.05,
    noise_decay_steps=max(int(total_timesteps * 0.7), 1),
    batch_size=128,
    replay_size=50_000,
    warmup_steps=min(1_000, max(64, total_timesteps // 20)),
    grad_clip_norm=1.0,
    seed=int(seed),
    device=str(device),
)
algo = MADDPGAlgorithm(maddpg_config)
replay = MultiAgentReplayBuffer(
    capacity=algo.config.replay_size,
    num_agents=num_agents,
    obs_dim=obs_dim,
    action_feature_dim=algo.action_feature_dim,
    mask_dim=algo.handover_dim,
    device=str(device),
)
```

Use these config fields in the training loop:

```python
if step_idx < algo.config.warmup_steps:
    env_actions, action_features, _ = algo.random_actions(masks)
else:
    env_actions, action_features, _ = algo.act(observations, masks, deterministic=False)
```

Store transitions with the shared replay buffer:

```python
replay.add(
    observations.astype(np.float32, copy=True),
    action_features.astype(np.float32, copy=True),
    float(reward_value),
    next_observations.astype(np.float32, copy=True),
    bool(done),
    masks.astype(bool, copy=True),
    next_masks.astype(bool, copy=True),
)
```

Use the shared update:

```python
if len(replay) >= max(algo.config.batch_size, algo.config.warmup_steps):
    stats = algo.update(replay)
    if stats:
        recent_actor_losses.append(float(stats.get("actor_loss", 0.0)))
        recent_critic_losses.append(float(stats.get("critic_loss", 0.0)))
```

Use shared eval calls:

```python
eval_result = evaluate_maddpg_policy(
    algorithm=algo,
    objective=objective,
    config=config,
    episodes=train_eval_episodes,
    seed=seed + 10_000,
    max_steps=max_steps,
)
```

For checkpointing, use `algo.save(checkpoint_path)` for the best model and save the final model by temporarily restoring final states or by adding a local helper that serializes `algo` state dicts. Keep the existing checkpoint filenames: `maddpg_model.pt` and `maddpg_final_model.pt`.

- [ ] **Step 3: Preserve current raw MADDPG hyperparameter semantics in history**

In the `training_history.json` config block, write values from `algo.config`:

```python
"actor_lr": algo.config.actor_lr,
"critic_lr": algo.config.critic_lr,
"gamma": algo.config.gamma,
"tau": algo.config.tau,
"noise_start": algo.config.noise_start,
"noise_final": algo.config.noise_final,
"noise_decay_steps": algo.config.noise_decay_steps,
"warmup_steps": algo.config.warmup_steps,
"batch_size": algo.config.batch_size,
"seed": algo.config.seed,
```

- [ ] **Step 4: Add a shared-action behavior test**

Append this to `tests/test_offpolicy_evaluation.py`:

```python
def test_maddpg_act_respects_action_mask_and_feature_shape():
    algorithm = MADDPGAlgorithm(
        MADDPGConfig(
            num_agents=2,
            obs_dim=3,
            max_candidates=2,
            actor_hidden_dims=(8,),
            critic_hidden_dims=(16,),
            seed=9,
            device="cpu",
        )
    )
    observations = np.zeros((2, 3), dtype=np.float32)
    masks = np.array(
        [
            [True, False, False],
            [False, False, True],
        ],
        dtype=bool,
    )

    env_actions, action_features, handover = algorithm.act(
        observations,
        masks,
        deterministic=True,
    )

    assert env_actions.shape == (2, 2)
    assert action_features.shape == (2, 4)
    assert handover.tolist() == [0, 2]
    assert np.allclose(action_features.sum(axis=1), np.array([1.0, 1.0]) + env_actions[:, 1])
```

- [ ] **Step 5: Run off-policy tests**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests\test_offpolicy_evaluation.py -q
```

Expected: all off-policy tests pass, including the existing "does not reset training env" test.

---

## Task 5: Align Raw PDQN And HAN Off-Policy Config Records

**Files:**
- Modify: `scripts/compare_system_baselines.py`
- Modify: `scripts/train.py`

- [ ] **Step 1: Add seed and explicit shared hyperparameters to raw PDQN config**

In `scripts/compare_system_baselines.py`, in `train_and_evaluate_pdqn_baseline(...)`, ensure the raw `PDQNConfig(...)` contains:

```python
batch_size=128,
warmup_steps=1_000,
replay_size=50_000,
target_update_interval=int(config.get("target_update_interval", 500)),
epsilon_start=float(config.get("epsilon_start", 1.0)),
epsilon_final=float(config.get("epsilon_final", 0.05)),
epsilon_decay_steps=max(
    int(total_timesteps * float(config.get("epsilon_decay_fraction", 0.4))),
    1_001,
    1,
),
seed=int(seed),
device=device,
```

- [ ] **Step 2: Record PDQN config values from `algo.config`**

In the raw PDQN `training_history.json` config block, include:

```python
"batch_size": algo.config.batch_size,
"warmup_steps": algo.config.warmup_steps,
"replay_size": algo.config.replay_size,
"target_update_interval": algo.config.target_update_interval,
"epsilon_start": algo.config.epsilon_start,
"epsilon_final": algo.config.epsilon_final,
"epsilon_decay_steps": algo.config.epsilon_decay_steps,
"seed": algo.config.seed,
```

- [ ] **Step 3: Record HAN off-policy seeds through algorithm config**

In `scripts/train.py`, ensure the HAN+MADDPG and HAN+PDQN config construction from Task 1 includes `seed=self.config.seed`. The checkpoint `config` saved by `MADDPGAlgorithm.save()` and `PDQNAlgorithm.save()` then includes the seed through `asdict(self.config)`.

- [ ] **Step 4: Run focused off-policy and plotting tests**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests\test_offpolicy_evaluation.py tests\test_baseline_plotting.py -q
```

Expected: both test files pass.

---

## Task 6: Final Verification

**Files:**
- No new source edits unless verification finds a concrete failure.

- [ ] **Step 1: Run the required test command**

Run exactly:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests\test_offpolicy_evaluation.py tests\test_baseline_plotting.py tests\test_env_metrics.py::test_step_exposes_per_agent_rewards_matching_scalar_mean tests\test_han_integration.py tests\test_mappo_entropy.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: If a test fails, use systematic debugging**

Use `superpowers:systematic-debugging` before changing code in response to the failure. Capture:

```text
Failing test:
Observed failure:
One specific cause confirmed from source:
Smallest fix:
```

- [ ] **Step 3: Check git diff for forbidden scope creep**

Run:

```powershell
git diff -- src/algorithm/maddpg.py src/algorithm/pdqn.py scripts/train.py scripts/compare_system_baselines.py tests/test_offpolicy_evaluation.py tests/test_han_integration.py tests/test_baseline_plotting.py
```

Expected: no reward-function edits, no end-to-end HAN changes, no broad file restructure, no unrelated generated artifacts.

---

## Self-Review

Spec coverage:
- Off-policy implementation drift: covered by Tasks 4 and 5, with raw MADDPG routed through `MADDPGAlgorithm` and raw/HAN configs aligned.
- MADDPG/PDQN seed: covered by Task 1 and propagated in Task 5.
- MAPPO per-agent rollout rewards: covered by Task 2.
- Action diagnostics in summaries: covered by Task 3, while keeping existing evaluation action collection intact.
- No end-to-end HAN and no reward-function changes: stated in constraints and checked in Task 6.

Placeholder scan:
- The plan contains no placeholder markers, no open-ended "handle later" items, and every code-edit step includes concrete code or exact replacement guidance.

Type consistency:
- `seed` is `int | None` in both config dataclasses and passed as `int(seed)` from compare runs.
- `agent_rewards` is always shaped as `(self.num_agents,)` before `self.buffer.add(...)`.
- Diagnostics keys use the existing `ACTION_DIAGNOSTIC_KEYS` list.
