# HAN+MADDPG and HAN+PDQN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add HAN+MADDPG and HAN+PDQN as first-class algorithms, add raw MADDPG/PDQN comparison paths, remove DQN and DQN+HAN from the main baseline suite, and keep HAN+MAPPO as the incumbent system.

**Architecture:** Reuse the existing environment, graph builder, HAN encoder, metrics, and plotting stack. For GPU practicality on a 3090, HAN is used exactly like the current MAPPO path: a no-grad cached feature encoder that produces 69-dimensional user observations, not an end-to-end off-policy training component. MADDPG and PDQN share an off-policy replay buffer and expose algorithm-level `act()`, `update()`, `save()`, and `load()` APIs.

**Tech Stack:** Python, PyTorch, Gymnasium environment in `src/environment/gym_env.py`, current HAN encoder in `src/model/hetero_gnn.py`, current training entrypoint `scripts/train.py`, current comparison entrypoint `scripts/compare_system_baselines.py`.

---

## Handoff Prompt For The Next Conversation

Use this exact prompt if starting a new Codex conversation:

```text
We are in D:\python_code\LEO_switch. Implement the plan in docs/superpowers/plans/2026-05-16-han-maddpg-pdqn-baselines.md.

Important constraints:
- Do not change src/environment/ or reward/metric definitions.
- Keep HAN+MAPPO. Do not delete it.
- Remove DQN and DQN+HAN from the default/main baseline suite.
- Add random back into DEFAULT_BASELINES.
- Implement HAN+MADDPG and HAN+PDQN, but do not implement true end-to-end HAN backprop for off-policy algorithms. Use the existing no-grad cached HAN feature path.
- Implement raw PDQN as the no-HAN counterpart for HAN+PDQN.
- Keep raw MADDPG as the no-HAN counterpart for HAN+MADDPG.
- Treat current post-5/12 environment as the canonical experiment setting; do not mix old 20260510 learned results into strict comparison tables.
- Use C:\Users\19704\.conda\envs\satellite.env\python.exe for tests and smoke runs.

Please execute task-by-task, verify with tests, and keep edits scoped.
```

---

## Decisions Already Made

- Keep `HAN+MAPPO`; it is the incumbent system and must remain available.
- Implement `HAN+MADDPG` and `HAN+PDQN`.
- Do not implement true end-to-end HAN training for MADDPG/PDQN because server GPU is a 3090 and replaying graph snapshots would increase memory/runtime substantially.
- Delete DQN and DQN+HAN from the main comparison plan.
- Include `random` in the default baseline list.
- Current environment changed after the 20260510 runs: service-continuity reward was added and failed-handover penalty default changed. Strict comparisons under current code should rerun learned methods.

---

## Final Baseline Suite

Set the default baseline list in `scripts/compare_system_baselines.py` to:

```python
DEFAULT_BASELINES = [
    "random",
    "min_distance",
    "full_local",
    "joint_greedy",
    "maddpg",
    "pdqn",
    "mappo_no_han",
    "han_maddpg",
    "han_pdqn",
]
```

`HAN+MAPPO` is not a baseline entry. It remains the system method loaded or trained by the existing system checkpoint path.

Recommended strict comparison methods after implementation:

- `HAN+MAPPO`
- `MAPPO no HAN`
- `MADDPG`
- `HAN+MADDPG`
- `PDQN`
- `HAN+PDQN`
- `Joint Greedy`
- `Min Distance`
- `Full Local`
- `Random`

---

## Algorithm Notes: MADDPG In This Hybrid Action Environment

Standard MADDPG is continuous-action. This codebase has a hybrid action:

```text
handover_action: discrete integer, 0..10
offload_ratio: continuous float, [0, 1]
```

The MADDPG implementation must therefore use a hybrid actor:

```text
obs_i -> shared MLP
      -> handover_logits: 11 values
      -> offload_ratio: sigmoid scalar in [0, 1]
```

Environment execution:

```text
masked_logits = logits with invalid handover candidates set to -inf
handover = argmax(masked_logits)
offload = clip(sigmoid_output + exploration_noise, 0, 1)
env_action = [handover, offload]
```

Critic action representation:

```text
handover one-hot: 11 dims
offload ratio:     1 dim
action_feature:   12 dims
```

Centralized critic input:

```text
all_obs:            (batch, num_agents, obs_dim)
all_action_feature: (batch, num_agents, 12)
optional sat_embed: (batch, num_satellites, 64)
Q(s_all, a_all):    (batch,)
```

Actor update uses a straight-through estimator for the discrete branch:

```python
probs = torch.softmax(masked_logits, dim=-1)
hard_indices = torch.argmax(probs, dim=-1)
hard = F.one_hot(hard_indices, num_classes=handover_dim).to(probs.dtype)
handover_feature = hard + probs - probs.detach()
```

Forward pass behaves like a one-hot discrete action. Backward pass sends gradients through the softmax probabilities.

MADDPG update:

```text
target_action = target_actor(next_obs)
target_q = target_critic(next_obs_all, target_action_all)
y = reward + gamma * (1 - done) * target_q

critic_loss = MSE(critic(obs_all, replay_action_all), y)
actor_loss = -critic(obs_all, actor_action_all).mean()
soft update target_actor and target_critic
```

MADDPG config:

```python
actor_lr = 5e-4
critic_lr = 1e-3
gamma = 0.99
tau = 0.01
batch_size = 128
replay_size = 50_000
warmup_steps = 1_000
noise_start = 0.35
noise_final = 0.05
noise_decay_steps = 100_000
grad_clip_norm = 1.0
actor_hidden_dims = (256, 128)
critic_hidden_dims = (512, 256, 128)
```

HAN+MADDPG input:

```python
obs_dim = 69
handover_dim = 11
action_feature_dim = 12
```

Raw MADDPG input:

```python
obs_dim = env.user_obs_dim
handover_dim = env.max_visible_sats + 1
action_feature_dim = handover_dim + 1
```

---

## Algorithm Notes: PDQN In This Hybrid Action Environment

PDQN is a parameterized-action DQN. It fits this environment naturally:

```text
discrete action: handover candidate
continuous parameter: offload ratio for that handover action
```

Architecture:

```text
ParameterNet_a(obs) -> offload ratio for discrete action a
QNetwork(obs, one_hot(a), offload_ratio) -> scalar Q value
```

Action selection:

```text
for each valid handover action a:
    lambda_a = ParameterNet_a(obs)
    q_a = QNetwork(obs, one_hot(a), lambda_a)
choose a = argmax(q_a)
execute [a, lambda_a]
```

Exploration:

```text
epsilon-greedy over valid handover actions
if random action: sample valid handover and random offload ratio
if greedy action: use argmax Q and corresponding ParameterNet output
```

PDQN target:

```text
for each valid next action a':
    lambda_next_a = target_param_net_a(next_obs)
    q_next_a = target_q_net(next_obs, one_hot(a'), lambda_next_a)
target = reward + gamma * (1 - done) * max_a'(q_next_a)
```

Losses:

```text
q_loss = MSE(Q(obs, replay_handover, replay_offload), target)
param_loss = -Q(obs, best_or_all_valid_action, ParameterNet(obs)).mean()
total_param_loss can include a small behavior-cloning regularizer, but Q-gradient should be primary
```

Do not use `MSE(ParameterNet(obs), replay_offload)` as the main parameter loss because that only imitates old behavior and does not directly optimize value.

PDQN config:

```python
lr = 1e-3
gamma = 0.99
batch_size = 128
replay_size = 50_000
warmup_steps = 1_000
target_update_interval = 500
epsilon_start = 1.0
epsilon_final = 0.05
epsilon_decay_steps = 100_000
grad_clip_norm = 1.0
param_loss_coef = 0.1
bc_loss_coef = 0.01
q_hidden_dims = (256, 128)
param_hidden_dims = (128, 64)
```

HAN+PDQN input:

```python
obs_dim = 69
num_discrete_actions = 11
continuous_param_dim = 1
```

Raw PDQN input:

```python
obs_dim = env.user_obs_dim
num_discrete_actions = env.max_visible_sats + 1
continuous_param_dim = 1
```

---

## File Structure

Create:

- `src/algorithm/replay_buffer.py`
  - Shared off-policy replay buffer for MADDPG and PDQN.

- `src/algorithm/maddpg.py`
  - `MADDPGConfig`
  - `MADDPGActor`
  - `HANCentralizedCritic`
  - `MADDPGAlgorithm`
  - helper functions for masks, one-hot action features, and soft updates

- `src/algorithm/pdqn.py`
  - `PDQNConfig`
  - `PDQNNetwork`
  - `PDQNParameterNet`
  - `PDQNParameterNets`
  - `PDQNAlgorithm`

- `tests/test_replay_buffer.py`
- `tests/test_maddpg_algorithm.py`
- `tests/test_pdqn_algorithm.py`

Modify:

- `src/algorithm/__init__.py`
  - Export new algorithms and replay buffer.

- `scripts/train.py`
  - Add `--algorithm {mappo,maddpg,pdqn}`.
  - Add `TrainConfig.algorithm`.
  - Add `HANMADDPGTrainer`.
  - Add `HANPDQNTrainer`.
  - Preserve default behavior as MAPPO.
  - Keep `HANMAPPOTrainer` checkpoint compatibility.

- `scripts/compare_system_baselines.py`
  - Update `DEFAULT_BASELINES`.
  - Remove DQN from default suite and dispatch.
  - Add raw `pdqn`.
  - Add `han_maddpg`.
  - Add `han_pdqn`.
  - Keep random, min_distance, full_local, joint_greedy, maddpg, mappo_no_han.

Do not modify:

- `src/environment/`
- `src/graph/`
- `src/model/hetero_gnn.py`
- reward definitions
- metric definitions

---

### Task 1: Shared Off-Policy Replay Buffer

**Files:**
- Create: `src/algorithm/replay_buffer.py`
- Test: `tests/test_replay_buffer.py`

- [ ] **Step 1: Write replay buffer tests**

Create `tests/test_replay_buffer.py` with tests for:

```python
import numpy as np

from src.algorithm.replay_buffer import MultiAgentReplayBuffer


def test_multi_agent_replay_buffer_sample_shapes():
    buffer = MultiAgentReplayBuffer(
        capacity=8,
        num_agents=3,
        obs_dim=5,
        action_feature_dim=4,
        mask_dim=4,
        device="cpu",
    )

    for idx in range(6):
        obs = np.full((3, 5), idx, dtype=np.float32)
        action_features = np.zeros((3, 4), dtype=np.float32)
        action_features[:, 0] = 1.0
        next_obs = obs + 1.0
        masks = np.ones((3, 4), dtype=bool)
        buffer.add(obs, action_features, float(idx), next_obs, False, masks, masks)

    batch = buffer.sample(4)

    assert batch["obs"].shape == (4, 3, 5)
    assert batch["actions"].shape == (4, 3, 4)
    assert batch["rewards"].shape == (4,)
    assert batch["next_obs"].shape == (4, 3, 5)
    assert batch["dones"].shape == (4,)
    assert batch["masks"].shape == (4, 3, 4)
    assert batch["next_masks"].shape == (4, 3, 4)


def test_multi_agent_replay_buffer_len_caps_capacity():
    buffer = MultiAgentReplayBuffer(
        capacity=3,
        num_agents=2,
        obs_dim=4,
        action_feature_dim=3,
        mask_dim=3,
        device="cpu",
    )
    obs = np.zeros((2, 4), dtype=np.float32)
    actions = np.zeros((2, 3), dtype=np.float32)
    masks = np.ones((2, 3), dtype=bool)

    for idx in range(5):
        buffer.add(obs, actions, float(idx), obs, False, masks, masks)

    assert len(buffer) == 3
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests/test_replay_buffer.py -v
```

Expected: import failure because `src.algorithm.replay_buffer` does not exist.

- [ ] **Step 3: Implement `MultiAgentReplayBuffer`**

Implement:

```python
from collections import deque
import random
from typing import Dict

import numpy as np
import torch


class MultiAgentReplayBuffer:
    def __init__(self, capacity, num_agents, obs_dim, action_feature_dim, mask_dim, device="cpu"):
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.action_feature_dim = int(action_feature_dim)
        self.mask_dim = int(mask_dim)
        self.device = torch.device(device)
        self.buffer = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, obs, action_features, reward, next_obs, done, masks, next_masks):
        self.buffer.append(
            (
                np.asarray(obs, dtype=np.float32).copy(),
                np.asarray(action_features, dtype=np.float32).copy(),
                float(reward),
                np.asarray(next_obs, dtype=np.float32).copy(),
                bool(done),
                np.asarray(masks, dtype=bool).copy(),
                np.asarray(next_masks, dtype=bool).copy(),
            )
        )

    def sample(self, batch_size) -> Dict[str, torch.Tensor]:
        if len(self.buffer) < int(batch_size):
            raise ValueError(f"Cannot sample {batch_size} transitions from buffer of size {len(self.buffer)}")
        batch = random.sample(self.buffer, int(batch_size))
        return {
            "obs": torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32, device=self.device),
            "actions": torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32, device=self.device),
            "rewards": torch.tensor([item[2] for item in batch], dtype=torch.float32, device=self.device),
            "next_obs": torch.tensor(np.stack([item[3] for item in batch]), dtype=torch.float32, device=self.device),
            "dones": torch.tensor([item[4] for item in batch], dtype=torch.float32, device=self.device),
            "masks": torch.tensor(np.stack([item[5] for item in batch]), dtype=torch.bool, device=self.device),
            "next_masks": torch.tensor(np.stack([item[6] for item in batch]), dtype=torch.bool, device=self.device),
        }
```

- [ ] **Step 4: Run replay buffer tests**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests/test_replay_buffer.py -v
```

Expected: pass.

---

### Task 2: MADDPG Algorithm Module

**Files:**
- Create: `src/algorithm/maddpg.py`
- Modify: `src/algorithm/__init__.py`
- Test: `tests/test_maddpg_algorithm.py`

- [ ] **Step 1: Write tests for MADDPG shapes and update**

Create `tests/test_maddpg_algorithm.py` with:

```python
import numpy as np

from src.algorithm.maddpg import MADDPGAlgorithm, MADDPGConfig
from src.algorithm.replay_buffer import MultiAgentReplayBuffer


def test_maddpg_act_outputs_env_actions_and_features():
    config = MADDPGConfig(num_agents=3, obs_dim=5, max_candidates=3, batch_size=2, warmup_steps=0, device="cpu")
    algo = MADDPGAlgorithm(config)
    obs = np.random.randn(3, 5).astype(np.float32)
    masks = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=bool)

    env_actions, action_features, handover = algo.act(obs, masks, deterministic=True)

    assert env_actions.shape == (3, 2)
    assert action_features.shape == (3, 5)
    assert handover.shape == (3,)
    assert np.all(env_actions[:, 1] >= 0.0)
    assert np.all(env_actions[:, 1] <= 1.0)
    for agent_id, action in enumerate(handover):
        assert masks[agent_id, int(action)]


def test_maddpg_update_returns_losses():
    config = MADDPGConfig(num_agents=3, obs_dim=5, max_candidates=3, batch_size=2, warmup_steps=0, device="cpu")
    algo = MADDPGAlgorithm(config)
    buffer = MultiAgentReplayBuffer(16, 3, 5, 5, 4, device="cpu")
    masks = np.ones((3, 4), dtype=bool)

    for _ in range(8):
        obs = np.random.randn(3, 5).astype(np.float32)
        next_obs = np.random.randn(3, 5).astype(np.float32)
        _, action_features, _ = algo.act(obs, masks, deterministic=False)
        buffer.add(obs, action_features, 1.0, next_obs, False, masks, masks)

    stats = algo.update(buffer)

    assert "actor_loss" in stats
    assert "critic_loss" in stats
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests/test_maddpg_algorithm.py -v
```

Expected: import failure because `src.algorithm.maddpg` does not exist.

- [ ] **Step 3: Implement `src/algorithm/maddpg.py`**

Use the existing baseline code in `scripts/compare_system_baselines.py` as the starting point:

- `MADDPGActor` from current script.
- `MADDPGCritic` concept, renamed and improved as `HANCentralizedCritic`.
- `maddpg_actor_action_features`.
- `maddpg_one_hot_action_features`.
- `soft_update`.

Required public API:

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
    device: str = "cpu"
```

`MADDPGAlgorithm.act()` must return:

```python
env_actions: np.ndarray      # (num_agents, 2)
action_features: np.ndarray  # (num_agents, max_candidates + 2)
handover: np.ndarray         # (num_agents,)
```

`MADDPGAlgorithm.update(replay_buffer)` must return:

```python
{
    "actor_loss": float,
    "critic_loss": float,
}
```

`MADDPGAlgorithm.save(path)` and `load(path)` must save/load actor, target actor, critic, target critic, optimizers, train step, and config.

- [ ] **Step 4: Export MADDPG**

Modify `src/algorithm/__init__.py`:

```python
from .replay_buffer import MultiAgentReplayBuffer
from .maddpg import MADDPGConfig, MADDPGActor, HANCentralizedCritic, MADDPGAlgorithm
```

Add corresponding names to `__all__`.

- [ ] **Step 5: Run MADDPG tests**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests/test_maddpg_algorithm.py tests/test_replay_buffer.py -v
```

Expected: pass.

---

### Task 3: PDQN Algorithm Module

**Files:**
- Create: `src/algorithm/pdqn.py`
- Modify: `src/algorithm/__init__.py`
- Test: `tests/test_pdqn_algorithm.py`

- [ ] **Step 1: Write tests for PDQN shapes and update**

Create `tests/test_pdqn_algorithm.py` with:

```python
import numpy as np

from src.algorithm.pdqn import PDQNAlgorithm, PDQNConfig
from src.algorithm.replay_buffer import MultiAgentReplayBuffer


def test_pdqn_act_outputs_env_actions_and_features():
    config = PDQNConfig(num_agents=3, obs_dim=5, max_candidates=3, batch_size=2, warmup_steps=0, device="cpu")
    algo = PDQNAlgorithm(config)
    obs = np.random.randn(3, 5).astype(np.float32)
    masks = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=bool)

    env_actions, action_features, handover = algo.act(obs, masks, epsilon=0.0)

    assert env_actions.shape == (3, 2)
    assert action_features.shape == (3, 5)
    assert handover.shape == (3,)
    assert np.all(env_actions[:, 1] >= 0.0)
    assert np.all(env_actions[:, 1] <= 1.0)
    for agent_id, action in enumerate(handover):
        assert masks[agent_id, int(action)]


def test_pdqn_update_returns_losses():
    config = PDQNConfig(num_agents=3, obs_dim=5, max_candidates=3, batch_size=2, warmup_steps=0, device="cpu")
    algo = PDQNAlgorithm(config)
    buffer = MultiAgentReplayBuffer(16, 3, 5, 5, 4, device="cpu")
    masks = np.ones((3, 4), dtype=bool)

    for _ in range(8):
        obs = np.random.randn(3, 5).astype(np.float32)
        next_obs = np.random.randn(3, 5).astype(np.float32)
        _, action_features, _ = algo.act(obs, masks, epsilon=1.0)
        buffer.add(obs, action_features, 1.0, next_obs, False, masks, masks)

    stats = algo.update(buffer)

    assert "q_loss" in stats
    assert "param_loss" in stats
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests/test_pdqn_algorithm.py -v
```

Expected: import failure because `src.algorithm.pdqn` does not exist.

- [ ] **Step 3: Implement `src/algorithm/pdqn.py`**

Required config:

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
    device: str = "cpu"
```

Networks:

```python
class PDQNNetwork(nn.Module):
    # input: obs + one_hot(discrete_action) + continuous_param

class PDQNParameterNet(nn.Module):
    # input: obs, output: sigmoid scalar

class PDQNParameterNets(nn.Module):
    # ModuleList of one PDQNParameterNet per discrete action
```

`PDQNAlgorithm.act(obs, masks, epsilon)` must return:

```python
env_actions: np.ndarray
action_features: np.ndarray
handover: np.ndarray
```

`action_features` must use the same representation as MADDPG: one-hot handover plus offload ratio.

`PDQNAlgorithm.update(replay_buffer)` must:

- Flatten sampled `(batch, agents, dim)` transitions to `(batch * agents, dim)` for Q/parameter updates.
- Decode replay action features into handover indices and offload ratios.
- Compute masked target max-Q over valid next actions.
- Update Q-network by Bellman loss.
- Freeze Q-network parameters while updating parameter nets with negative Q objective.
- Hard-update target networks every `target_update_interval`.

Return:

```python
{
    "q_loss": float,
    "param_loss": float,
    "epsilon": float,
}
```

- [ ] **Step 4: Export PDQN**

Modify `src/algorithm/__init__.py`:

```python
from .pdqn import PDQNConfig, PDQNNetwork, PDQNParameterNet, PDQNParameterNets, PDQNAlgorithm
```

Add corresponding names to `__all__`.

- [ ] **Step 5: Run PDQN tests**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests/test_pdqn_algorithm.py tests/test_replay_buffer.py -v
```

Expected: pass.

---

### Task 4: HAN+MADDPG and HAN+PDQN Trainers

**Files:**
- Modify: `scripts/train.py`
- Test: existing tests plus smoke commands

- [ ] **Step 1: Add config fields**

Add to `TrainConfig`:

```python
algorithm: str = "mappo"
maddpg_actor_lr: float = 5e-4
maddpg_critic_lr: float = 1e-3
pdqn_lr: float = 1e-3
replay_size: int = 50_000
warmup_steps: int = 1_000
noise_start: float = 0.35
noise_final: float = 0.05
epsilon_start: float = 1.0
epsilon_final: float = 0.05
target_update_interval: int = 500
```

- [ ] **Step 2: Add CLI argument**

In `parse_args()`:

```python
parser.add_argument(
    "--algorithm",
    type=str,
    default="mappo",
    choices=["mappo", "maddpg", "pdqn"],
    help="Training algorithm: mappo, maddpg, or pdqn",
)
```

Copy `args.algorithm` into `config.algorithm`.

- [ ] **Step 3: Implement `HANMADDPGTrainer`**

Subclass `HANMAPPOTrainer` and override:

- `_init_mappo()` to create `self.algorithm = MADDPGAlgorithm(...)` and alias `self.maddpg = self.algorithm`.
- `_init_buffer()` to create `MultiAgentReplayBuffer`.
- `train()` to run one-step off-policy collection and update.
- `_evaluate()` to use deterministic MADDPG actor.
- `_save_checkpoint()` to save MADDPG states and `han_state_dict`.
- `load_checkpoint()` to load MADDPG states and `han_state_dict`.

Keep `_encode_graph_state()` inherited. Do not remove the no-grad cache.

Training loop shape:

```python
self.han_encoder.eval()
obs, _ = self.env.reset(seed=self.config.seed)
observations, sat_embeddings, masks = self._encode_graph_state()

for step_idx in range(self.config.total_timesteps):
    if step_idx < self.config.warmup_steps:
        env_actions, action_features, _ = self.algorithm.random_actions(masks)
    else:
        env_actions, action_features, _ = self.algorithm.act(observations, masks, deterministic=False)

    _, reward, terminated, truncated, _ = self.env.step(env_actions, return_observation=False, return_info=False)
    done = terminated or truncated
    next_observations, next_sat_embeddings, next_masks = self._encode_graph_state()
    self.buffer.add(observations, action_features, reward, next_observations, done, masks, next_masks)

    if len(self.buffer) >= max(self.config.batch_size, self.config.warmup_steps):
        update_stats = self.algorithm.update(self.buffer)

    observations, sat_embeddings, masks = next_observations, next_sat_embeddings, next_masks
    self.total_steps += 1
    if done:
        self.episodes += 1
        self.env.reset(seed=self.config.seed + self.total_steps)
        observations, sat_embeddings, masks = self._encode_graph_state()
```

Use the existing `HANMAPPOTrainer._empty_env_stats`, `_accumulate_env_stats`, `_save_training_history`, and record fields as much as possible.

- [ ] **Step 4: Implement `HANPDQNTrainer`**

Subclass `HANMADDPGTrainer` and override:

- `_init_mappo()` or `_init_algorithm()` to create `PDQNAlgorithm`.
- action selection call to use `epsilon` rather than Gaussian noise.
- checkpoint save/load names.

- [ ] **Step 5: Update `main()` dispatch**

Replace:

```python
trainer = HANMAPPOTrainer(config)
```

with:

```python
trainer_cls = {
    "mappo": HANMAPPOTrainer,
    "maddpg": HANMADDPGTrainer,
    "pdqn": HANPDQNTrainer,
}[config.algorithm]
trainer = trainer_cls(config)
```

- [ ] **Step 6: Run existing trainer integration tests**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests/test_han_integration.py tests/test_mappo_entropy.py -v
```

Expected: pass. Default MAPPO behavior must remain unchanged.

- [ ] **Step 7: Run smoke training**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe scripts/train.py --algorithm maddpg --total_timesteps 2000 --max_steps 100 --eval_episodes 1 --save_path results/smoke_han_maddpg --log_path results/smoke_logs --device cpu
C:\Users\19704\.conda\envs\satellite.env\python.exe scripts/train.py --algorithm pdqn --total_timesteps 2000 --max_steps 100 --eval_episodes 1 --save_path results/smoke_han_pdqn --log_path results/smoke_logs --device cpu
```

Expected: both complete, write `training_history.json`, and save `final_model.pt`.

---

### Task 5: Baseline Comparison Integration

**Files:**
- Modify: `scripts/compare_system_baselines.py`

- [ ] **Step 1: Update default baselines and names**

Set:

```python
DEFAULT_BASELINES = [
    "random",
    "min_distance",
    "full_local",
    "joint_greedy",
    "maddpg",
    "pdqn",
    "mappo_no_han",
    "han_maddpg",
    "han_pdqn",
]
```

Update `DISPLAY_NAME_MAP`:

```python
"pdqn": "PDQN",
"han_maddpg": "HAN+MADDPG",
"han_pdqn": "HAN+PDQN",
```

Keep DQN code only if removing it would cause large churn, but do not include DQN in defaults. If removing code, also remove DQN CLI options and dispatch cleanly.

- [ ] **Step 2: Add raw PDQN baseline function**

Implement:

```python
def train_and_evaluate_pdqn_baseline(...):
    # use raw env observations from env.reset/step
    # obs_dim = env.user_obs_dim
    # algorithm = PDQNAlgorithm(PDQNConfig(obs_dim=obs_dim, ...))
    # replay = MultiAgentReplayBuffer(...)
    # train off-policy like raw MADDPG
    # evaluate deterministic epsilon=0.0
```

Save:

```text
output_dir / "learned_baselines" / "pdqn" / "training_history.json"
output_dir / "learned_baselines" / "pdqn" / "pdqn_model.pt"
```

Return result from `summarize_results("pdqn", ...)`.

- [ ] **Step 3: Add HAN+MADDPG baseline function**

Implement:

```python
def train_and_evaluate_han_maddpg_baseline(...):
    save_dir = output_dir / "learned_baselines" / "han_maddpg"
    config = train_config_from_dict(..., save_path=save_dir, exp_name="han_maddpg")
    config.algorithm = "maddpg"
    trainer = HANMADDPGTrainer(config)
    trainer.train()
    checkpoint = save_dir / "best_model.pt" or "final_model.pt"
    result = evaluate_han_maddpg_checkpoint(...)
```

Evaluation should instantiate `HANMADDPGTrainer`, load checkpoint, run deterministic `algorithm.act()`, and call `summarize_results("han_maddpg", ...)`.

- [ ] **Step 4: Add HAN+PDQN baseline function**

Same as HAN+MADDPG but:

```python
config.algorithm = "pdqn"
trainer = HANPDQNTrainer(config)
method_name = "han_pdqn"
```

- [ ] **Step 5: Update dispatch**

In the baseline loop:

```python
elif baseline_name == "pdqn":
    result = train_and_evaluate_pdqn_baseline(...)
    result["source"] = "pdqn_train_eval"
elif baseline_name == "han_maddpg":
    result = train_and_evaluate_han_maddpg_baseline(...)
    result["source"] = "han_maddpg_train_eval"
elif baseline_name == "han_pdqn":
    result = train_and_evaluate_han_pdqn_baseline(...)
    result["source"] = "han_pdqn_train_eval"
```

- [ ] **Step 6: Add CLI timesteps for PDQN**

Add:

```python
parser.add_argument("--pdqn-timesteps", type=int, default=None, help="Training steps for the PDQN baseline. Defaults to --total-timesteps.")
```

Use `args.pdqn_timesteps or args.total_timesteps`.

- [ ] **Step 7: Run baseline smoke test**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe scripts/compare_system_baselines.py --baselines random maddpg pdqn han_maddpg han_pdqn --total-timesteps 2000 --maddpg-timesteps 2000 --pdqn-timesteps 2000 --episodes 1 --max-steps 100 --device cpu
```

Expected: completes and writes summary JSON/CSV.

---

### Task 6: Verification And Regression

**Files:**
- Existing tests
- Optional: `docs/EXPERIMENT_LOG.md`

- [ ] **Step 1: Run focused tests**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests/test_replay_buffer.py tests/test_maddpg_algorithm.py tests/test_pdqn_algorithm.py tests/test_han_integration.py -v
```

Expected: pass.

- [ ] **Step 2: Run full tests if time allows**

Run:

```powershell
C:\Users\19704\.conda\envs\satellite.env\python.exe -m pytest tests/ -v
```

Expected: pass. If slow, record which tests were run and which were skipped.

- [ ] **Step 3: Run GPU smoke on server**

Run with CUDA on the 3090:

```bash
python scripts/train.py --algorithm maddpg --total_timesteps 5000 --max_steps 200 --eval_episodes 1 --device cuda --save_path results/smoke_han_maddpg_cuda
python scripts/train.py --algorithm pdqn --total_timesteps 5000 --max_steps 200 --eval_episodes 1 --device cuda --save_path results/smoke_han_pdqn_cuda
```

Expected: no CUDA OOM, losses logged, final checkpoints saved.

- [ ] **Step 4: Record experiment protocol**

Update `docs/EXPERIMENT_LOG.md` before final reporting with:

- code commit hash
- current reward/environment setting is post-5/12
- methods to rerun strictly: `han_mappo`, `mappo_no_han`, `maddpg`, `han_maddpg`, `pdqn`, `han_pdqn`
- heuristic methods can be re-evaluated cheaply
- old 20260510 learned results are historical reference only

---

## Main Run Recommendation After Implementation

Because the environment reward changed after the 20260510 run, do not mix old learned results into strict current-code comparisons.

Recommended strict run:

```bash
python scripts/compare_system_baselines.py \
  --baselines random min_distance full_local joint_greedy maddpg pdqn mappo_no_han han_maddpg han_pdqn \
  --total-timesteps 1200000 \
  --maddpg-timesteps 1200000 \
  --pdqn-timesteps 1200000 \
  --episodes 10 \
  --device cuda
```

If server time is constrained, run in stages:

```bash
python scripts/compare_system_baselines.py --baselines random min_distance full_local joint_greedy --episodes 10 --device cuda
python scripts/compare_system_baselines.py --baselines maddpg han_maddpg --total-timesteps 1200000 --maddpg-timesteps 1200000 --episodes 10 --device cuda
python scripts/compare_system_baselines.py --baselines pdqn han_pdqn --total-timesteps 1200000 --pdqn-timesteps 1200000 --episodes 10 --device cuda
python scripts/compare_system_baselines.py --baselines mappo_no_han --total-timesteps 1200000 --episodes 10 --device cuda
```

Also rerun or load a current-code `HAN+MAPPO` system checkpoint for strict comparison.

---

## Self-Review Checklist

- DQN and DQN+HAN are not in the default suite.
- Random is in the default suite.
- HAN+MAPPO remains available and unchanged by default.
- No environment or reward files are modified.
- HAN+MADDPG and HAN+PDQN use no-grad cached HAN features, not end-to-end HAN replay.
- MADDPG discrete handover branch uses masked logits, argmax execution, and straight-through one-hot during actor update.
- PDQN uses Q-gradient for parameter networks, not behavior cloning as the primary parameter objective.
- Tests cover buffer shape, MADDPG action/update, PDQN action/update.
- Smoke commands are included for CPU and CUDA.
