import logging
from types import SimpleNamespace

import numpy as np
import torch

from src.algorithm.pdqn import PDQNAlgorithm, PDQNConfig
from src.algorithm.maddpg import MADDPGAlgorithm, MADDPGConfig
from src.algorithm.replay_buffer import MultiAgentReplayBuffer
from scripts.train import HANMADDPGTrainer


class _FakeEnv:
    def __init__(self, name):
        self.name = name
        self.reset_calls = 0
        self.step_calls = 0
        self.closed = False
        self.config = SimpleNamespace(seed=7)

    def reset(self, seed=None):
        self.reset_calls += 1
        return None, {}

    def step(self, actions, return_observation=False, return_info=False):
        self.step_calls += 1
        return None, 1.0, True, False, {}

    def get_stats_summary(self):
        return {
            "total_tasks": 1,
            "completed_tasks": 1,
            "resolved_tasks": 1,
            "total_energy": 1.0,
            "total_user_seconds": 1.0,
            "service_continuity_rate": 1.0,
            "service_availability_rate": 1.0,
            "task_completion_rate": 1.0,
            "task_success_rate": 1.0,
            "task_resolution_rate": 1.0,
            "avg_delay": 1.0,
            "effective_latency_score": 0.5,
        }

    def close(self):
        self.closed = True


def test_han_offpolicy_evaluation_does_not_reset_training_env():
    trainer = HANMADDPGTrainer.__new__(HANMADDPGTrainer)
    training_env = _FakeEnv("train")
    eval_env = _FakeEnv("eval")
    trainer.env = training_env
    trainer.config = SimpleNamespace(eval_episodes=1, seed=7, best_model_metric="effective_latency_score")
    trainer.episodes = 0
    trainer.total_steps = 100
    trainer.best_reward = float("-inf")
    trainer.best_model_score = float("-inf")
    trainer.eval_history = []
    trainer.logger = logging.getLogger("test_han_offpolicy_evaluation")

    trainer._empty_env_stats = lambda: {
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
    trainer._accumulate_env_stats = lambda target, source: target.update(source) or target
    trainer._create_eval_env = lambda: eval_env
    trainer._save_checkpoint = lambda best=False, final=False: None
    trainer._reset_encoded_env = lambda seed=None: (
        trainer.env.reset(seed=seed),
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        np.ones((1, 1), dtype=bool),
    )[1:]
    trainer._encode_graph_state = lambda: (
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        np.ones((1, 1), dtype=bool),
    )
    trainer._select_eval_action = lambda observations, masks: (
        np.array([[0.0, 0.0]], dtype=np.float32),
        None,
        None,
    )

    trainer._evaluate()

    assert trainer.env is training_env
    assert training_env.reset_calls == 0
    assert training_env.step_calls == 0
    assert eval_env.reset_calls == 1
    assert eval_env.step_calls == 1
    assert eval_env.closed


def test_replay_buffer_preserves_per_agent_rewards():
    buffer = MultiAgentReplayBuffer(
        capacity=4,
        num_agents=2,
        obs_dim=3,
        action_feature_dim=4,
        mask_dim=3,
        device="cpu",
    )

    buffer.add(
        obs=np.zeros((2, 3), dtype=np.float32),
        action_features=np.zeros((2, 4), dtype=np.float32),
        reward=np.array([1.0, -2.0], dtype=np.float32),
        next_obs=np.ones((2, 3), dtype=np.float32),
        done=False,
        masks=np.ones((2, 3), dtype=bool),
        next_masks=np.ones((2, 3), dtype=bool),
    )

    batch = buffer.sample(1)

    assert batch["rewards"].shape == torch.Size([1, 2])
    assert torch.allclose(batch["rewards"], torch.tensor([[1.0, -2.0]]))


def test_pdqn_update_accepts_per_agent_rewards():
    buffer = MultiAgentReplayBuffer(
        capacity=8,
        num_agents=2,
        obs_dim=3,
        action_feature_dim=4,
        mask_dim=3,
        device="cpu",
    )
    for idx in range(4):
        action_features = np.zeros((2, 4), dtype=np.float32)
        action_features[:, 0] = 1.0
        buffer.add(
            obs=np.full((2, 3), idx, dtype=np.float32),
            action_features=action_features,
            reward=np.array([1.0, -1.0], dtype=np.float32),
            next_obs=np.full((2, 3), idx + 1, dtype=np.float32),
            done=False,
            masks=np.ones((2, 3), dtype=bool),
            next_masks=np.ones((2, 3), dtype=bool),
        )

    algorithm = PDQNAlgorithm(
        PDQNConfig(
            num_agents=2,
            obs_dim=3,
            max_candidates=2,
            batch_size=4,
            replay_size=8,
            device="cpu",
        )
    )

    stats = algorithm.update(buffer)

    assert "q_loss" in stats
    assert "param_loss" in stats


def test_maddpg_update_averages_per_agent_rewards_for_central_critic():
    buffer = MultiAgentReplayBuffer(
        capacity=8,
        num_agents=2,
        obs_dim=3,
        action_feature_dim=4,
        mask_dim=3,
        device="cpu",
    )
    for idx in range(4):
        action_features = np.zeros((2, 4), dtype=np.float32)
        action_features[:, 0] = 1.0
        buffer.add(
            obs=np.full((2, 3), idx, dtype=np.float32),
            action_features=action_features,
            reward=np.array([2.0, -1.0], dtype=np.float32),
            next_obs=np.full((2, 3), idx + 1, dtype=np.float32),
            done=False,
            masks=np.ones((2, 3), dtype=bool),
            next_masks=np.ones((2, 3), dtype=bool),
        )

    algorithm = MADDPGAlgorithm(
        MADDPGConfig(
            num_agents=2,
            obs_dim=3,
            max_candidates=2,
            actor_hidden_dims=(8,),
            critic_hidden_dims=(16,),
            batch_size=4,
            replay_size=8,
            device="cpu",
        )
    )

    stats = algorithm.update(buffer)

    assert "actor_loss" in stats
    assert "critic_loss" in stats
