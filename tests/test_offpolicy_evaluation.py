import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from src.algorithm.pdqn import PDQNAlgorithm, PDQNConfig
from src.algorithm.maddpg import MADDPGAlgorithm, MADDPGConfig
from src.algorithm.replay_buffer import MultiAgentReplayBuffer
from scripts import compare_system_baselines as compare_baselines
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
            "handover_frequency": 0.0,
        }

    def close(self):
        self.closed = True


def test_han_offpolicy_evaluation_does_not_reset_training_env():
    trainer = HANMADDPGTrainer.__new__(HANMADDPGTrainer)
    training_env = _FakeEnv("train")
    eval_env = _FakeEnv("eval")
    trainer.env = training_env
    trainer.config = SimpleNamespace(eval_episodes=1, seed=7, best_model_metric="avg_delay")
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
        "handover_frequency": 0.0,
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


def test_pdqn_defaults_use_fast_exploration_schedule_and_light_bc():
    config = PDQNConfig()

    assert config.epsilon_final == 0.02
    assert config.bc_loss_coef == 0.001


def test_pdqn_networks_normalize_observation_features():
    algorithm = PDQNAlgorithm(
        PDQNConfig(
            num_agents=2,
            obs_dim=3,
            max_candidates=2,
            q_hidden_dims=(8,),
            param_hidden_dims=(8,),
            device="cpu",
        )
    )

    assert isinstance(algorithm.q_net.obs_norm, nn.LayerNorm)
    assert tuple(algorithm.q_net.obs_norm.normalized_shape) == (3,)
    for param_net in algorithm.param_nets.nets:
        assert isinstance(param_net.obs_norm, nn.LayerNorm)
        assert tuple(param_net.obs_norm.normalized_shape) == (3,)


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


def test_evaluate_maddpg_policy_uses_shared_algorithm_core(monkeypatch):
    class TinyEvalEnv:
        def __init__(self):
            self.actions = []

        def reset(self, seed=None):
            self.seed = seed
            return np.zeros((2, 3), dtype=np.float32), {}

        def step(self, actions):
            self.actions.append(np.asarray(actions, dtype=np.float32).copy())
            return np.zeros((2, 3), dtype=np.float32), 1.5, True, False, {}

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
                "handover_frequency": 0.0,
            }

    class RecordingAlgorithm:
        def __init__(self):
            self.calls = []

        def act(self, observations, masks, deterministic=False):
            self.calls.append((observations.copy(), masks.copy(), deterministic))
            env_actions = np.array([[0.0, 0.25], [1.0, 0.75]], dtype=np.float32)
            return env_actions, np.zeros((2, 4), dtype=np.float32), np.array([0, 1])

    env = TinyEvalEnv()
    algorithm = RecordingAlgorithm()
    monkeypatch.setattr(
        compare_baselines,
        "build_env_for_objective",
        lambda objective, config, seed, max_steps: env,
    )
    monkeypatch.setattr(
        compare_baselines,
        "maddpg_action_mask",
        lambda env: np.array([[True, False, False], [True, True, False]], dtype=bool),
    )

    result = compare_baselines.evaluate_maddpg_policy(
        algorithm=algorithm,
        objective="multi_objective",
        config={"min_effective_offload_ratio": 0.05},
        episodes=1,
        seed=7,
        max_steps=1,
    )

    assert algorithm.calls[0][2] is True
    assert env.actions[0].tolist() == [[0.0, 0.25], [1.0, 0.75]]
    assert result["handover_action_rate"] == 0.5
    assert result["mean_offload_ratio"] == 0.5


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


def test_raw_pdqn_training_history_records_shared_algorithm_config(tmp_path, monkeypatch):
    class TinyEnv:
        user_obs_dim = 3
        num_users = 2
        max_visible_sats = 2

        def reset(self, seed=None):
            return np.zeros((self.num_users, self.user_obs_dim), dtype=np.float32), {}

    monkeypatch.setattr(
        compare_baselines,
        "build_env_for_objective",
        lambda objective, config, seed, max_steps: TinyEnv(),
    )
    monkeypatch.setattr(
        compare_baselines,
        "evaluate_pdqn_policy",
        lambda algorithm, objective, config, episodes, seed, max_steps: {
            "method": "pdqn",
            "display_name": "PDQN",
            "episodes": episodes,
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "episode_metrics": [],
        },
    )

    result = compare_baselines.train_and_evaluate_pdqn_baseline(
        config={"epsilon_decay_fraction": 0.4},
        objective="multi_objective",
        output_dir=tmp_path,
        episodes=1,
        seed=77,
        max_steps=1,
        total_timesteps=0,
        device_name="cpu",
    )

    history = json.loads(Path(result["training_history"]).read_text(encoding="utf-8"))
    history_config = history["config"]
    assert history_config["batch_size"] == 128
    assert history_config["warmup_steps"] == 1_000
    assert history_config["replay_size"] == 50_000
    assert history_config["target_update_interval"] == 500
    assert history_config["epsilon_start"] == 1.0
    assert history_config["epsilon_final"] == 0.02
    assert history_config["epsilon_decay_steps"] == 1_001
    assert history_config["bc_loss_coef"] == 0.001
    assert history_config["seed"] == 77
