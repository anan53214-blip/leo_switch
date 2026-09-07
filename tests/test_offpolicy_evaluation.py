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
from scripts.validate_offload_bimodality import (
    ElevationJointGreedyPolicy,
    StayJointGreedyPolicy,
    paired_method_deltas,
)


def test_raw_pdqn_action_mask_applies_the_same_pre_handover_gate():
    users = [SimpleNamespace(user_id=0), SimpleNamespace(user_id=1)]
    env = SimpleNamespace(
        num_users=2,
        max_visible_sats=3,
        user_manager=SimpleNamespace(users=users),
        _get_handover_candidates=lambda user: [
            SimpleNamespace(sat_id=10),
            SimpleNamespace(sat_id=11),
        ],
        get_pre_handover_mask=lambda: np.asarray([False, True], dtype=bool),
    )

    masks = compare_baselines.pdqn_action_mask(env)

    assert masks[0].tolist() == [True, False, False, False]
    assert masks[1].tolist() == [True, True, True, False]


def test_raw_maddpg_action_mask_applies_the_same_pre_handover_gate():
    users = [SimpleNamespace(user_id=0), SimpleNamespace(user_id=1)]
    env = SimpleNamespace(
        num_users=2,
        max_visible_sats=2,
        user_manager=SimpleNamespace(users=users),
        _get_handover_candidates=lambda user: [
            SimpleNamespace(sat_id=10),
            SimpleNamespace(sat_id=11),
        ],
        get_pre_handover_mask=lambda: np.asarray([False, True], dtype=bool),
    )

    masks = compare_baselines.maddpg_action_mask(env)

    assert masks[0].tolist() == [True, False, False]
    assert masks[1].tolist() == [True, True, True]


def test_raw_maddpg_action_mask_uses_new_environment_feasibility_mask():
    expected = np.asarray(
        [
            [True, False, True],
            [True, True, False],
        ],
        dtype=bool,
    )
    env = SimpleNamespace(
        num_users=2,
        max_visible_sats=2,
        get_handover_action_mask=lambda max_candidates: expected.copy(),
    )

    masks = compare_baselines.maddpg_action_mask(env)

    assert np.array_equal(masks, expected)


def test_raw_dqn_action_mask_keeps_only_stay_bins_for_safe_users():
    users = [SimpleNamespace(user_id=0), SimpleNamespace(user_id=1)]
    env = SimpleNamespace(
        num_users=2,
        max_visible_sats=2,
        user_manager=SimpleNamespace(users=users),
        _get_handover_candidates=lambda user: [
            SimpleNamespace(sat_id=10),
            SimpleNamespace(sat_id=11),
        ],
        get_pre_handover_mask=lambda: np.asarray([False, True], dtype=bool),
    )

    masks = compare_baselines.dqn_action_mask(env, [0.0, 0.5, 1.0])

    assert masks[0].tolist() == [True, True, True, False, False, False, False, False, False]
    assert masks[1].tolist() == [True] * 9


def test_handover_actionability_reports_gate_blocked_load_relief():
    users = [
        SimpleNamespace(user_id=0, serving_satellite=0),
        SimpleNamespace(user_id=1, serving_satellite=0),
    ]
    current_server = SimpleNamespace(
        satellite_id=0,
        queue_length=3,
        task_queue=[
            {"status": "processing", "remaining_cycles": 10e9},
            {"status": "processing", "remaining_cycles": 10e9},
            {"status": "queued", "remaining_cycles": 10e9},
        ],
        total_capacity_ghz=10.0,
        utilization=0.5,
        is_full=False,
        config=SimpleNamespace(mec_max_concurrent_tasks=2),
    )
    target_server = SimpleNamespace(
        satellite_id=1,
        queue_length=0,
        task_queue=[],
        total_capacity_ghz=10.0,
        utilization=0.0,
        is_full=False,
        config=SimpleNamespace(mec_max_concurrent_tasks=2),
    )
    servers = {0: current_server, 1: target_server}

    def handover_mask(max_candidates, apply_pre_handover_gate):
        del max_candidates
        if apply_pre_handover_gate:
            return np.asarray([[1, 0], [1, 1]], dtype=bool)
        return np.asarray([[1, 1], [1, 1]], dtype=bool)

    env = SimpleNamespace(
        num_users=2,
        max_visible_sats=1,
        user_manager=SimpleNamespace(users=users),
        config=SimpleNamespace(pre_handover_rvt_sec=60.0),
        mec_manager=SimpleNamespace(
            get_server=lambda sat_id: servers.get(sat_id),
            prepare_user_task_migration=lambda **kwargs: SimpleNamespace(
                feasible=True,
                failure_reason=None,
            ),
        ),
        get_pre_handover_mask=lambda: np.asarray([False, True], dtype=bool),
        get_handover_action_mask=handover_mask,
        _get_handover_candidates=lambda user: [
            SimpleNamespace(sat_id=1)
        ],
        _get_satellite_visibility=lambda user, sat_id: SimpleNamespace(
            is_visible=True,
            rvt_seconds=120.0 if user.user_id == 0 else 20.0,
        ),
        _check_handover_link_feasibility=lambda candidate: (True, None),
    )
    accumulator = compare_baselines.HandoverActionabilityAccumulator()

    accumulator.observe(env)
    summary = accumulator.summary()

    assert summary["pre_handover_gate_open_rate"] == 0.5
    assert summary["ungated_feasible_switch_user_rate"] == 1.0
    assert summary["gated_feasible_switch_user_rate"] == 0.5
    assert summary["gate_block_share_of_feasible_switch_user_steps"] == 0.5
    assert summary["congestion_relief_opportunity_user_rate"] == 1.0
    assert summary["gate_blocked_congestion_relief_user_rate"] == 0.5
    assert summary["mean_positive_queue_reduction_tasks"] == 3.0
    assert summary["mean_positive_workload_wait_reduction_sec"] == 3.0
    assert summary["mean_serving_satellite_hhi"] == 1.0
    assert summary["mean_effective_serving_satellites"] == 1.0
    assert summary["gate_open_reason_rates"] == {"low_rvt": 1.0}
    assert summary["raw_candidate_status_rates"] == {"legal": 1.0}


def test_handover_ablation_policies_restrict_only_satellite_selection():
    visible = [
        SimpleNamespace(sat_id=1, elevation_deg=50.0),
        SimpleNamespace(sat_id=2, elevation_deg=70.0),
    ]
    connected = SimpleNamespace(user_id=0, serving_satellite=0)
    disconnected = SimpleNamespace(user_id=0, serving_satellite=-1)
    env = SimpleNamespace(
        max_visible_sats=2,
        _get_satellite_visibility=lambda user, sat_id: SimpleNamespace(
            is_visible=True,
            elevation_deg=60.0,
        ),
    )
    legal = np.asarray([1, 1, 1], dtype=bool)
    stay = StayJointGreedyPolicy("multi_objective", [0.0, 1.0])
    elevation = ElevationJointGreedyPolicy(
        "multi_objective",
        [0.0, 1.0],
    )

    assert stay._handover_candidate_actions(
        env,
        connected,
        visible,
        legal,
    ) == [0]
    assert stay._handover_candidate_actions(
        env,
        disconnected,
        visible,
        legal,
    ) == [0, 1, 2]
    assert elevation._handover_candidate_actions(
        env,
        connected,
        visible,
        legal,
    ) == [2]


def test_handover_ablation_uses_paired_episode_deltas():
    methods = [
        {
            "method": "joint",
            "episode_metrics": [
                {"reward": 3.0, "task_success_rate": 0.8},
                {"reward": 5.0, "task_success_rate": 0.9},
            ],
        },
        {
            "method": "stay",
            "episode_metrics": [
                {"reward": 1.0, "task_success_rate": 0.7},
                {"reward": 2.0, "task_success_rate": 0.7},
            ],
        },
    ]

    rows = paired_method_deltas(methods)

    assert len(rows) == 1
    assert rows[0]["control"] == "stay"
    assert rows[0]["reward_delta_mean"] == 2.5
    assert np.isclose(
        rows[0]["task_success_rate_delta_mean"],
        0.15,
    )


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


def test_pdqn_defaults_use_fast_exploration_schedule_without_privileged_bc():
    config = PDQNConfig()

    assert config.epsilon_final == 0.02
    assert config.bc_loss_coef == 0.0


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

    assert isinstance(algorithm.actor.obs_norm, nn.Identity)
    assert isinstance(algorithm.critic.obs_norm, nn.Identity)
    stats = algorithm.update(buffer)

    assert "actor_loss" in stats
    assert "critic_loss" in stats
    assert "q_mean" in stats
    assert "target_q_mean" in stats
    assert "td_abs_mean" in stats
    assert "actor_grad_norm" in stats
    assert "critic_grad_norm" in stats
    assert all(np.isfinite(value) for value in stats.values())


def test_maddpg_critic_normalizes_observations_without_changing_actor_inputs():
    algorithm = MADDPGAlgorithm(
        MADDPGConfig(
            num_agents=2,
            obs_dim=3,
            max_candidates=2,
            actor_hidden_dims=(8,),
            critic_hidden_dims=(16,),
            normalize_critic_observations=True,
            device="cpu",
        )
    )

    assert isinstance(algorithm.actor.obs_norm, nn.Identity)
    assert isinstance(algorithm.critic.obs_norm, nn.LayerNorm)
    assert tuple(algorithm.critic.obs_norm.normalized_shape) == (3,)


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


def test_raw_maddpg_logs_progress_with_new_environment_interface(tmp_path, monkeypatch):
    class TinyTrainingEnv:
        user_obs_dim = 3
        num_users = 2
        max_visible_sats = 2

        def __init__(self):
            self.episode_step = 0
            self.last_user_rewards = np.ones(self.num_users, dtype=np.float32)

        def reset(self, seed=None):
            self.episode_step = 0
            return np.zeros((self.num_users, self.user_obs_dim), dtype=np.float32), {}

        def step(self, actions):
            self.episode_step += 1
            done = self.episode_step >= 2
            observations = np.full(
                (self.num_users, self.user_obs_dim),
                self.episode_step,
                dtype=np.float32,
            )
            return observations, 1.0, done, False, {}

        def get_handover_action_mask(self, max_candidates):
            return np.ones(
                (self.num_users, max_candidates + 1),
                dtype=bool,
            )

    monkeypatch.setattr(
        compare_baselines,
        "build_env_for_objective",
        lambda objective, config, seed, max_steps: TinyTrainingEnv(),
    )
    monkeypatch.setattr(
        compare_baselines,
        "evaluate_maddpg_policy",
        lambda algorithm, objective, config, episodes, seed, max_steps: {
            "method": "maddpg",
            "display_name": "MADDPG",
            "episodes": episodes,
            "mean_reward": 2.0,
            "std_reward": 0.0,
            "episode_metrics": [{"episode": 0, "reward": 2.0}],
        },
    )

    result = compare_baselines.train_and_evaluate_maddpg_baseline(
        config={"log_interval": 1, "eval_episodes": 1},
        objective="multi_objective",
        output_dir=tmp_path,
        episodes=1,
        seed=17,
        max_steps=2,
        total_timesteps=4,
        device_name="cpu",
    )

    log_text = Path(result["training_log"]).read_text(encoding="utf-8")
    assert "Steps: 2/4" in log_text
    assert "Steps: 4/4" in log_text
    assert "LastEpReward: 2.000" in log_text
    history = json.loads(Path(result["training_history"]).read_text(encoding="utf-8"))
    assert history["config"]["log_interval_steps"] == 2
    assert history["config"]["seed"] == 17
    assert history["config"]["normalize_actor_observations"] is False
    assert history["config"]["normalize_critic_observations"] is True
    assert history["config"]["critic_huber_beta"] == 1.0
    assert history["config"]["reward_scale"] == 1.0
    assert history["config"]["raise_on_nonfinite_loss"] is True


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
    assert history_config["bc_loss_coef"] == 0.0
    assert history_config["safe_exploration_probability"] == 0.0
    assert history_config["seed"] == 77


def test_raw_pdqn_logs_progress_and_saves_periodic_checkpoints(tmp_path, monkeypatch):
    class TinyTrainingEnv:
        user_obs_dim = 3
        num_users = 2
        max_visible_sats = 2

        def __init__(self):
            self.episode_step = 0
            self.last_user_rewards = np.ones(self.num_users, dtype=np.float32)

        def reset(self, seed=None):
            self.episode_step = 0
            return np.zeros((self.num_users, self.user_obs_dim), dtype=np.float32), {}

        def step(self, actions):
            self.episode_step += 1
            done = self.episode_step >= 2
            observations = np.full(
                (self.num_users, self.user_obs_dim),
                self.episode_step,
                dtype=np.float32,
            )
            return observations, 1.0, done, False, {}

        def get_stats_summary(self):
            return {
                "total_tasks": 2,
                "completed_tasks": 2,
                "resolved_tasks": 2,
                "total_energy": 1.0,
                "total_user_seconds": 4.0,
                "service_continuity_rate": 1.0,
                "service_availability_rate": 1.0,
                "task_completion_rate": 1.0,
                "task_success_rate": 1.0,
                "task_resolution_rate": 1.0,
                "avg_delay": 1.0,
                "handover_frequency": 0.0,
            }

    monkeypatch.setattr(
        compare_baselines,
        "build_env_for_objective",
        lambda objective, config, seed, max_steps: TinyTrainingEnv(),
    )
    monkeypatch.setattr(
        compare_baselines,
        "pdqn_action_mask",
        lambda env: np.ones((env.num_users, env.max_visible_sats + 1), dtype=bool),
    )
    monkeypatch.setattr(
        compare_baselines,
        "pdqn_mixed_safe_random_actions",
        lambda env, algorithm, masks, safe_probability: (
            np.zeros((env.num_users, 2), dtype=np.float32),
            np.pad(
                np.ones((env.num_users, 1), dtype=np.float32),
                ((0, 0), (0, env.max_visible_sats + 1)),
            ),
            np.zeros(env.num_users, dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        compare_baselines,
        "evaluate_pdqn_policy",
        lambda algorithm, objective, config, episodes, seed, max_steps: {
            "method": "pdqn",
            "display_name": "PDQN",
            "episodes": episodes,
            "mean_reward": 2.0,
            "std_reward": 0.0,
            "episode_metrics": [{"episode": 0, "reward": 2.0}],
        },
    )

    result = compare_baselines.train_and_evaluate_pdqn_baseline(
        config={"save_interval": 2, "log_interval": 1},
        objective="multi_objective",
        output_dir=tmp_path,
        episodes=1,
        seed=17,
        max_steps=2,
        total_timesteps=4,
        device_name="cpu",
    )

    artifact_dir = tmp_path / "learned_baselines" / "pdqn"
    assert (artifact_dir / "checkpoint_2.pt").is_file()
    assert (artifact_dir / "checkpoint_4.pt").is_file()
    assert (artifact_dir / "pdqn_model.pt").is_file()

    log_text = Path(result["training_log"]).read_text(encoding="utf-8")
    assert "Steps: 2/4" in log_text
    assert "Steps: 4/4" in log_text
    assert "周期权重已保存" in log_text

    history = json.loads(Path(result["training_history"]).read_text(encoding="utf-8"))
    assert history["config"]["save_interval"] == 2
    assert history["config"]["log_interval_steps"] == 2
    assert history["summary"]["total_steps"] == 4
    assert history["summary"]["episodes"] == 2
    assert history["evaluation"] == [{"episode": 0, "reward": 2.0}]
