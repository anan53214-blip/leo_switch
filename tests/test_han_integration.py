import shutil
from types import SimpleNamespace
from uuid import uuid4
import numpy as np
import pytest

from scripts.train import HANMAPPOTrainer, HANPDQNTrainer, TrainConfig


@pytest.fixture(scope="module")
def trainer():
    run_id = uuid4().hex
    save_path = f"results/han_integration_{run_id}"
    log_path = f"results/han_integration_logs_{run_id}"
    config = TrainConfig(
        num_users=3,
        max_steps=20,
        total_timesteps=128,
        n_steps=32,
        batch_size=32,
        eval_episodes=1,
        device="cpu",
        save_path=save_path,
        log_path=log_path,
    )
    trainer_obj = HANMAPPOTrainer(config)
    yield trainer_obj
    trainer_obj.env.close()
    shutil.rmtree(save_path, ignore_errors=True)
    shutil.rmtree(log_path, ignore_errors=True)


def test_han_encoder_and_policy_share_consistent_shapes(trainer):
    obs, _ = trainer.env.reset(seed=trainer.config.seed)

    observations, global_state, available_actions = trainer._get_observations(obs)
    encoded_observations, satellite_embeddings, encoded_actions = (
        trainer._encode_graph_state()
    )

    assert observations.shape == (trainer.num_agents, trainer.obs_dim)
    assert encoded_observations.shape == observations.shape
    assert global_state.shape == (trainer.global_state_dim,)
    assert available_actions.shape == (
        trainer.num_agents,
        trainer.max_candidates + 1,
    )
    assert np.array_equal(available_actions, encoded_actions)
    assert satellite_embeddings.ndim == 2
    assert satellite_embeddings.shape[1] == trainer.config.han_out_dim

    actions, log_probs, value = trainer.mappo.act(
        encoded_observations,
        encoded_actions,
        satellite_embeddings=satellite_embeddings,
    )

    assert actions["handover"].shape == (trainer.num_agents,)
    assert actions["offload"].shape == (trainer.num_agents,)
    assert log_probs.shape == (trainer.num_agents,)
    assert np.isfinite(value)


def test_mappo_act_rejects_misaligned_candidate_mask_shape(trainer):
    observations, satellite_embeddings, available_actions = (
        trainer._encode_graph_state()
    )
    broken_masks = available_actions[:, :-1]

    with pytest.raises(ValueError, match="candidate_masks must have shape"):
        trainer.mappo.act(
            observations,
            broken_masks,
            satellite_embeddings=satellite_embeddings,
        )


def test_han_pdqn_observation_includes_raw_obs_han_and_light_features():
    run_id = uuid4().hex
    save_path = f"results/han_pdqn_obs_{run_id}"
    log_path = f"results/han_pdqn_obs_logs_{run_id}"
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
    trainer_obj = HANPDQNTrainer(config)
    try:
        trainer_obj.env.reset(seed=trainer_obj.config.seed)
        observations, _, _ = trainer_obj._encode_graph_state()
        raw_obs = trainer_obj.env._get_observation()

        assert trainer_obj.obs_dim == trainer_obj.raw_obs_dim + trainer_obj.config.han_out_dim + 5
        assert observations.shape == (trainer_obj.num_agents, trainer_obj.obs_dim)
        assert np.allclose(observations[:, : trainer_obj.raw_obs_dim], raw_obs)
    finally:
        trainer_obj.env.close()
        shutil.rmtree(save_path, ignore_errors=True)
        shutil.rmtree(log_path, ignore_errors=True)


def test_han_pdqn_warmup_uses_seventy_percent_safe_heuristic_mix():
    trainer_obj = HANPDQNTrainer.__new__(HANPDQNTrainer)
    trainer_obj.config = SimpleNamespace(warmup_steps=10)
    trainer_obj.algorithm = SimpleNamespace()
    calls = []

    def record_mix(masks, safe_probability=0.7):
        calls.append(float(safe_probability))
        return "mixed-actions"

    trainer_obj._mixed_safe_random_actions = record_mix

    result = trainer_obj._select_train_action(
        np.zeros((2, 3), dtype=np.float32),
        np.ones((2, 3), dtype=bool),
        step_idx=0,
    )

    assert result == "mixed-actions"
    assert calls == [0.7]


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
