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
    encoded_observations, satellite_embeddings, encoded_actions, candidate_sat_ids = (
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
    observations, satellite_embeddings, available_actions, candidate_sat_ids = (
        trainer._encode_graph_state()
    )
    broken_masks = available_actions[:, :-1]

    with pytest.raises(ValueError, match="candidate_masks must have shape"):
        trainer.mappo.act(
            observations,
            broken_masks,
            satellite_embeddings=satellite_embeddings,
        )


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
        observations, _, _, _ = trainer_obj._encode_graph_state()
        raw_obs = trainer_obj.env._get_observation()

        assert trainer_obj.obs_dim == trainer_obj.raw_obs_dim + trainer_obj.config.han_out_dim + 5
        assert observations.shape == (trainer_obj.num_agents, trainer_obj.obs_dim)
        assert np.allclose(observations[:, : trainer_obj.raw_obs_dim], raw_obs)
        han_start = trainer_obj.raw_obs_dim
        han_end = han_start + trainer_obj.config.han_out_dim
        light_features = observations[:, han_end:]
        assert light_features.shape == (trainer_obj.num_agents, 5)
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

    def act(self, observations, available_actions, satellite_embeddings=None, candidate_sat_ids=None):
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

    def get_pre_handover_mask(self):
        return np.zeros(len(self.last_user_rewards), dtype=bool)

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
        np.full((2, 2), -1, dtype=np.int64),
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


def test_env_exposes_pre_handover_mask_for_low_rvt():
    from src.environment.gym_env import EnvConfig, LEOSatelliteEnv

    env = LEOSatelliteEnv(EnvConfig(num_users=2, max_steps=5, pre_handover_rvt_sec=30.0, seed=7))
    env.reset(seed=7)

    mask = env.get_pre_handover_mask()

    assert mask.shape == (2,)
    assert mask.dtype == np.bool_


def test_pre_handover_mask_blocks_switch_actions_for_safe_users():
    trainer_obj = HANMAPPOTrainer.__new__(HANMAPPOTrainer)
    trainer_obj.config = SimpleNamespace(num_agents=3, max_candidates=2)
    trainer_obj.num_agents = 3
    trainer_obj.max_candidates = 2

    available = np.ones((trainer_obj.num_agents, trainer_obj.max_candidates + 1), dtype=np.float32)
    pre_mask = np.zeros(trainer_obj.num_agents, dtype=bool)

    gated = trainer_obj._apply_pre_handover_action_mask(available, pre_mask)

    assert np.all(gated[:, 0] == 1.0)
    assert np.all(gated[:, 1:] == 0.0)


def test_training_defaults_match_prehandover_plan():
    from scripts.train import TrainConfig

    config = TrainConfig()

    assert config.max_steps == 600
    assert config.n_steps == 1024
    assert config.pre_handover_rvt_sec == 30.0


def test_graph_contains_visible_serving_nearby_and_isl_relations():
    from scripts.train import TrainConfig, HANMAPPOTrainer

    trainer = HANMAPPOTrainer(TrainConfig(num_users=4, max_steps=5, n_steps=4, device="cpu"))
    trainer.env.reset(seed=11)
    graph = trainer.graph_builder.build(trainer.env)

    assert ("user", "visible", "satellite") in graph.edge_index
    assert ("satellite", "visible_rev", "user") in graph.edge_index
    assert ("user", "serving", "satellite") in graph.edge_index
    assert ("satellite", "serving_rev", "user") in graph.edge_index
    assert ("user", "nearby", "user") in graph.edge_index
    assert ("satellite", "isl", "satellite") in graph.edge_index


def test_collect_rollouts_clears_han_cache_after_episode_reset():
    from types import SimpleNamespace
    import numpy as np
    from scripts.train import HANMAPPOTrainer

    trainer = HANMAPPOTrainer.__new__(HANMAPPOTrainer)
    trainer.config = SimpleNamespace(n_steps=1)
    trainer.num_agents = 1
    trainer.total_steps = 0
    trainer.episodes = 0
    trainer.recent_rewards = []
    trainer._cached_han_user_embed = np.ones((1, 64), dtype=np.float32)
    trainer._cached_sat_embed = np.ones((2, 64), dtype=np.float32)

    class OneStepEnv:
        stats = {}

        def reset(self):
            return None, {}

        def step(self, actions, return_observation=False, return_info=False):
            return None, np.array([0.0], dtype=np.float32), False, True, {
                "user_rewards": np.array([0.0], dtype=np.float32)
            }

        def get_pre_handover_mask(self):
            return np.zeros(1, dtype=bool)

        def get_stats_summary(self):
            return {}

    class Mode:
        def eval(self):
            return None

    class Mappo:
        actor = Mode()
        critic = Mode()

        def act(self, observations, available_actions, satellite_embeddings=None, candidate_sat_ids=None):
            return (
                {"handover": np.zeros(1, dtype=np.int64), "offload": np.zeros(1, dtype=np.float32)},
                np.zeros(1, dtype=np.float32),
                0.0,
            )

        def get_value(self, observations, satellite_embeddings=None):
            return 0.0

    class Buffer:
        def reset(self):
            pass

        def add(self, **kwargs):
            pass

        def compute_returns_and_advantages(self, last_value, last_done):
            pass

    trainer.env = OneStepEnv()
    trainer.mappo = Mappo()
    trainer.han_encoder = Mode()
    trainer.buffer = Buffer()
    trainer._encode_graph_state = lambda: (
        np.zeros((1, 69), dtype=np.float32),
        np.zeros((2, 64), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.full((1, 3), -1, dtype=np.int64),
    )
    trainer._process_actions = lambda actions: np.column_stack(
        [actions["handover"], actions["offload"]]
    ).astype(np.float32)
    trainer._empty_env_stats = lambda: {}
    trainer._accumulate_env_stats = lambda target, source: target.update(source) or target

    trainer.collect_rollouts()

    assert trainer._cached_han_user_embed is None
    assert trainer._cached_sat_embed is None


def test_training_cli_defaults_match_prehandover_plan(monkeypatch):
    import sys
    import scripts.train as train_script

    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = train_script.parse_args()

    assert args.max_steps == 600
    assert args.n_steps == 1024
    assert args.pre_handover_rvt_sec == 30.0


def test_server_training_defaults_match_prehandover_plan():
    from scripts.run_server_training import (
        STANDARD_CONFIG,
        FAST_STANDARD_CONFIG,
        LARGE_SCALE_CONFIG,
        QUICK_TEST_CONFIG,
    )

    for config in [STANDARD_CONFIG, FAST_STANDARD_CONFIG, LARGE_SCALE_CONFIG, QUICK_TEST_CONFIG]:
        assert config["max_steps"] == 600
        assert config["n_steps"] == 1024
        assert config["pre_handover_rvt_sec"] == 30.0


def test_visible_edge_features_do_not_encode_serving_flag():
    from scripts.train import TrainConfig, HANMAPPOTrainer

    trainer = HANMAPPOTrainer(TrainConfig(num_users=3, max_steps=5, n_steps=4, device="cpu"))
    try:
        trainer.env.reset(seed=17)
        graph = trainer.graph_builder.build(trainer.env)
        visible_features = graph.edge_features[("user", "visible", "satellite")]

        assert visible_features.shape[1] == 5
    finally:
        trainer.env.close()


def test_graph_builder_metapaths_use_current_relation_names():
    from src.graph.builder import HeteroGraphBuilder

    metapaths = HeteroGraphBuilder().get_metapaths()
    flattened = [edge for path in metapaths for edge in path]

    assert ("user", "visible", "satellite") in flattened
    assert ("satellite", "visible_rev", "user") in flattened
    assert ("user", "serving", "satellite") in flattened
    assert ("satellite", "serving_rev", "user") in flattened
    assert ("user", "nearby", "user") in flattened
    assert ("user", "connect", "satellite") not in flattened
    assert ("satellite", "serve", "user") not in flattened


def test_isl_reverse_edges_have_matching_index_and_feature_counts():
    from scripts.train import TrainConfig, HANMAPPOTrainer

    trainer = HANMAPPOTrainer(TrainConfig(num_users=2, max_steps=5, n_steps=4, device="cpu"))
    try:
        trainer.env.reset(seed=23)
        graph = trainer.graph_builder.build(trainer.env)

        src, dst = graph.edge_index[("satellite", "isl", "satellite")]
        features = graph.edge_features[("satellite", "isl", "satellite")]

        assert src.shape == dst.shape
        assert features.shape[0] == src.shape[0]
    finally:
        trainer.env.close()


def test_mappo_act_passes_candidate_satellite_embeddings_to_actor(monkeypatch):
    import numpy as np
    import torch
    from src.algorithm.mappo import MAPPO, MAPPOConfig

    mappo = MAPPO(MAPPOConfig(
        num_agents=2,
        obs_dim=69,
        global_state_dim=138,
        max_candidates=2,
        sat_embed_dim=64,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        device="cpu",
    ))

    captured = {}

    def fake_sample_all(user_embeddings, candidate_masks=None, deterministic=False, candidate_satellite_embeddings=None):
        captured["candidate_satellite_embeddings"] = candidate_satellite_embeddings
        return {
            "handover": torch.zeros(2, dtype=torch.long),
            "offload": torch.zeros(2, dtype=torch.float32),
            "log_prob": torch.zeros(2, dtype=torch.float32),
        }

    monkeypatch.setattr(mappo.actor, "sample_all", fake_sample_all)

    observations = np.zeros((2, 69), dtype=np.float32)
    masks = np.ones((2, 3), dtype=np.float32)
    satellite_embeddings = np.arange(4 * 64, dtype=np.float32).reshape(4, 64)
    candidate_sat_ids = np.array([[1, 2], [3, -1]], dtype=np.int64)

    mappo.act(
        observations,
        masks,
        satellite_embeddings=satellite_embeddings,
        candidate_sat_ids=candidate_sat_ids,
    )

    gathered = captured["candidate_satellite_embeddings"]
    assert gathered.shape == (2, 2, 64)
    assert torch.allclose(gathered[0, 0], torch.tensor(satellite_embeddings[1]))
    assert torch.allclose(gathered[0, 1], torch.tensor(satellite_embeddings[2]))
    assert torch.allclose(gathered[1, 0], torch.tensor(satellite_embeddings[3]))
    assert torch.all(gathered[1, 1] == 0.0)


def test_actor_logits_change_when_candidate_satellite_embeddings_change():
    from src.model.actor import HybridActor, ActorConfig
    import torch

    actor = HybridActor(ActorConfig(input_dim=69, max_candidates=2))
    user = torch.zeros((1, 69), dtype=torch.float32)
    mask = torch.ones((1, 3), dtype=torch.float32)
    sat_a = torch.zeros((1, 2, 64), dtype=torch.float32)
    sat_b = torch.ones((1, 2, 64), dtype=torch.float32)

    logits_a = actor.handover_logits(user, mask, candidate_satellite_embeddings=sat_a)
    logits_b = actor.handover_logits(user, mask, candidate_satellite_embeddings=sat_b)

    assert not torch.allclose(logits_a, logits_b)


def test_actor_accepts_configured_candidate_satellite_embedding_dim():
    from src.model.actor import HybridActor, ActorConfig
    import torch

    actor = HybridActor(ActorConfig(input_dim=16, max_candidates=2, sat_embed_dim=32))
    user = torch.zeros((1, 16), dtype=torch.float32)
    mask = torch.ones((1, 3), dtype=torch.float32)
    sat_embeddings = torch.ones((1, 2, 32), dtype=torch.float32)

    logits = actor.handover_logits(
        user,
        mask,
        candidate_satellite_embeddings=sat_embeddings,
    )

    assert logits.shape == (1, 3)
