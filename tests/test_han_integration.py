import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import numpy as np
import pytest

from scripts.train import (
    ENVIRONMENT_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    HANMAPPOTrainer,
    HANPDQNTrainer,
    TrainConfig,
)


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


def test_han_mappo_observation_includes_raw_obs_and_han_without_duplicates():
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

        expected_dim = trainer_obj.raw_obs_dim + trainer_obj.config.han_out_dim
        assert trainer_obj.obs_dim == expected_dim
        assert observations.shape == (trainer_obj.num_agents, expected_dim)
        assert np.allclose(observations[:, : trainer_obj.raw_obs_dim], raw_obs)
    finally:
        trainer_obj.env.close()
        shutil.rmtree(save_path, ignore_errors=True)
        shutil.rmtree(log_path, ignore_errors=True)


def test_han_attn_trainer_appends_shared_constraints_and_context_features():
    from scripts.train import HANCandidateAttentionMAPPOTrainer
    from src.features.satellite_load import (
        SATELLITE_CONTEXT_FEATURE_DIM,
        SHARED_CONSTRAINT_DIM,
    )

    run_id = uuid4().hex
    save_path = f"results/han_attn_obs_{run_id}"
    log_path = f"results/han_attn_obs_logs_{run_id}"
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
        algorithm="han_attn",
    )
    trainer_obj = HANCandidateAttentionMAPPOTrainer(config)
    try:
        trainer_obj.env.reset(seed=trainer_obj.config.seed)
        observations, satellite_tokens, available_actions, candidate_sat_ids = (
            trainer_obj._encode_graph_state()
        )

        expected_obs_dim = (
            trainer_obj.raw_obs_dim
            + trainer_obj.config.han_out_dim
            + SHARED_CONSTRAINT_DIM
        )
        expected_sat_dim = trainer_obj.config.han_out_dim + SATELLITE_CONTEXT_FEATURE_DIM
        assert trainer_obj.algorithm_name == "han_attn"
        assert trainer_obj.obs_dim == expected_obs_dim
        assert observations.shape == (trainer_obj.num_agents, expected_obs_dim)
        assert satellite_tokens.shape == (trainer_obj.env.num_satellites, expected_sat_dim)
        assert trainer_obj.mappo.config.sat_embed_dim == expected_sat_dim
        assert available_actions.shape == (
            trainer_obj.num_agents,
            trainer_obj.max_candidates + 1,
        )
        assert candidate_sat_ids.shape == (trainer_obj.num_agents, trainer_obj.max_candidates)
    finally:
        trainer_obj.env.close()
        shutil.rmtree(save_path, ignore_errors=True)
        shutil.rmtree(log_path, ignore_errors=True)


def test_han_attn_trainer_act_path_accepts_augmented_shapes():
    from scripts.train import HANCandidateAttentionMAPPOTrainer

    run_id = uuid4().hex
    save_path = f"results/han_attn_act_{run_id}"
    log_path = f"results/han_attn_act_logs_{run_id}"
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
        algorithm="han_attn",
    )
    trainer_obj = HANCandidateAttentionMAPPOTrainer(config)
    try:
        trainer_obj.env.reset(seed=trainer_obj.config.seed)
        observations, satellite_tokens, available_actions, candidate_sat_ids = (
            trainer_obj._encode_graph_state()
        )

        actions, log_probs, value = trainer_obj.mappo.act(
            observations,
            available_actions,
            satellite_embeddings=satellite_tokens,
            candidate_sat_ids=candidate_sat_ids,
        )

        assert actions["handover"].shape == (trainer_obj.num_agents,)
        assert actions["offload"].shape == (trainer_obj.num_agents,)
        assert log_probs.shape == (trainer_obj.num_agents,)
        assert np.isfinite(value)
    finally:
        trainer_obj.env.close()
        shutil.rmtree(save_path, ignore_errors=True)
        shutil.rmtree(log_path, ignore_errors=True)


def test_han_pdqn_observation_includes_raw_obs_and_han_without_duplicates():
    import torch
    from src.model.hetero_gnn import HANConfig, HANEncoder

    run_id = uuid4().hex
    save_path = f"results/han_pdqn_obs_{run_id}"
    log_path = f"results/han_pdqn_obs_logs_{run_id}"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    pretrained_path = Path(save_path) / "pretrained_han.pt"
    encoder = HANEncoder(
        HANConfig(
            satellite_in_dim=10,
            user_in_dim=16,
            hidden_dim=64,
            out_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            use_edge_features=True,
            user_sat_edge_dim=5,
            isl_edge_dim=3,
        )
    )
    torch.save(
        {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "geometry_schema_version": 3,
            "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
            "han_state_dict": encoder.state_dict(),
        },
        pretrained_path,
    )
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
        pretrained_han_path=str(pretrained_path),
    )
    trainer_obj = HANPDQNTrainer(config)
    try:
        trainer_obj.env.reset(seed=trainer_obj.config.seed)
        observations, _, _, _ = trainer_obj._encode_graph_state()
        raw_obs = trainer_obj.env._get_observation()

        assert trainer_obj.obs_dim == trainer_obj.raw_obs_dim + trainer_obj.config.han_out_dim
        assert observations.shape == (trainer_obj.num_agents, trainer_obj.obs_dim)
        assert np.allclose(observations[:, : trainer_obj.raw_obs_dim], raw_obs)
        assert observations.shape[1] == (
            trainer_obj.raw_obs_dim + trainer_obj.config.han_out_dim
        )
    finally:
        trainer_obj.env.close()
        shutil.rmtree(save_path, ignore_errors=True)
        shutil.rmtree(log_path, ignore_errors=True)


def test_attention_mappo_trainer_builds_attention_state_without_han():
    from scripts.train import AttentionMAPPOTrainer

    run_id = uuid4().hex
    save_path = f"results/attn_mappo_obs_{run_id}"
    log_path = f"results/attn_mappo_obs_logs_{run_id}"
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
        algorithm="attn_mappo",
    )
    trainer_obj = AttentionMAPPOTrainer(config)
    try:
        trainer_obj.env.reset(seed=trainer_obj.config.seed)
        observations, satellite_features, available_actions, candidate_sat_ids = (
            trainer_obj._encode_graph_state()
        )
        raw_obs = trainer_obj.env._get_observation()

        assert trainer_obj.obs_dim == trainer_obj.raw_obs_dim
        assert observations.shape == (trainer_obj.num_agents, trainer_obj.raw_obs_dim)
        assert np.allclose(observations, raw_obs)
        assert satellite_features.shape == (
            trainer_obj.env.num_satellites,
            trainer_obj.sat_feature_dim,
        )
        assert available_actions.shape == (
            trainer_obj.num_agents,
            trainer_obj.max_candidates + 1,
        )
        assert candidate_sat_ids.shape == (
            trainer_obj.num_agents,
            trainer_obj.max_candidates,
        )
    finally:
        trainer_obj.env.close()
        shutil.rmtree(save_path, ignore_errors=True)
        shutil.rmtree(log_path, ignore_errors=True)


def test_attention_mappo_rollout_update_and_checkpoint_do_not_require_graph_cache(
    tmp_path,
):
    import torch
    from scripts.train import AttentionMAPPOTrainer

    save_path = tmp_path / "attn_mappo"
    log_path = tmp_path / "logs"
    trainer_obj = AttentionMAPPOTrainer(
        TrainConfig(
            num_users=2,
            max_steps=2,
            total_timesteps=2,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            eval_episodes=0,
            device="cpu",
            save_path=str(save_path),
            log_path=str(log_path),
            algorithm="attn_mappo",
        )
    )
    try:
        rollout_stats = trainer_obj.collect_rollouts()
        update_stats = trainer_obj.mappo.update()

        assert rollout_stats["total_steps"] == 2
        assert trainer_obj.buffer.pos == 2
        assert trainer_obj.mappo.representation_batch_fn is None
        assert trainer_obj.mappo.representation_optimizer is None
        assert all(
            snapshot is None
            for snapshot in trainer_obj.buffer.graph_snapshots[:trainer_obj.buffer.pos]
        )
        assert np.isfinite(update_stats["actor_loss"])
        assert np.isfinite(update_stats["critic_loss"])

        trainer_obj._save_checkpoint(final=True)
        checkpoint_path = save_path / "final_model.pt"
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["han_optimizer_state_dict"] is None
        trainer_obj.load_checkpoint(str(checkpoint_path))
    finally:
        trainer_obj.env.close()
        for handler in trainer_obj.logger.handlers:
            handler.close()
        trainer_obj.logger.handlers.clear()


def test_han_attention_rollout_preserves_context_and_updates_han(tmp_path):
    import torch
    from scripts.train import HANCandidateAttentionMAPPOTrainer
    from src.features.satellite_load import (
        SATELLITE_CONTEXT_FEATURE_DIM,
        SHARED_CONSTRAINT_DIM,
    )

    save_path = tmp_path / "han_attn"
    log_path = tmp_path / "logs"
    trainer_obj = HANCandidateAttentionMAPPOTrainer(
        TrainConfig(
            num_users=2,
            max_steps=2,
            total_timesteps=2,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            eval_episodes=0,
            device="cpu",
            save_path=str(save_path),
            log_path=str(log_path),
            algorithm="han_attn",
        )
    )
    try:
        before = {
            name: parameter.detach().clone()
            for name, parameter in trainer_obj.han_encoder.named_parameters()
        }

        trainer_obj.collect_rollouts()
        update_stats = trainer_obj.mappo.update()

        assert trainer_obj.mappo.representation_batch_fn is not None
        assert trainer_obj.mappo.representation_optimizer is not None
        assert all(
            snapshot is not None
            for snapshot in trainer_obj.buffer.graph_snapshots[:trainer_obj.buffer.pos]
        )
        assert trainer_obj.buffer.observations.shape[-1] == (
            trainer_obj.raw_obs_dim
            + trainer_obj.config.han_out_dim
            + SHARED_CONSTRAINT_DIM
        )
        assert trainer_obj.buffer.satellite_embeddings.shape[-1] == (
            trainer_obj.config.han_out_dim
            + SATELLITE_CONTEXT_FEATURE_DIM
        )
        assert update_stats["han_grad_norm"] > 0.0
        assert update_stats["han_parameter_delta"] > 0.0
        assert any(
            not torch.equal(before[name], parameter.detach())
            for name, parameter in trainer_obj.han_encoder.named_parameters()
        )
    finally:
        trainer_obj.env.close()
        for handler in trainer_obj.logger.handlers:
            handler.close()
        trainer_obj.logger.handlers.clear()


def test_han_pdqn_warmup_disables_privileged_safe_heuristic_by_default():
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
    assert calls == [0.0]


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

    def act(
        self,
        observations,
        available_actions,
        satellite_embeddings=None,
        candidate_sat_ids=None,
        continuous_action_mask=None,
    ):
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
        self.user_tasks = {0: None, 1: None}
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
            "handover_frequency": 0.0,
        }


def test_collect_rollouts_stores_per_agent_local_plus_cooperative_reward():
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


def test_raw_candidate_link_rows_match_handover_candidate_ids():
    from src.environment.gym_env import EnvConfig, LEOSatelliteEnv

    env = LEOSatelliteEnv(EnvConfig(num_users=4, max_steps=5, seed=17))
    try:
        observations, _ = env.reset(seed=17)
        candidate_offset = 3 + 1 + 5
        for user_id, user in enumerate(env.user_manager.users):
            candidates = env._get_handover_candidates(user)[: env.max_visible_sats]
            for candidate_index, candidate in enumerate(candidates):
                row_start = candidate_offset + candidate_index * 6
                encoded_satellite_id = observations[user_id, row_start]
                assert encoded_satellite_id == pytest.approx(
                    candidate.sat_id / env.num_satellites
                )
                assert candidate.sat_id != user.serving_satellite
    finally:
        env.close()


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

    assert config.max_steps == 512
    assert config.n_steps == 1024
    assert config.pre_handover_rvt_sec == 60.0


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

        def act(
            self,
            observations,
            available_actions,
            satellite_embeddings=None,
            candidate_sat_ids=None,
            continuous_action_mask=None,
        ):
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

    assert args.max_steps == 512
    assert args.n_steps == 1024
    assert args.pre_handover_rvt_sec == 60.0


def test_server_training_entrypoint_is_removed_from_docs_and_scripts():
    project_root = Path(__file__).resolve().parents[1]

    assert not (project_root / "scripts" / "run_server_training.py").exists()
    assert "run_server_training.py" not in (project_root / "README.md").read_text(
        encoding="utf-8"
    )


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

    def fake_sample_all(
        user_embeddings,
        candidate_masks=None,
        deterministic=False,
        candidate_satellite_embeddings=None,
        continuous_action_mask=None,
    ):
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


def test_actor_offload_distribution_depends_on_handover_target():
    from src.model.actor import ActorConfig, HybridActor
    import torch

    torch.manual_seed(5)
    actor = HybridActor(
        ActorConfig(
            input_dim=16,
            hidden_dims=[32, 16],
            max_candidates=2,
            sat_embed_dim=8,
            dropout=0.0,
        )
    )
    actor.eval()
    observations = torch.randn(1, 16)
    candidate_embeddings = torch.zeros(1, 2, 8)
    candidate_embeddings[:, 1] = 2.0

    _, offload_dist = actor(
        observations,
        torch.ones(1, 3),
        candidate_embeddings,
    )

    assert offload_dist.mean.shape == (1, 3)
    assert not torch.allclose(
        offload_dist.mean[:, 1],
        offload_dist.mean[:, 2],
    )


def test_actor_zero_inflated_gate_emits_exact_local_or_valid_offload():
    from src.model.actor import ActorConfig, HybridActor
    import torch

    actor = HybridActor(
        ActorConfig(
            input_dim=8,
            hidden_dims=[16, 8],
            max_candidates=1,
            sat_embed_dim=4,
            min_offload_ratio=0.05,
            dropout=0.0,
        )
    )
    actor.eval()
    observations = torch.zeros(2, 8)
    masks = torch.ones(2, 2)

    with torch.no_grad():
        actor.offload_mode_head[-1].weight.zero_()
        actor.offload_mode_head[-1].bias.fill_(-20.0)
    _, local, _ = actor.sample(observations, masks, deterministic=True)
    assert torch.equal(local, torch.zeros_like(local))

    with torch.no_grad():
        actor.offload_mode_head[-1].bias.fill_(20.0)
    _, offload, _ = actor.sample(observations, masks, deterministic=True)
    assert torch.all(offload >= 0.05)
    assert torch.all(offload <= 1.0)


def test_actor_zero_inflated_sample_log_prob_matches_ppo_evaluation():
    from src.model.actor import ActorConfig, HybridActor
    import torch

    torch.manual_seed(17)
    actor = HybridActor(
        ActorConfig(
            input_dim=8,
            hidden_dims=[16, 8],
            max_candidates=1,
            sat_embed_dim=4,
            min_offload_ratio=0.05,
            dropout=0.0,
        )
    )
    actor.eval()
    observations = torch.randn(4, 8)
    masks = torch.ones(4, 2)

    handover, offload, sampled_log_prob = actor.sample(
        observations,
        masks,
        deterministic=False,
    )
    evaluated_log_prob, _ = actor.evaluate(
        observations,
        handover,
        offload,
        masks,
    )

    assert torch.allclose(sampled_log_prob, evaluated_log_prob, atol=1e-5)


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
