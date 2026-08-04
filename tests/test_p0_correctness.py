import shutil
from uuid import uuid4

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.train import (
    ENVIRONMENT_SCHEMA_VERSION,
    HANMAPPOTrainer,
    MODEL_SCHEMA_VERSION,
    TrainConfig,
)
from src.algorithm.buffer import MultiAgentRolloutBuffer
from src.environment.gym_env import EnvConfig, LEOSatelliteEnv
from src.graph.builder import HeteroGraphBuilder
from src.model.hetero_gnn import HANConfig, HANEncoder


def _tensor_graph(graph):
    node_features = {
        key: torch.as_tensor(value, dtype=torch.float32)
        for key, value in graph.node_features.items()
    }
    edge_index = {
        key: (
            torch.as_tensor(value[0], dtype=torch.long),
            torch.as_tensor(value[1], dtype=torch.long),
        )
        for key, value in graph.edge_index.items()
    }
    edge_features = {
        key: torch.as_tensor(value, dtype=torch.float32)
        for key, value in graph.edge_features.items()
    }
    return node_features, edge_index, edge_features


def test_all_configured_metapaths_execute_with_consistent_shapes():
    env = LEOSatelliteEnv(
        EnvConfig(
            num_users=3,
            max_steps=5,
            randomize_episode_start=False,
            seed=7,
        )
    )
    try:
        env.reset(seed=7)
        graph = HeteroGraphBuilder(add_reverse_edges=True).build(env)
        encoder = HANEncoder(
            HANConfig(hidden_dim=64, out_dim=64, num_heads=4)
        )
        node_features, edge_index, edge_features = _tensor_graph(graph)
        projected = {
            key: F.relu(encoder.han_layers[0].node_projections[key](value))
            for key, value in node_features.items()
        }

        for name, metapath_encoder in encoder.han_layers[0].metapath_encoders.items():
            output = metapath_encoder(projected, edge_index, edge_features)
            expected_nodes = projected[metapath_encoder.output_node_type].shape[0]
            assert output.shape == (expected_nodes, 64), name
            assert torch.isfinite(output).all(), name
    finally:
        env.close()


def test_serving_metapath_uses_two_dimensional_edge_features():
    encoder = HANEncoder(HANConfig(hidden_dim=64, out_dim=64, num_heads=4))

    serving = encoder.han_layers[0].metapath_encoders["user-serving-satellite-user"]

    assert serving.layers[0].W_edge.in_features == 2
    assert serving.layers[1].W_edge.in_features == 2


def test_han_num_layers_controls_encoder_depth():
    one_layer = HANEncoder(HANConfig(num_layers=1))
    two_layers = HANEncoder(HANConfig(num_layers=2))

    assert len(one_layer.han_layers) == 1
    assert len(two_layers.han_layers) == 2
    assert sum(p.numel() for p in two_layers.parameters()) > sum(
        p.numel() for p in one_layer.parameters()
    )


def test_gae_does_not_cross_episode_boundary():
    buffer = MultiAgentRolloutBuffer(
        buffer_size=2,
        num_agents=1,
        obs_dim=1,
        global_state_dim=1,
        gamma=0.99,
        gae_lambda=0.95,
    )
    common = {
        "obs": np.zeros((1, 1), dtype=np.float32),
        "global_state": np.zeros(1, dtype=np.float32),
        "satellite_embeddings": None,
        "actions_discrete": np.zeros(1, dtype=np.int64),
        "actions_continuous": np.zeros(1, dtype=np.float32),
        "value": 0.0,
        "log_probs": np.zeros(1, dtype=np.float32),
    }
    buffer.add(rewards=np.array([1.0], dtype=np.float32), done=True, **common)
    buffer.add(rewards=np.array([10.0], dtype=np.float32), done=False, **common)

    buffer.compute_returns_and_advantages(last_value=0.0, last_done=True)

    assert buffer.advantages[0, 0] == pytest.approx(1.0)
    assert buffer.advantages[1, 0] == pytest.approx(10.0)


def test_time_limit_bootstraps_terminal_value_without_crossing_reset():
    buffer = MultiAgentRolloutBuffer(
        buffer_size=1,
        num_agents=1,
        obs_dim=1,
        global_state_dim=1,
        gamma=0.9,
        gae_lambda=0.95,
    )
    buffer.add(
        obs=np.zeros((1, 1), dtype=np.float32),
        global_state=np.zeros(1, dtype=np.float32),
        satellite_embeddings=None,
        actions_discrete=np.zeros(1, dtype=np.int64),
        actions_continuous=np.zeros(1, dtype=np.float32),
        rewards=np.ones(1, dtype=np.float32),
        done=True,
        value=2.0,
        log_probs=np.zeros(1, dtype=np.float32),
        timeout_bootstrap_value=3.0,
    )

    buffer.compute_returns_and_advantages(last_value=99.0, last_done=True)

    assert buffer.advantages[0, 0] == pytest.approx(1.0 + 0.9 * 3.0 - 2.0)


def test_rollout_buffer_grouped_batches_keep_complete_time_steps():
    buffer = MultiAgentRolloutBuffer(
        buffer_size=5,
        num_agents=3,
        obs_dim=1,
        global_state_dim=3,
        device="cpu",
    )
    common = {
        "global_state": np.zeros(3, dtype=np.float32),
        "satellite_embeddings": None,
        "actions_discrete": np.zeros(3, dtype=np.int64),
        "actions_continuous": np.zeros(3, dtype=np.float32),
        "rewards": np.zeros(3, dtype=np.float32),
        "done": False,
        "value": 0.0,
        "log_probs": np.zeros(3, dtype=np.float32),
    }
    for time_index in range(5):
        buffer.add(
            obs=np.full((3, 1), time_index, dtype=np.float32),
            **common,
        )

    batches = list(
        buffer.get_batches(
            batch_size=7,
            shuffle=False,
            group_by_time=True,
        )
    )

    observed_times = []
    for batch in batches:
        time_indices = batch["time_indices"].tolist()
        agent_indices = batch["agent_indices"].tolist()
        for time_index in sorted(set(time_indices)):
            agents = [
                agent
                for batch_time, agent in zip(time_indices, agent_indices)
                if batch_time == time_index
            ]
            assert agents == [0, 1, 2]
            observed_times.append(time_index)

    assert observed_times == [0, 1, 2, 3, 4]


def test_batched_graph_encoding_matches_separate_graphs():
    env = LEOSatelliteEnv(
        EnvConfig(
            num_users=3,
            max_steps=5,
            randomize_episode_start=False,
            seed=7,
        )
    )
    try:
        env.reset(seed=7)
        builder = HeteroGraphBuilder(add_reverse_edges=True)
        first_graph = builder.build(env)
        env._update_environment()
        second_graph = builder.build(env)

        trainer = object.__new__(HANMAPPOTrainer)
        trainer.device = torch.device("cpu")
        trainer.han_encoder = HANEncoder(
            HANConfig(
                hidden_dim=64,
                out_dim=64,
                num_heads=4,
                dropout=0.0,
            )
        ).eval()

        separate = [
            trainer._encode_graph_snapshot(graph)
            for graph in (first_graph, second_graph)
        ]
        batched = trainer._encode_graph_snapshots(
            (first_graph, second_graph)
        )

        for node_type, graph_outputs in batched.items():
            for graph_index, graph_output in enumerate(graph_outputs):
                assert torch.allclose(
                    graph_output,
                    separate[graph_index][node_type],
                    atol=1e-5,
                    rtol=1e-5,
                )
    finally:
        env.close()


def _elevations(env):
    vectors = env.constellation._all_pos_ecef - env._user_pos_ecef[0][None, :]
    up = vectors @ env._user_e_up[0]
    east = vectors @ env._user_e_east[0]
    north = vectors @ env._user_e_north[0]
    return np.degrees(np.arctan2(up, np.sqrt(east**2 + north**2)))


def test_descending_rvt_decreases_toward_visibility_horizon():
    env = LEOSatelliteEnv(
        EnvConfig(
            num_users=1,
            randomize_episode_start=False,
            seed=7,
        )
    )
    try:
        env.reset(seed=7)
        period = int(env.constellation.orbital_period)
        selected = None
        for offset in range(2, period, 20):
            env.constellation.reset(time_offset_sec=offset - 1)
            previous = _elevations(env)
            env.constellation.reset(time_offset_sec=offset)
            current = _elevations(env)
            env.constellation.reset(time_offset_sec=offset + 1)
            following = _elevations(env)
            candidates = np.where(
                (previous > current)
                & (current > following)
                & (following > env.config.min_elevation_deg + 1.0)
            )[0]
            if candidates.size:
                selected = (
                    offset,
                    int(candidates[0]),
                    float(current[candidates[0]]),
                    float(following[candidates[0]]),
                )
                break

        assert selected is not None
        offset, sat_id, current_elev, following_elev = selected
        user = env.user_manager.users[0]
        env.constellation.reset(time_offset_sec=offset)
        current_rvt = env._estimate_rvt(user, sat_id, current_elev)
        env.constellation.reset(time_offset_sec=offset + 1)
        following_rvt = env._estimate_rvt(user, sat_id, following_elev)

        assert following_rvt < current_rvt
    finally:
        env.close()


def test_propagation_advances_geometry_version_and_refreshes_visibility():
    env = LEOSatelliteEnv(
        EnvConfig(
            num_users=1,
            max_steps=5,
            randomize_episode_start=False,
            seed=7,
        )
    )
    try:
        env.reset(seed=7)
        env._get_visible_satellites(env.user_manager.users[0])
        version_before = env.geometry_version

        env._update_environment()

        assert env.geometry_version == version_before + 1
        assert env._visibility_cache_version == env.geometry_version
    finally:
        env.close()


def test_action_index_targets_satellite_from_observed_candidate_mapping():
    env = LEOSatelliteEnv(
        EnvConfig(
            num_users=1,
            max_steps=5,
            randomize_episode_start=False,
            seed=7,
        )
    )
    try:
        env.reset(seed=7)
        user = env.user_manager.users[0]
        observed_candidates = env._get_handover_candidates(user)
        assert observed_candidates
        expected_satellite = observed_candidates[0].sat_id
        captured = {}

        def capture_handover(_user, target):
            captured["satellite_id"] = target.sat_id
            return 0.0

        env._execute_handover = capture_handover
        env.step(np.asarray([[1.0, 0.0]], dtype=np.float32))

        assert captured["satellite_id"] == expected_satellite
    finally:
        env.close()


def test_han_parameters_change_after_ppo_update():
    run_id = uuid4().hex
    save_path = f"results/p0_han_update_{run_id}"
    log_path = f"results/p0_han_update_logs_{run_id}"
    trainer = HANMAPPOTrainer(
        TrainConfig(
            num_users=2,
            max_steps=8,
            total_timesteps=4,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            eval_episodes=0,
            device="cpu",
            save_path=save_path,
            log_path=log_path,
        )
    )
    try:
        before = {
            name: parameter.detach().clone()
            for name, parameter in trainer.han_encoder.named_parameters()
        }

        trainer.collect_rollouts()
        stats = trainer.mappo.update()

        changed = any(
            not torch.equal(before[name], parameter.detach())
            for name, parameter in trainer.han_encoder.named_parameters()
        )
        assert changed
        assert stats["han_grad_norm"] > 0.0
        assert stats["han_parameter_delta"] > 0.0
        assert stats["han_metapaths_executed"] == (
            trainer.han_encoder.config.num_layers
            * len(trainer.han_encoder.config.metapaths)
        )

        trainer._save_checkpoint(final=True)
        checkpoint_path = f"{save_path}/final_model.pt"
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["model_schema_version"] == MODEL_SCHEMA_VERSION
        assert checkpoint["geometry_schema_version"] == 3
        assert checkpoint["environment_schema_version"] == ENVIRONMENT_SCHEMA_VERSION
        assert checkpoint["han_optimizer_state_dict"] is not None
        trainer.load_checkpoint(checkpoint_path)
    finally:
        trainer.env.close()
        for handler in trainer.logger.handlers:
            handler.close()
        trainer.logger.handlers.clear()
        shutil.rmtree(save_path, ignore_errors=True)
        shutil.rmtree(log_path, ignore_errors=True)
