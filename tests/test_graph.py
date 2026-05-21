import numpy as np
import pytest

from src.environment.gym_env import EnvConfig, LEOSatelliteEnv
from src.graph import HeteroGraphBuilder, FeatureExtractor, NodeFeatures, EdgeFeatures
from src.graph.builder import HeteroGraphData


def test_feature_extractor_defaults():
    extractor = FeatureExtractor()

    assert extractor.normalize is True
    assert extractor.include_velocity is True
    assert extractor.include_load is True


def test_node_and_edge_feature_containers():
    node_features = NodeFeatures(
        satellite_features=np.zeros((66, 10), dtype=np.float32),
        user_features=np.zeros((5, 13), dtype=np.float32),
        satellite_ids=list(range(66)),
        user_ids=list(range(5)),
    )
    edge_features = EdgeFeatures(
        user_satellite_edges=[(0, 1), (1, 2)],
        user_satellite_features=np.zeros((2, 6), dtype=np.float32),
        inter_satellite_edges=[(0, 1)],
        inter_satellite_features=np.zeros((1, 3), dtype=np.float32),
    )

    assert node_features.satellite_features.shape == (66, 10)
    assert node_features.user_features.shape == (5, 13)
    assert edge_features.user_satellite_features.shape == (2, 6)
    assert edge_features.inter_satellite_features.shape == (1, 3)


def test_graph_builder_configuration_and_metapaths():
    builder = HeteroGraphBuilder()

    assert builder.add_reverse_edges is True
    assert builder.add_self_loops is False
    assert all(len(edge) == 3 for metapath in builder.get_metapaths() for edge in metapath)


def test_graph_data_counts_and_serializes():
    graph = HeteroGraphData()
    graph.node_features["satellite"] = np.zeros((10, 5), dtype=np.float32)
    graph.node_features["user"] = np.zeros((2, 3), dtype=np.float32)
    graph.num_nodes["satellite"] = 10
    graph.num_nodes["user"] = 2
    graph.edge_index[("user", "connect", "satellite")] = (
        np.array([0, 1]),
        np.array([3, 4]),
    )
    graph.edge_features[("user", "connect", "satellite")] = np.ones((2, 6), dtype=np.float32)
    graph.metadata["scenario"] = "unit"

    payload = graph.to_dict()

    assert graph.get_node_types() == ["satellite", "user"]
    assert graph.num_edges() == 2
    assert graph.num_edges(("user", "connect", "satellite")) == 2
    assert payload["metadata"]["scenario"] == "unit"
    assert "('user', 'connect', 'satellite')" in payload["edge_index"]


@pytest.fixture
def env():
    instance = LEOSatelliteEnv(
        EnvConfig(
            num_users=3,
            max_steps=12,
            time_step_sec=5.0,
            seed=42,
        )
    )
    try:
        instance.reset(seed=42)
        yield instance
    finally:
        instance.close()


def test_graph_builder_integrates_with_current_environment(env):
    graph = HeteroGraphBuilder().build(env)

    assert graph.node_features["satellite"].shape[0] == env.num_satellites
    assert graph.node_features["user"].shape[0] == env.num_users
    assert graph.num_nodes["satellite"] == env.num_satellites
    assert graph.num_nodes["user"] == env.num_users
    assert graph.metadata["num_satellites"] == env.num_satellites
    assert graph.metadata["num_users"] == env.num_users


def test_feature_extractor_integrates_with_current_environment(env):
    extractor = FeatureExtractor()

    node_features = extractor.extract_node_features(env)
    edge_features = extractor.extract_edge_features(env)

    assert node_features.satellite_features.shape == (env.num_satellites, 10)
    assert node_features.user_features.shape == (env.num_users, 13)
    assert len(edge_features.inter_satellite_edges) > 0
    assert edge_features.inter_satellite_features.shape[1] == 3
    if edge_features.user_satellite_edges:
        assert edge_features.user_satellite_features.shape[1] == 6


def test_graph_summary_prints_core_sections(env, capsys):
    builder = HeteroGraphBuilder()
    graph = builder.build(env)

    builder.print_graph_summary(graph)
    captured = capsys.readouterr()

    assert "satellite" in captured.out
    assert "user" in captured.out
