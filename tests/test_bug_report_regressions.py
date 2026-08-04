from types import SimpleNamespace

import numpy as np
import pytest

from src.environment.gym_env import EnvConfig, LEOSatelliteEnv
from src.environment.mec import ComputeResult, OffloadingCalculator
from src.graph.features import FeatureExtractor


class _WeightedObjectiveCalculator(OffloadingCalculator):
    def compute_offloading_result(self, *args, offload_ratio, max_delay, **kwargs):
        if offload_ratio == pytest.approx(0.0):
            return ComputeResult(
                total_delay=0.1,
                total_energy=1000.0,
                deadline_met=True,
            )
        return ComputeResult(
            total_delay=1.9,
            total_energy=990.0,
            deadline_met=True,
        )


def test_weighted_offload_objective_normalizes_energy_scale():
    calc = _WeightedObjectiveCalculator()

    best_ratio, best_result = calc.find_optimal_offload_ratio(
        data_size_bits=1.0,
        computation_cycles=1.0,
        max_delay=2.0,
        distance_km=1000.0,
        elevation_deg=45.0,
        objective="weighted",
        num_samples=2,
    )

    assert best_ratio == pytest.approx(0.0)
    assert best_result.total_delay == pytest.approx(0.1)


def test_position_feature_normalization_uses_environment_orbit_radius():
    extractor = FeatureExtractor(include_velocity=False)
    radius_km = 8000.0
    server = SimpleNamespace(
        utilization=0.0,
        queue_length=0,
        connected_users=[],
        available_freq_ghz=1.0,
        config=SimpleNamespace(satellite_max_cpu_freq_ghz=1.0),
    )
    user = SimpleNamespace(
        state=SimpleNamespace(value=0),
        serving_satellite=-1,
        total_service_time=0.0,
        handover_count=0,
        successful_handovers=0,
        service_quality=1.0,
    )
    env = SimpleNamespace(
        num_satellites=1,
        num_users=1,
        constellation=SimpleNamespace(
            semi_major_axis=radius_km,
            _all_pos_ecef=np.array([[radius_km, 0.0, 0.0]], dtype=np.float32),
            _all_vel_eci=np.zeros((1, 3), dtype=np.float32),
        ),
        mec_manager=SimpleNamespace(get_server=lambda sat_id: server),
        user_manager=SimpleNamespace(users=[user]),
        _user_pos_ecef=np.array([[radius_km, 0.0, 0.0]], dtype=np.float32),
        user_tasks={},
        config=SimpleNamespace(rvt_threshold_sec=60.0),
        _get_satellite_visibility=lambda user, sat_id: None,
    )

    sat_features = extractor._extract_satellite_features(env)
    user_features = extractor._extract_user_features(env)

    assert sat_features[0, 0] == pytest.approx(1.0)
    assert user_features[0, 0] == pytest.approx(1.0)


def test_satellite_mec_features_use_real_capacity_and_queue_limits():
    extractor = FeatureExtractor(include_velocity=False)
    server = SimpleNamespace(
        utilization=0.25,
        queue_length=3,
        connected_users=[0],
        available_freq_ghz=10.0,
        total_capacity_ghz=20.0,
        config=SimpleNamespace(
            max_queue_size=6,
            satellite_max_cpu_freq_ghz=8.0,
            satellite_num_cores=4,
        ),
    )
    env = SimpleNamespace(
        num_satellites=1,
        num_users=2,
        constellation=SimpleNamespace(
            semi_major_axis=7000.0,
            _all_pos_ecef=np.array([[7000.0, 0.0, 0.0]], dtype=np.float32),
            _all_vel_eci=np.zeros((1, 3), dtype=np.float32),
        ),
        mec_manager=SimpleNamespace(get_server=lambda sat_id: server),
    )

    features = extractor._extract_satellite_features(env)
    mec_start = 3

    assert features[0, mec_start] == pytest.approx(0.25)
    assert features[0, mec_start + 1] == pytest.approx(0.5)
    assert features[0, mec_start + 2] == pytest.approx(0.5)
    assert features[0, mec_start + 3] == pytest.approx(0.5)


def test_walker_isl_topology_keeps_all_wraparound_edges():
    env = LEOSatelliteEnv(
        EnvConfig(
            num_users=1,
            num_planes=6,
            sats_per_plane=11,
            task_arrival_prob=0.0,
            randomize_episode_start=False,
            seed=29,
        )
    )
    try:
        extractor = FeatureExtractor()
        edges, _ = extractor._extract_inter_satellite_edges(env)
        edge_set = set(edges)

        assert len(edges) == 2 * env.num_satellites
        assert (0, 10) in edge_set
        assert (0, 55) in edge_set
    finally:
        env.close()


def test_reset_observation_contains_task_used_by_first_action():
    env = LEOSatelliteEnv(
        EnvConfig(
            num_users=1,
            max_steps=2,
            task_arrival_prob=1.0,
            randomize_episode_start=False,
            seed=17,
        )
    )
    try:
        observation, _ = env.reset(seed=17)
        task_start = env.user_obs_dim - 7
        observed_task = observation[0, task_start:].copy()
        task_id = env.user_tasks[0].task_id

        assert np.any(observed_task != 0.0)
        assert env.stats["total_tasks"] == 1

        env.step(np.asarray([[0.0, 0.0]], dtype=np.float32))

        assert all(
            task.task_id != task_id
            for task in env.user_task_queues[0]
        )
    finally:
        env.close()


def test_step_generates_next_task_before_returning_next_observation():
    env = LEOSatelliteEnv(
        EnvConfig(
            num_users=1,
            max_steps=3,
            task_arrival_prob=1.0,
            randomize_episode_start=False,
            seed=23,
        )
    )
    try:
        env.reset(seed=23)
        next_observation, _, _, truncated, _ = env.step(
            np.asarray([[0.0, 0.0]], dtype=np.float32)
        )
        task_start = env.user_obs_dim - 7

        assert not truncated
        assert env.user_tasks[0] is not None
        assert np.any(next_observation[0, task_start:] != 0.0)
        assert env.stats["total_tasks"] == 2
    finally:
        env.close()


def test_blocked_user_can_still_complete_task_locally():
    env = LEOSatelliteEnv(
        EnvConfig(
            num_users=1,
            max_steps=2,
            task_arrival_prob=1.0,
            randomize_episode_start=False,
            seed=31,
        )
    )
    try:
        env.reset(seed=31)
        user = env.user_manager.users[0]
        env._block_user(user, user.serving_satellite)
        task_id = env.user_tasks[0].task_id

        env.step(np.asarray([[0.0, 1.0]], dtype=np.float32))

        assert all(task.task_id != task_id for task in env.user_task_queues[0])
        assert env.stats["completed_tasks"] + env.stats["deadline_violations"] >= 1
    finally:
        env.close()


def test_public_handover_mask_contains_only_feasible_targets():
    env = LEOSatelliteEnv(
        EnvConfig(
            num_users=3,
            task_arrival_prob=0.0,
            randomize_episode_start=False,
            seed=37,
        )
    )
    try:
        env.reset(seed=37)
        mask = env.get_handover_action_mask(apply_pre_handover_gate=False)
        for user_id, user in enumerate(env.user_manager.users):
            candidates = env._get_handover_candidates(user)
            for action in np.flatnonzero(mask[user_id, 1:]) + 1:
                feasible, _ = env._check_handover_link_feasibility(
                    candidates[action - 1]
                )
                assert feasible
    finally:
        env.close()
