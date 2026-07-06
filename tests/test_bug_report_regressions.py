from types import SimpleNamespace

import numpy as np
import pytest

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
