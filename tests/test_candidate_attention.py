from types import SimpleNamespace

import numpy as np
import torch


class _FakeServer:
    def __init__(
        self,
        *,
        queue_length=0,
        max_queue_size=6,
        wait_time=0.0,
        utilization=0.0,
        available_freq_ghz=20.0,
        total_capacity_ghz=20.0,
        connected_users=None,
    ):
        self.queue_length = queue_length
        self.config = SimpleNamespace(max_queue_size=max_queue_size)
        self._wait_time = wait_time
        self.utilization = utilization
        self.available_freq_ghz = available_freq_ghz
        self.total_capacity_ghz = total_capacity_ghz
        self.connected_users = list(connected_users or [])
        self.is_full = queue_length >= max_queue_size

    def get_estimated_wait_time(self):
        return self._wait_time


class _FakeMECManager:
    def __init__(self, servers):
        self.servers = servers

    def get_server(self, sid):
        return self.servers.get(sid)


class _FakeChannel:
    def compute_snr_db(self, distance_km, elevation_deg):
        return 25.0


def _make_cpq_feature_env(*, wait_time=0.0, task_creation=0.0, rvt_seconds=30.0):
    users = [
        SimpleNamespace(user_id=0, serving_satellite=0),
        SimpleNamespace(user_id=1, serving_satellite=1),
    ]
    visible = {
        0: [
            SimpleNamespace(
                sat_id=0,
                distance_km=800.0,
                elevation_deg=45.0,
                rvt_seconds=rvt_seconds,
                is_visible=True,
            )
        ],
        1: [
            SimpleNamespace(
                sat_id=1,
                distance_km=900.0,
                elevation_deg=50.0,
                rvt_seconds=90.0,
                is_visible=True,
            )
        ],
    }
    servers = {
        0: _FakeServer(
            queue_length=3,
            wait_time=wait_time,
            utilization=0.6,
            available_freq_ghz=8.0,
            total_capacity_ghz=20.0,
            connected_users=[0],
        ),
        1: _FakeServer(
            queue_length=1,
            wait_time=1.0,
            utilization=0.2,
            available_freq_ghz=16.0,
            total_capacity_ghz=20.0,
            connected_users=[1],
        ),
    }
    env = SimpleNamespace(
        num_satellites=2,
        num_users=2,
        current_time=8.0,
        config=SimpleNamespace(pre_handover_rvt_sec=60.0),
        user_manager=SimpleNamespace(users=users),
        user_tasks={
            0: SimpleNamespace(
                data_size=20e6,
                computation=2e9,
                max_delay=10.0,
                creation_time=task_creation,
                task_type=SimpleNamespace(value=1),
            )
        },
        mec_manager=_FakeMECManager(servers),
        channel=_FakeChannel(),
    )
    env._get_visible_satellites = lambda user: visible[user.user_id]
    env._get_satellite_visibility = lambda user, sid: next(
        (vis for vis in visible[user.user_id] if vis.sat_id == sid),
        None,
    )
    return env


def test_candidate_attention_actor_masks_invalid_candidates_and_samples_shapes():
    from src.model.candidate_attention import (
        CandidateAttentionActor,
        CandidateAttentionConfig,
    )

    max_candidates = 2
    obs_dim = 3 + 1 + 5 + max_candidates * 6 + 4
    actor = CandidateAttentionActor(
        CandidateAttentionConfig(
            user_obs_dim=obs_dim,
            sat_feature_dim=8,
            hidden_dim=32,
            num_heads=4,
            max_candidates=max_candidates,
        )
    )

    observations = torch.zeros(2, obs_dim)
    candidate_masks = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    satellite_features = torch.randn(4, 8)
    candidate_sat_ids = torch.tensor([[1, 2], [3, -1]], dtype=torch.long)

    logits = actor.handover_logits(
        observations,
        candidate_masks,
        satellite_features,
        candidate_sat_ids,
    )
    assert logits.shape == (2, max_candidates + 1)
    assert logits[1, 2].item() < -1e8

    actions = actor.sample_all(
        observations,
        candidate_masks,
        satellite_features=satellite_features,
        candidate_sat_ids=candidate_sat_ids,
    )
    assert actions["handover"].shape == (2,)
    assert actions["offload"].shape == (2,)
    assert actions["log_prob"].shape == (2,)


def test_candidate_attention_logits_change_when_satellite_load_changes():
    from src.model.candidate_attention import (
        CandidateAttentionActor,
        CandidateAttentionConfig,
    )

    max_candidates = 2
    obs_dim = 3 + 1 + 5 + max_candidates * 6 + 4
    actor = CandidateAttentionActor(
        CandidateAttentionConfig(
            user_obs_dim=obs_dim,
            sat_feature_dim=8,
            hidden_dim=32,
            num_heads=4,
            max_candidates=max_candidates,
        )
    )

    observations = torch.zeros(1, obs_dim)
    candidate_masks = torch.ones(1, max_candidates + 1)
    candidate_sat_ids = torch.tensor([[0, 1]], dtype=torch.long)

    low_load = torch.zeros(3, 8)
    high_load = low_load.clone()
    high_load[1, 1] = 1.0
    high_load[1, 2] = 1.0
    high_load[1, 3] = 1.0

    logits_low = actor.handover_logits(
        observations,
        candidate_masks,
        low_load,
        candidate_sat_ids,
    )
    logits_high = actor.handover_logits(
        observations,
        candidate_masks,
        high_load,
        candidate_sat_ids,
    )

    assert not torch.allclose(logits_low, logits_high)


def test_candidate_attention_offload_distribution_is_conditioned_per_handover_action():
    from src.model.candidate_attention import (
        CandidateAttentionActor,
        CandidateAttentionConfig,
    )

    max_candidates = 2
    obs_dim = 3 + 1 + 5 + max_candidates * 6 + 4
    actor = CandidateAttentionActor(
        CandidateAttentionConfig(
            user_obs_dim=obs_dim,
            sat_feature_dim=8,
            hidden_dim=32,
            num_heads=4,
            max_candidates=max_candidates,
            dropout=0.0,
        )
    )
    actor.eval()

    observations = torch.zeros(1, obs_dim)
    candidate_masks = torch.ones(1, max_candidates + 1)
    satellite_features = torch.randn(3, 8)
    candidate_sat_ids = torch.tensor([[0, 1]], dtype=torch.long)

    _, offload_dist = actor.forward(
        observations,
        candidate_masks,
        satellite_features,
        candidate_sat_ids,
    )

    assert offload_dist.mean.shape == (1, max_candidates + 1)


def test_satellite_load_encoder_accepts_global_satellite_tokens():
    from src.model.candidate_attention import SatelliteLoadEncoder

    encoder = SatelliteLoadEncoder(
        sat_feature_dim=8,
        hidden_dim=32,
        num_heads=4,
    )

    encoded = encoder(torch.randn(5, 8))

    assert encoded.shape == (5, 32)


def test_satellite_context_features_include_queue_wait_deadline_and_rvt_risk():
    from src.features.satellite_load import (
        SATELLITE_CONTEXT_FEATURE_DIM,
        SATELLITE_RISK_FEATURE_SLICE,
        build_satellite_context_features,
    )

    low_risk_env = _make_cpq_feature_env(
        wait_time=1.0,
        task_creation=7.5,
        rvt_seconds=55.0,
    )
    high_risk_env = _make_cpq_feature_env(
        wait_time=8.0,
        task_creation=0.0,
        rvt_seconds=5.0,
    )

    low_features = build_satellite_context_features(low_risk_env, num_agents=2)
    high_features = build_satellite_context_features(high_risk_env, num_agents=2)

    assert low_features.shape == (2, SATELLITE_CONTEXT_FEATURE_DIM)
    assert high_features.shape == (2, SATELLITE_CONTEXT_FEATURE_DIM)
    assert np.all((high_features[:, SATELLITE_RISK_FEATURE_SLICE] >= 0.0))
    assert np.all((high_features[:, SATELLITE_RISK_FEATURE_SLICE] <= 1.0))
    assert high_features[0, 8] > low_features[0, 8]
    assert high_features[0, 10] > low_features[0, 10]
    assert high_features[0, 11] > low_features[0, 11]


def test_shared_constraint_vector_reflects_global_queue_pressure():
    from src.features.satellite_load import (
        SHARED_CONSTRAINT_DIM,
        build_shared_constraint_vector,
    )

    low_pressure = build_shared_constraint_vector(
        _make_cpq_feature_env(wait_time=0.5),
        num_agents=2,
    )
    high_pressure_env = _make_cpq_feature_env(wait_time=9.0)
    high_pressure_env.mec_manager.servers[0].queue_length = 6
    high_pressure_env.mec_manager.servers[0].is_full = True
    high_pressure_env.mec_manager.servers[0].utilization = 0.95
    high_pressure = build_shared_constraint_vector(high_pressure_env, num_agents=2)

    assert low_pressure.shape == (SHARED_CONSTRAINT_DIM,)
    assert high_pressure.shape == (SHARED_CONSTRAINT_DIM,)
    assert np.all((high_pressure >= 0.0) & (high_pressure <= 1.0))
    assert high_pressure[1] > low_pressure[1]
    assert high_pressure[2] > low_pressure[2]
    assert high_pressure[3] > low_pressure[3]


def test_candidate_attention_logits_change_when_risk_features_change():
    from src.model.candidate_attention import (
        CandidateAttentionActor,
        CandidateAttentionConfig,
    )

    max_candidates = 2
    obs_dim = 3 + 1 + 5 + max_candidates * 6 + 4
    actor = CandidateAttentionActor(
        CandidateAttentionConfig(
            user_obs_dim=obs_dim,
            sat_feature_dim=13,
            hidden_dim=32,
            num_heads=4,
            max_candidates=max_candidates,
            risk_feature_start=8,
            risk_feature_dim=5,
            dropout=0.0,
        )
    )
    actor.eval()

    observations = torch.zeros(1, obs_dim)
    candidate_masks = torch.ones(1, max_candidates + 1)
    candidate_sat_ids = torch.tensor([[0, 1]], dtype=torch.long)
    low_risk = torch.zeros(3, 13)
    high_risk = low_risk.clone()
    high_risk[1, 8:13] = torch.tensor([0.9, 0.1, 0.8, 0.9, 1.0])

    logits_low = actor.handover_logits(
        observations,
        candidate_masks,
        low_risk,
        candidate_sat_ids,
    )
    logits_high = actor.handover_logits(
        observations,
        candidate_masks,
        high_risk,
        candidate_sat_ids,
    )

    assert not torch.allclose(logits_low, logits_high)
