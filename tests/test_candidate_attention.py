import torch


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


def test_satellite_load_encoder_accepts_global_satellite_tokens():
    from src.model.candidate_attention import SatelliteLoadEncoder

    encoder = SatelliteLoadEncoder(
        sat_feature_dim=8,
        hidden_dim=32,
        num_heads=4,
    )

    encoded = encoder(torch.randn(5, 8))

    assert encoded.shape == (5, 32)
