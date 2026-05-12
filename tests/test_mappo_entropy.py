from src.algorithm.mappo import MAPPO, MAPPOConfig


def _build_mappo(entropy_schedule: str) -> MAPPO:
    return MAPPO(
        MAPPOConfig(
            num_agents=1,
            obs_dim=8,
            global_state_dim=8,
            max_candidates=2,
            sat_embed_dim=4,
            actor_hidden_dims=[8],
            critic_hidden_dims=[8],
            entropy_coef=0.005,
            entropy_schedule=entropy_schedule,
            device="cpu",
        )
    )


def test_constant_entropy_schedule_does_not_decay_late_in_training():
    mappo = _build_mappo("constant")
    mappo.train_step = 600

    assert mappo._current_entropy_coef() == 0.005


def test_linear_entropy_schedule_remains_available_for_ablation():
    mappo = _build_mappo("linear")
    mappo.train_step = 600

    assert mappo._current_entropy_coef() == 0.0005
