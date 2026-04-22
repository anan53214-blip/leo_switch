import shutil
from uuid import uuid4
import numpy as np
import pytest

from scripts.train import HANMAPPOTrainer, TrainConfig


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
