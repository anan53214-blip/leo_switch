import numpy as np
import pytest

from scripts.train import TrainConfig, compute_model_selection_score
from src.environment.gym_env import EnvConfig, LEOSatelliteEnv


def test_effective_latency_score_fallback_uses_task_success_rate():
    record = {
        "avg_delay": 1.0,
        "service_continuity_rate": 0.8,
        "task_success_rate": 0.5,
        "task_resolution_rate": 1.0,
        "task_completion_rate": 0.9,
    }

    assert compute_model_selection_score(record, "effective_latency_score") == pytest.approx(0.2)


def test_task_success_rate_can_be_used_as_selection_metric():
    record = {"task_success_rate": 0.73}

    assert compute_model_selection_score(record, "task_success_rate") == pytest.approx(0.73)


def _build_single_user_env(**overrides) -> LEOSatelliteEnv:
    params = {
        "num_users": 1,
        "max_steps": 5,
        "seed": 7,
        "task_arrival_prob": 0.0,
    }
    params.update(overrides)
    config = EnvConfig(**params)
    return LEOSatelliteEnv(config)


class _DeterministicRng:
    def random(self):
        return 0.0


def test_handover_weight_changes_reward_signal():
    env_low = _build_single_user_env(reward_handover_weight=0.0)
    env_high = _build_single_user_env(reward_handover_weight=1.0)

    try:
        env_low.reset(seed=7)
        env_high.reset(seed=7)

        user_low = env_low.user_manager.users[0]
        visible_low = env_low._get_visible_satellites(user_low)
        action_index = next(
            (
                i + 1
                for i, sat in enumerate(visible_low)
                if sat.sat_id != user_low.serving_satellite
            ),
            None,
        )
        assert action_index is not None

        env_low.rng = _DeterministicRng()
        env_high.rng = _DeterministicRng()
        actions = np.array([[float(action_index), 0.0]], dtype=np.float32)
        _, reward_low, *_ = env_low.step(actions)
        _, reward_high, *_ = env_high.step(actions)

        assert reward_high != reward_low
    finally:
        env_low.close()
        env_high.close()


def test_pending_offload_reward_uses_split_task_total_delay():
    env = _build_single_user_env()

    try:
        env.reset(seed=7)
        env._offload_task_meta[(0, 99)] = {
            'local_delay': 2.0,
            'local_energy': 0.4,
        }

        env.mec_manager.process_all_queues = lambda current_time, time_step: [
            {
                'user_id': 0,
                'task_id': 99,
                'total_delay': 0.5,
                'max_delay': 1.0,
                'upload_energy': 0.1,
                'deadline_met': True,
            }
        ]

        env._update_environment()

        assert env.stats['completed_tasks'] == 0
        assert env.stats['deadline_violations'] == 1
        assert env.stats['total_delay'] == 2.0
        assert env.pending_rewards[0] < 0.0
    finally:
        env.close()


def test_info_contains_reward_breakdown_and_load_balance():
    env = _build_single_user_env()

    try:
        env.reset(seed=7)
        _, reward, _, _, info = env.step(np.array([[0.0, 0.0]], dtype=np.float32))

        assert isinstance(reward, float)
        assert 'load_balance_score' in info
        for key in [
            'reward_delay',
            'reward_energy',
            'reward_qos',
            'reward_service_continuity',
            'reward_handover',
            'reward_load_balance',
            'reward_enqueue',
            'penalty_deadline',
            'penalty_queue_full',
        ]:
            assert key in info['stats']
    finally:
        env.close()


def test_service_continuity_reward_penalizes_only_interruptions():
    env = _build_single_user_env(reward_service_continuity_weight=0.15)

    try:
        assert env._compute_service_continuity_reward(10.0, 0.0) == pytest.approx(0.0)
        assert env._compute_service_continuity_reward(10.0, 2.0) == pytest.approx(-0.03)
        assert env._compute_service_continuity_reward(10.0, 10.0) == pytest.approx(-0.15)
    finally:
        env.close()


def test_service_continuity_term_stays_bounded_over_long_episodes():
    env = _build_single_user_env(reward_service_continuity_weight=0.15, max_steps=2000)

    try:
        per_step = env._compute_service_continuity_reward(20.0, 0.24)
        assert per_step == pytest.approx(-0.0018)
        assert per_step * 2000 == pytest.approx(-3.6)
    finally:
        env.close()


def test_training_defaults_use_balanced_update_budget():
    config = TrainConfig()

    assert config.n_epochs == 6
    assert config.batch_size == 256
    assert config.entropy_schedule == "constant"
    assert config.reward_failed_handover_penalty == pytest.approx(0.3)


def test_reward_default_weights_are_balanced():
    env_config = EnvConfig()
    train_config = TrainConfig()

    expected = {
        "reward_delay_weight": 0.25,
        "reward_energy_weight": 0.15,
        "reward_handover_weight": 0.10,
        "reward_load_balance_weight": 0.05,
        "reward_qos_weight": 0.30,
        "reward_service_continuity_weight": 0.15,
        "reward_deadline_penalty": 0.30,
    }

    for key, value in expected.items():
        assert getattr(env_config, key) == pytest.approx(value)
        assert getattr(train_config, key) == pytest.approx(value)


def test_server_training_defaults_use_balanced_reward_weights():
    from scripts.run_server_training import STANDARD_CONFIG, build_training_config

    config = build_training_config(STANDARD_CONFIG, algorithm="mappo")

    assert config.reward_delay_weight == pytest.approx(0.25)
    assert config.reward_energy_weight == pytest.approx(0.15)
    assert config.reward_handover_weight == pytest.approx(0.10)
    assert config.reward_load_balance_weight == pytest.approx(0.05)
    assert config.reward_qos_weight == pytest.approx(0.30)
    assert config.reward_service_continuity_weight == pytest.approx(0.15)
    assert config.reward_deadline_penalty == pytest.approx(0.30)
