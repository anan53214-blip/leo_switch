import numpy as np
import pytest

from scripts.train import TrainConfig, compute_model_selection_score
from src.environment.gym_env import (
    EnvConfig,
    LEOSatelliteEnv,
    REWARD_BREAKDOWN_KEYS,
    REWARD_ENERGY_REFERENCE_J,
    TASK_FAILURE_PENALTY,
    TASK_SUCCESS_REWARD,
)
from src.environment.channel import ChannelConfig
from src.environment.mec import MECConfig, OffloadingCalculator
from src.environment.task import Task
from src.environment.user import UserState


def test_custom_composite_scores_are_not_model_selection_metrics():
    from scripts.train import BEST_MODEL_METRIC_CHOICES

    assert "effective_latency_score" not in BEST_MODEL_METRIC_CHOICES
    assert "latency_priority_score" not in BEST_MODEL_METRIC_CHOICES


def test_task_success_rate_can_be_used_as_selection_metric():
    record = {"task_success_rate": 0.73}

    assert compute_model_selection_score(record, "task_success_rate") == pytest.approx(0.73)


def test_default_compute_capacity_matches_confirmed_environment():
    config = MECConfig()

    assert config.user_cpu_freq_ghz == pytest.approx(1.0)
    assert config.satellite_cpu_freq_ghz == pytest.approx(5.0)
    assert config.satellite_num_cores == 2
    assert config.mec_max_concurrent_tasks == 2


def test_upload_energy_uses_terminal_battery_draw_and_excludes_propagation():
    channel_config = ChannelConfig(
        bandwidth_mhz=10.0,
        user_tx_power_dbm=30.0,
        user_pa_efficiency=0.5,
        user_circuit_power_w=0.25,
    )
    calculator = OffloadingCalculator(
        mec_config=MECConfig(),
        channel_config=channel_config,
    )
    data_size_bits = 2e6
    bandwidth_mhz = 5.0
    distance_km = 900.0
    elevation_deg = 60.0

    rate_bps = calculator.channel.compute_channel_capacity(
        distance_km,
        elevation_deg,
        "uplink",
        bandwidth_mhz,
    )
    expected_airtime = data_size_bits / rate_bps
    expected_battery_power = 1.0 / 0.5 + 0.25
    upload_delay, _ = calculator.compute_transmission_delay(
        data_size_bits,
        distance_km,
        elevation_deg,
        bandwidth_mhz=bandwidth_mhz,
    )
    energy = calculator.compute_transmission_energy(
        data_size_bits,
        distance_km,
        elevation_deg,
        bandwidth_mhz=bandwidth_mhz,
    )

    assert energy == pytest.approx(expected_battery_power * expected_airtime)
    assert upload_delay > expected_airtime
    assert energy < expected_battery_power * upload_delay


def test_ofdma_narrower_uplink_bandwidth_increases_delay_and_terminal_energy():
    calculator = OffloadingCalculator(
        mec_config=MECConfig(),
        channel_config=ChannelConfig(
            bandwidth_mhz=10.0,
            user_pa_efficiency=0.38,
            user_circuit_power_w=0.05,
        ),
    )
    arguments = (8e6, 800.0, 45.0)

    full_delay, _ = calculator.compute_transmission_delay(
        *arguments,
        bandwidth_mhz=10.0,
    )
    shared_delay, _ = calculator.compute_transmission_delay(
        *arguments,
        bandwidth_mhz=5.0,
    )
    full_energy = calculator.compute_transmission_energy(
        *arguments,
        bandwidth_mhz=10.0,
    )
    shared_energy = calculator.compute_transmission_energy(
        *arguments,
        bandwidth_mhz=5.0,
    )

    assert shared_delay > full_delay
    assert shared_energy > full_energy


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


def test_same_satellite_offloaders_receive_equal_ofdma_uplink_shares():
    env = _build_single_user_env(
        num_users=2,
        bandwidth_mhz=10.0,
        ofdma_uplink_sharing=True,
    )

    try:
        env.reset(seed=7)
        sat_id = 0
        server = env.mec_manager.get_server(sat_id)
        assert server is not None
        server.task_queue.clear()
        for user_id, user in enumerate(env.user_manager.users):
            user.serving_satellite = sat_id
            user.state = UserState.CONNECTED
            task = Task(
                task_id=100 + user_id,
                user_id=user_id,
                data_size=5e6,
                computation=1e9,
                max_delay=5.0,
                creation_time=env.current_time,
            )
            env.user_task_queues[user_id] = [task]
            env.user_tasks[user_id] = task

        allocations = env._plan_ofdma_uplink_allocations(
            np.asarray([1.0, 0.5], dtype=np.float32)
        )

        assert allocations[0].admission_allowed
        assert allocations[1].admission_allowed
        assert allocations[0].concurrent_users == 2
        assert allocations[1].concurrent_users == 2
        assert allocations[0].bandwidth_mhz == pytest.approx(5.0)
        assert allocations[1].bandwidth_mhz == pytest.approx(5.0)
    finally:
        env.close()


def test_interruption_weight_changes_successful_handover_reward():
    env_low = _build_single_user_env(reward_interruption_weight=0.0)
    env_high = _build_single_user_env(reward_interruption_weight=1.0)

    try:
        env_low.reset(seed=7)
        env_high.reset(seed=7)

        user_low = env_low.user_manager.users[0]
        handover_candidates = env_low._get_handover_candidates(user_low)
        assert handover_candidates
        action_index = 1

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
        for key in REWARD_BREAKDOWN_KEYS:
            assert key in info['stats']
    finally:
        env.close()


def test_reward_breakdown_uses_same_user_mean_scale_as_global_reward():
    env = _build_single_user_env(num_users=2)

    try:
        env._record_reward_terms(reward_task_success=1.0, penalty_delay=-0.4)
        env._record_reward_terms(
            average_over_users=False,
            penalty_service_interruption=-0.2,
        )

        assert env.stats["reward_task_success"] == pytest.approx(0.5)
        assert env.stats["penalty_delay"] == pytest.approx(-0.2)
        assert env.stats["penalty_service_interruption"] == pytest.approx(-0.2)
    finally:
        env.close()


def test_service_interruption_penalty_is_computed_per_user():
    env = _build_single_user_env(
        num_users=3,
        reward_interruption_weight=0.3,
    )

    try:
        penalties = env._compute_service_interruption_penalties(
            np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
        )

        assert penalties == pytest.approx([0.0, -0.15, -0.3])
    finally:
        env.close()


def test_service_interruption_penalty_is_bounded_by_its_weight():
    env = _build_single_user_env(reward_interruption_weight=0.3)

    try:
        penalties = env._compute_service_interruption_penalties(
            np.asarray([10.0], dtype=np.float32)
        )

        assert penalties[0] == pytest.approx(-0.3)
    finally:
        env.close()


def test_load_balance_reward_uses_reachable_jain_fairness(monkeypatch):
    env = _build_single_user_env(reward_load_balance_weight=0.05)

    try:
        monkeypatch.setattr(
            env,
            "_compute_systemwide_jain_mec_load_fairness",
            lambda: 0.8,
        )
        assert env._compute_load_balance_reward() == pytest.approx(0.04)

        monkeypatch.setattr(
            env,
            "_compute_systemwide_jain_mec_load_fairness",
            lambda: None,
        )
        assert env._compute_load_balance_reward() == pytest.approx(0.0)

        monkeypatch.setattr(
            env,
            "_compute_systemwide_jain_mec_load_fairness",
            lambda: 0.8,
        )
        # 持续保持公平状态时每步都获得同样的 dense cooperative reward。
        assert env._compute_load_balance_reward() == pytest.approx(0.04)
    finally:
        env.close()


def test_load_balance_state_is_reset_at_episode_boundary():
    env = _build_single_user_env(reward_load_balance_weight=0.05)

    try:
        env._previous_load_balance_potential = 0.8
        env.reset(seed=17)

        assert env._previous_load_balance_potential == pytest.approx(0.0)
    finally:
        env.close()


def test_task_reward_prioritizes_deadline_success_over_energy_savings():
    env = _build_single_user_env()

    try:
        success_reward, success_terms = env._compute_task_reward(
            total_delay=1.0,
            total_energy=0.8,
            max_delay=2.0,
        )
        late_reward, late_terms = env._compute_task_reward(
            total_delay=2.2,
            total_energy=0.0,
            max_delay=2.0,
        )

        assert success_terms["reward_task_success"] == pytest.approx(TASK_SUCCESS_REWARD)
        assert success_terms["penalty_delay"] == pytest.approx(-0.3)
        assert success_terms["penalty_energy"] == pytest.approx(
            -0.4 * 0.8 / 1.8
        )
        assert late_terms["penalty_task_failure"] == pytest.approx(-TASK_FAILURE_PENALTY)
        assert late_terms["penalty_delay"] == pytest.approx(0.0)
        assert late_terms["penalty_energy"] == pytest.approx(0.0)
        assert late_reward == pytest.approx(-TASK_FAILURE_PENALTY)
        assert success_reward > late_reward
    finally:
        env.close()


def test_energy_reward_uses_smooth_monotonic_normalization_without_hard_clip():
    env = _build_single_user_env()

    try:
        _, one_joule_terms = env._compute_task_reward(
            total_delay=0.5,
            total_energy=1.0,
            max_delay=2.0,
        )
        _, seven_joule_terms = env._compute_task_reward(
            total_delay=0.5,
            total_energy=7.0,
            max_delay=2.0,
        )

        assert one_joule_terms["penalty_energy"] == pytest.approx(-0.2)
        assert seven_joule_terms["penalty_energy"] == pytest.approx(-0.35)
        assert seven_joule_terms["penalty_energy"] < one_joule_terms["penalty_energy"]
        assert seven_joule_terms["penalty_energy"] > -0.4
    finally:
        env.close()


def test_failed_task_does_not_receive_additional_delay_or_energy_penalties():
    env = _build_single_user_env()

    try:
        _, late_terms = env._compute_task_reward(
            total_delay=3.0,
            total_energy=10.0,
            max_delay=1.0,
        )

        assert late_terms["penalty_delay"] == pytest.approx(0.0)
        assert late_terms["penalty_energy"] == pytest.approx(0.0)
        assert late_terms["penalty_task_failure"] == pytest.approx(-1.0)
    finally:
        env.close()


def test_training_defaults_use_balanced_update_budget():
    config = TrainConfig()

    assert config.n_epochs == 4
    assert config.learning_rate == pytest.approx(1e-4)
    assert config.batch_size == 512
    assert config.entropy_schedule == "constant"
    assert config.reward_failed_handover_penalty == pytest.approx(0.2)
    assert REWARD_ENERGY_REFERENCE_J == pytest.approx(1.0)


def test_reward_default_weights_are_balanced():
    env_config = EnvConfig()
    train_config = TrainConfig()

    expected = {
        "reward_delay_weight": 0.60,
        "reward_energy_weight": 0.40,
        "reward_interruption_weight": 0.30,
        "reward_failed_handover_penalty": 0.20,
        "reward_load_balance_weight": 0.05,
    }

    for key, value in expected.items():
        assert getattr(env_config, key) == pytest.approx(value)
        assert getattr(train_config, key) == pytest.approx(value)


def test_comparison_defaults_use_qos_gated_reward_weights():
    from scripts.compare_system_baselines import build_default_train_config

    config = build_default_train_config(
        objective="multi_objective",
        seed=42,
        max_steps=600,
        num_users=10,
        best_model_metric="avg_delay",
    )

    assert config["reward_delay_weight"] == pytest.approx(0.60)
    assert config["reward_energy_weight"] == pytest.approx(0.40)
    assert config["reward_interruption_weight"] == pytest.approx(0.30)
    assert config["reward_failed_handover_penalty"] == pytest.approx(0.20)
    assert config["reward_load_balance_weight"] == pytest.approx(0.05)
