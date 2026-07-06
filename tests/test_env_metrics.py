import math

import numpy as np
import pytest

from src.environment.gym_env import EnvConfig, LEOSatelliteEnv, summarize_env_stats
from src.environment.mec import MECConfig, MECServer
from src.environment.user import UserState
from scripts.compare_system_baselines import (
    HIGHER_IS_BETTER,
    REWARD_BREAKDOWN_KEYS,
    SUMMARY_METRIC_KEYS,
    load_variance_samples_for_plot,
    summarize_results,
)


def _build_single_user_env(**overrides) -> LEOSatelliteEnv:
    config_kwargs = {
        'num_users': 1,
        'max_steps': 5,
        'seed': 11,
        'task_arrival_prob': 0.0,
    }
    config_kwargs.update(overrides)
    config = EnvConfig(
        **config_kwargs,
    )
    return LEOSatelliteEnv(config)


def test_stats_summary_uses_external_task_denominator_for_deadline_violation_rate():
    env = _build_single_user_env()

    try:
        env.stats.update({
            'total_handovers': 4,
            'successful_handovers': 3,
            'forced_disconnects': 1,
            'total_tasks': 10,
            'completed_tasks': 3,
            'deadline_violations': 2,
            'total_delay': 10.0,
            'load_balance_sum': 4.5,
            'load_balance_samples': 5,
        })

        stats = env._get_info()['stats']

        assert math.isclose(stats['handover_success_rate'], 0.75)
        assert math.isclose(stats['service_continuity_rate'], 0.8)
        assert stats['resolved_tasks'] == 5
        assert stats['pending_tasks'] == 5
        assert math.isclose(stats['task_completion_rate'], 0.6)
        assert math.isclose(stats['task_resolution_rate'], 0.5)
        assert math.isclose(stats['pending_task_rate'], 0.5)
        assert math.isclose(stats['deadline_violation_rate'], 0.2)
        assert math.isclose(stats['avg_delay'], 2.0)
        assert math.isclose(stats['mec_load_fairness'], 0.9)
        assert math.isclose(stats['avg_load_balance_score'], 0.9)
        assert math.isclose(stats['active_load_balance_score'], 0.9)
        assert math.isclose(stats['energy_per_successful_task'], 0.0)
        assert 'mec_activity_score' not in stats
        assert 'mec_load_mean' not in stats
        assert 'service_downtime_rate' not in stats
    finally:
        env.close()


def test_task_success_metrics_distinguish_success_failure_and_settlement():
    stats = {
        "total_tasks": 100,
        "completed_tasks": 40,
        "deadline_violations": 55,
        "total_delay": 250.0,
        "total_user_seconds": 1000.0,
        "blocked_user_seconds": 0.0,
        "handover_interruption_seconds": 0.0,
        "service_interruption_seconds": 0.0,
        "total_energy": 80.0,
    }

    summary = summarize_env_stats(stats)

    assert summary["resolved_tasks"] == 95
    assert summary["pending_tasks"] == 5
    assert summary["task_success_rate"] == pytest.approx(0.40)
    assert summary["task_failure_rate"] == pytest.approx(0.55)
    assert summary["task_settlement_rate"] == pytest.approx(0.95)
    assert summary["task_resolution_rate"] == pytest.approx(0.95)
    assert summary["task_completion_rate"] == pytest.approx(40 / 95)
    assert summary["deadline_violation_rate"] == pytest.approx(0.55)
    assert summary["energy_per_successful_task"] == pytest.approx(2.0)
    assert "mec_activity_score" not in summary
    assert "mec_load_mean" not in summary
    assert "service_downtime_rate" not in summary


def test_compare_summary_includes_task_success_metrics():
    assert "task_success_rate" in SUMMARY_METRIC_KEYS
    assert "task_failure_rate" in SUMMARY_METRIC_KEYS
    assert "task_settlement_rate" in SUMMARY_METRIC_KEYS
    assert "handover_frequency" in SUMMARY_METRIC_KEYS
    assert "mec_load_fairness" in SUMMARY_METRIC_KEYS
    assert "energy_per_successful_task" in SUMMARY_METRIC_KEYS
    assert "service_downtime_rate" not in SUMMARY_METRIC_KEYS
    assert "mec_activity_score" not in SUMMARY_METRIC_KEYS
    assert "mec_load_mean" not in SUMMARY_METRIC_KEYS
    assert HIGHER_IS_BETTER["task_success_rate"] is True
    assert HIGHER_IS_BETTER["task_failure_rate"] is False
    assert HIGHER_IS_BETTER["task_settlement_rate"] is True
    assert HIGHER_IS_BETTER["handover_frequency"] is False
    assert HIGHER_IS_BETTER["mec_load_fairness"] is True
    assert HIGHER_IS_BETTER["energy_per_successful_task"] is False
    assert "service_downtime_rate" not in HIGHER_IS_BETTER


def test_load_balance_score_ignores_connection_only_distribution():
    env = _build_single_user_env(num_users=2)

    try:
        env.reset(seed=11)
        env.mec_manager.servers[0].add_user(0)
        env.mec_manager.servers[1].add_user(1)

        assert env._compute_load_balance_score() == pytest.approx(0.0)
    finally:
        env.close()


def test_load_balance_score_rewards_balanced_mec_workload():
    env = _build_single_user_env(num_users=2)

    try:
        env.reset(seed=11)

        first = env.mec_manager.servers[0]
        second = env.mec_manager.servers[1]
        first.task_queue = [{} for _ in range(3)]
        second.task_queue = [{} for _ in range(3)]
        first.available_freq_ghz = first.total_capacity_ghz * 0.5
        second.available_freq_ghz = second.total_capacity_ghz * 0.5
        balanced = env._compute_load_balance_score()

        first.task_queue = [{} for _ in range(6)]
        second.task_queue = []
        first.available_freq_ghz = 0.0
        second.available_freq_ghz = second.total_capacity_ghz
        imbalanced = env._compute_load_balance_score()

        assert balanced > imbalanced
    finally:
        env.close()


def test_time_point_load_variance_penalizes_single_busy_mec_among_idle_servers():
    env = _build_single_user_env(num_users=2)

    try:
        env.reset(seed=11)

        first = env.mec_manager.servers[0]
        first.task_queue = [{} for _ in range(3)]
        first.available_freq_ghz = 0.0

        assert env._compute_load_balance_variance() > 0.0
    finally:
        env.close()


def test_mec_load_fairness_ignores_idle_satellites_when_mec_is_used():
    env = _build_single_user_env(num_users=2)

    try:
        env.reset(seed=11)

        first = env.mec_manager.servers[0]
        second = env.mec_manager.servers[1]
        first.task_queue = [{} for _ in range(1)]
        second.task_queue = [{} for _ in range(1)]

        assert env._compute_mec_load_fairness() == pytest.approx(1.0)
    finally:
        env.close()


def test_summary_reports_mec_load_fairness_without_activity_or_mean_load():
    env = _build_single_user_env(num_users=2)

    try:
        env.reset(seed=11)

        first = env.mec_manager.servers[0]
        second = env.mec_manager.servers[1]
        first.task_queue = [{} for _ in range(1)]
        second.task_queue = [{} for _ in range(1)]
        env.stats['load_balance_sum'] = env._compute_mec_load_fairness()
        env.stats['load_balance_samples'] = 1

        summary = env.get_stats_summary()

        assert summary["mec_load_fairness"] == pytest.approx(1.0)
        assert summary["active_load_balance_score"] == pytest.approx(1.0)
        assert "mec_activity_score" not in summary
        assert "mec_load_mean" not in summary
    finally:
        env.close()


def test_env_records_only_active_time_point_load_variance_samples_for_cdf():
    env = _build_single_user_env(num_users=2)

    try:
        env.reset(seed=11)

        env._record_load_balance_metrics()
        assert env.get_stats_summary()["load_variance_sample_count"] == 0

        first = env.mec_manager.servers[0]
        first.task_queue = [{} for _ in range(3)]
        first.available_freq_ghz = 0.0
        env._record_load_balance_metrics()

        summary = env.get_stats_summary()

        assert summary["load_variance_sample_count"] == 1
        assert len(summary["load_variance_samples"]) == 1
        assert summary["load_variance_samples"][0] > 0.0
        assert len(summary["load_variance_cdf"]) == 1
    finally:
        env.close()


def test_load_variance_cdf_plot_requires_real_variance_samples():
    assert load_variance_samples_for_plot({"avg_load_balance_score": 0.5}) == []
    assert load_variance_samples_for_plot(
        {
            "load_variance_cdf": [
                {"x": 0.03, "cdf": 0.5},
                {"x": 0.05, "cdf": 1.0},
            ]
        }
    ) == pytest.approx([0.03, 0.05])


def test_compare_summary_preserves_reward_breakdown_metrics():
    summary = {
        "avg_delay": 1.0,
        "total_tasks": 2,
        "completed_tasks": 2,
        "reward_delay": 2.5,
        "reward_energy": 1.5,
        "reward_qos": 0.8,
        "reward_service_continuity": 0.4,
        "penalty_deadline": -0.1,
    }

    result = summarize_results("example", [4.0], [summary])
    episode = result["episode_metrics"][0]

    assert "reward_service_continuity" in REWARD_BREAKDOWN_KEYS
    assert result["reward_delay"] == pytest.approx(2.5)
    assert result["reward_service_continuity"] == pytest.approx(0.4)
    assert episode["penalty_deadline"] == pytest.approx(-0.1)


def test_stats_summary_removes_custom_composite_latency_score():
    base = {
        "total_tasks": 100,
        "total_delay": 100.0,
        "total_user_seconds": 1000.0,
        "blocked_user_seconds": 0.0,
        "handover_interruption_seconds": 0.0,
        "service_interruption_seconds": 0.0,
    }
    good = summarize_env_stats({**base, "completed_tasks": 80, "deadline_violations": 15})
    bad = summarize_env_stats({**base, "completed_tasks": 20, "deadline_violations": 75})

    assert good["task_settlement_rate"] == pytest.approx(bad["task_settlement_rate"])
    assert good["task_success_rate"] > bad["task_success_rate"]
    assert "effective_latency_score" not in good
    assert "effective_latency_score" not in bad


def test_stats_summary_prefers_time_based_service_reliability_metrics():
    env = _build_single_user_env()

    try:
        env.stats.update({
            'total_user_seconds': 100.0,
            'blocked_user_seconds': 8.0,
            'handover_interruption_seconds': 5.0,
            'service_interruption_seconds': 13.0,
            'total_handovers': 4,
            'successful_handovers': 4,
            'total_tasks': 1,
            'completed_tasks': 1,
        })

        stats = env._get_info()['stats']

        assert math.isclose(stats['service_availability_rate'], 0.92)
        assert math.isclose(stats['service_continuity_rate'], 0.87)
        assert math.isclose(stats['forced_termination_rate'], 0.0)
    finally:
        env.close()


def test_external_task_generation_includes_blocked_users():
    env = _build_single_user_env(task_arrival_prob=1.0)

    try:
        env.reset(seed=11)
        env.user_manager.users[0].state = UserState.BLOCKED
        env.user_tasks[0] = None

        env._generate_tasks()

        assert env.stats['total_tasks'] == 1
        assert env.user_tasks[0] is not None
    finally:
        env.close()


def test_blocked_pending_task_expires_as_deadline_violation():
    env = _build_single_user_env(task_arrival_prob=1.0)

    try:
        env.reset(seed=11)
        env.user_manager.users[0].state = UserState.BLOCKED
        env._generate_tasks()
        task = env.user_tasks[0]
        task.max_delay = 0.5

        env.current_time = 1.0
        env._expire_pending_user_tasks()

        assert env.stats['deadline_violations'] == 1
        assert env.stats['total_delay'] == 1.0
        assert env.user_tasks[0] is None
    finally:
        env.close()


def test_expired_pending_task_adds_deadline_penalty_signal():
    env = _build_single_user_env(task_arrival_prob=1.0)

    try:
        env.reset(seed=11)
        env.user_manager.users[0].state = UserState.BLOCKED
        env._generate_tasks()
        task = env.user_tasks[0]
        task.max_delay = 0.5

        env.current_time = 1.0
        env._expire_pending_user_tasks()

        assert env.stats['penalty_deadline'] < 0.0
        assert env.stats['reward_energy'] == 0.0
        assert env.pending_rewards[0] < 0.0
    finally:
        env.close()


def test_task_arrivals_use_independent_rng_from_handover_outcomes():
    env_a = _build_single_user_env(num_users=2, task_arrival_prob=0.5, seed=31)
    env_b = _build_single_user_env(num_users=2, task_arrival_prob=0.5, seed=31)

    try:
        env_a.reset(seed=31)
        env_b.reset(seed=31)

        # Consume the environment RNG only in env_b. Task arrivals should still
        # be identical because they use a dedicated task-arrival generator.
        for _ in range(7):
            env_b.rng.random()

        env_a._generate_tasks()
        env_b._generate_tasks()

        arrivals_a = [task is not None for task in env_a.user_tasks.values()]
        arrivals_b = [task is not None for task in env_b.user_tasks.values()]
        assert arrivals_a == arrivals_b
    finally:
        env_a.close()
        env_b.close()


def test_step_exposes_per_agent_rewards_matching_scalar_mean():
    env = _build_single_user_env(num_users=2, task_arrival_prob=0.0)

    try:
        env.reset(seed=11)
        _, reward, _, _, info = env.step(np.zeros((2, 2), dtype=np.float32))

        assert env.last_user_rewards.shape == (2,)
        assert "user_rewards" in info
        assert np.allclose(info["user_rewards"], env.last_user_rewards)
        assert float(np.mean(env.last_user_rewards)) == pytest.approx(reward)
    finally:
        env.close()


def test_handover_frequency_uses_time_normalized_handover_count():
    summary = summarize_env_stats({
        'total_tasks': 100,
        'completed_tasks': 60,
        'deadline_violations': 40,
        'total_delay': 220.0,
        'total_user_seconds': 100.0,
        'service_interruption_seconds': 3.0,
        'total_handovers': 8,
    })

    assert summary['handover_frequency'] == pytest.approx(0.08)


def test_handover_success_probability_prefers_high_quality_targets():
    env = _build_single_user_env()

    try:
        high_quality = env._compute_handover_success_probability(
            elevation_deg=70.0,
            rvt_seconds=180.0,
            snr_db=20.0,
            utilization=0.2,
            queue_ratio=0.1,
            migration_load=0,
        )
        low_quality = env._compute_handover_success_probability(
            elevation_deg=12.0,
            rvt_seconds=10.0,
            snr_db=-3.0,
            utilization=0.95,
            queue_ratio=0.9,
            migration_load=5,
        )

        assert high_quality > low_quality
        assert 0.1 <= low_quality <= 0.995
        assert 0.1 <= high_quality <= 0.995
    finally:
        env.close()


def test_mec_server_uses_all_cores_when_processing_queue():
    server = MECServer(
        satellite_id=0,
        config=MECConfig(
            satellite_cpu_freq_ghz=5.0,
            satellite_num_cores=4,
            max_queue_size=10,
        ),
    )

    for task_id in range(4):
        assert server.enqueue_task(
            user_id=task_id,
            task_id=task_id,
            offload_cycles=5e9,
            offload_data_bits=0.0,
            max_delay=10.0,
            arrival_time=0.0,
        )

    completed = server.process_queue(current_time=0.0, time_step=1.0)

    assert len(completed) == 4


def test_reset_randomizes_constellation_start_time_when_enabled():
    from src.environment.gym_env import EnvConfig, LEOSatelliteEnv

    config = EnvConfig(
        num_users=2,
        max_steps=5,
        randomize_episode_start=True,
        episode_start_time_jitter_sec=600.0,
        seed=123,
    )
    env = LEOSatelliteEnv(config)

    env.reset(seed=123)
    first_time = env.constellation.current_time
    first_positions = env.constellation._all_pos_ecef.copy()

    env.reset(seed=124)
    second_time = env.constellation.current_time
    second_positions = env.constellation._all_pos_ecef.copy()

    assert first_time != second_time
    assert not np.allclose(first_positions, second_positions)
