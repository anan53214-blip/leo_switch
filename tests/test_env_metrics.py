import math

import pytest

from src.environment.gym_env import EnvConfig, LEOSatelliteEnv, summarize_env_stats
from src.environment.mec import MECConfig, MECServer
from src.environment.user import UserState


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
        assert math.isclose(stats['avg_load_balance_score'], 0.9)
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


def test_effective_latency_score_uses_task_success_rate_not_settlement_rate():
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
    assert good["effective_latency_score"] > bad["effective_latency_score"]


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


def test_effective_latency_score_penalizes_low_service_coverage():
    reliable = summarize_env_stats({
        'total_tasks': 100,
        'completed_tasks': 60,
        'deadline_violations': 40,
        'total_delay': 220.0,
        'total_user_seconds': 100.0,
        'service_interruption_seconds': 3.0,
    })
    low_coverage = summarize_env_stats({
        'total_tasks': 100,
        'completed_tasks': 55,
        'deadline_violations': 15,
        'total_delay': 140.0,
        'total_user_seconds': 100.0,
        'service_interruption_seconds': 57.0,
    })

    assert low_coverage['avg_delay'] < reliable['avg_delay']
    assert reliable['effective_latency_score'] > low_coverage['effective_latency_score']


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
    assert server.queue_length == 0
