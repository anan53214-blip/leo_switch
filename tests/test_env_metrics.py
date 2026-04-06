import math

from src.environment.gym_env import EnvConfig, LEOSatelliteEnv
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


def test_stats_summary_uses_resolved_tasks_for_rates():
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
        assert math.isclose(stats['avg_delay'], 2.0)
        assert math.isclose(stats['avg_load_balance_score'], 0.9)
    finally:
        env.close()


def test_task_generation_skips_blocked_users():
    env = _build_single_user_env(task_arrival_prob=1.0)

    try:
        env.reset(seed=11)
        env.user_manager.users[0].state = UserState.BLOCKED
        env.user_tasks[0] = None

        env._generate_tasks()

        assert env.stats['total_tasks'] == 0
        assert env.user_tasks[0] is None
    finally:
        env.close()


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
