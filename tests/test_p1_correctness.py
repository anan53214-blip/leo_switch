import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch.nn as nn

from scripts.train import HANMAPPOTrainer
from src.environment.gym_env import EnvConfig, LEOSatelliteEnv
from src.environment.mec import MECConfig, MECManager, MECServer
from src.environment.user import UserState
from src.environment.visibility import VisibilityInfo


def _enqueue(
    server: MECServer,
    *,
    user_id: int,
    task_id: int,
    cycles: float,
) -> None:
    assert server.enqueue_task(
        user_id=user_id,
        task_id=task_id,
        offload_cycles=cycles,
        offload_data_bits=0.0,
        max_delay=100.0,
        arrival_time=0.0,
        upload_delay=0.0,
        download_delay=0.0,
    )


def test_mec_completion_time_is_interpolated_inside_current_slot():
    server = MECServer(
        satellite_id=0,
        config=MECConfig(
            satellite_cpu_freq_ghz=1.0,
            satellite_num_cores=1,
            max_queue_size=2,
        ),
    )
    _enqueue(server, user_id=0, task_id=1, cycles=0.25e9)

    completed = server.process_queue(current_time=0.0, time_step=1.0)

    assert len(completed) == 1
    assert completed[0]["start_processing_time"] == pytest.approx(0.0)
    assert completed[0]["finish_time"] == pytest.approx(0.25)
    assert completed[0]["total_delay"] == pytest.approx(0.25)


def test_mec_limits_processing_to_two_slots_and_preserves_fcfs_order():
    server = MECServer(
        satellite_id=0,
        config=MECConfig(
            satellite_cpu_freq_ghz=5.0,
            satellite_num_cores=2,
            mec_max_concurrent_tasks=2,
            max_queue_size=6,
        ),
    )
    for task_id in range(3):
        _enqueue(
            server,
            user_id=task_id,
            task_id=task_id,
            cycles=5e9,
        )

    first_completed = server.process_queue(current_time=0.0, time_step=1.0)

    assert [task["task_id"] for task in first_completed] == [0, 1]
    assert [task["task_id"] for task in server.task_queue] == [2]
    assert server.task_queue[0]["status"] == "processing"
    assert server.task_queue[0]["start_processing_time"] == pytest.approx(1.0)

    second_completed = server.process_queue(current_time=1.0, time_step=1.0)

    assert [task["task_id"] for task in second_completed] == [2]
    assert second_completed[0]["queue_wait"] == pytest.approx(1.0)
    assert second_completed[0]["finish_time"] == pytest.approx(1.5)


def test_queued_mec_task_can_timeout_without_consuming_a_processing_slot():
    server = MECServer(
        satellite_id=0,
        config=MECConfig(
            satellite_cpu_freq_ghz=1.0,
            satellite_num_cores=1,
            mec_max_concurrent_tasks=1,
            max_queue_size=3,
        ),
    )
    _enqueue(server, user_id=0, task_id=1, cycles=10e9)
    assert server.enqueue_task(
        user_id=1,
        task_id=2,
        offload_cycles=1e9,
        offload_data_bits=0.0,
        max_delay=0.5,
        arrival_time=0.0,
    )

    completed = server.process_queue(current_time=0.0, time_step=1.0)

    timed_out = next(task for task in completed if task["task_id"] == 2)
    assert timed_out["status"] == "timeout"
    assert timed_out["start_processing_time"] is None
    assert timed_out["processing_time"] == pytest.approx(0.0)
    assert timed_out["queue_wait"] == pytest.approx(1.0)


def test_environment_processes_mec_over_slot_start_time():
    env = LEOSatelliteEnv(
        EnvConfig(
            num_users=1,
            max_steps=2,
            task_arrival_prob=0.0,
            randomize_episode_start=False,
            seed=7,
        )
    )
    try:
        calls = []

        def capture_process(current_time, time_step):
            calls.append((current_time, time_step))
            return []

        env.mec_manager.process_all_queues = capture_process
        env._update_environment()

        assert calls == [(0.0, env.config.time_step_sec)]
        assert env.current_time == pytest.approx(env.config.time_step_sec)
    finally:
        env.close()


def test_migration_is_all_or_nothing_and_preserves_progress():
    manager = MECManager(
        num_satellites=2,
        config=MECConfig(max_queue_size=2),
    )
    old_server = manager.get_server(0)
    new_server = manager.get_server(1)
    assert old_server is not None
    assert new_server is not None
    _enqueue(old_server, user_id=3, task_id=11, cycles=1e9)
    _enqueue(old_server, user_id=3, task_id=12, cycles=2e9)
    _enqueue(new_server, user_id=9, task_id=22, cycles=1e9)
    old_server.task_queue[0]["remaining_cycles"] = 0.4e9

    plan = manager.prepare_user_task_migration(
        user_id=3,
        old_sat_id=0,
        new_sat_id=1,
    )

    assert not plan.feasible
    assert plan.failure_reason == "target_queue_capacity"
    result = manager.commit_user_task_migration(plan, handover_delay=0.6)

    assert result["migrated"] == 0
    assert result["failed"] == 2
    assert result["failed_task_ids"] == [11, 12]
    assert [task["task_id"] for task in old_server.task_queue] == [11, 12]
    assert [task["task_id"] for task in new_server.task_queue] == [22]
    assert old_server.task_queue[0]["upload_delay"] == pytest.approx(0.0)

    new_server.task_queue.clear()
    retry_plan = manager.prepare_user_task_migration(
        user_id=3,
        old_sat_id=0,
        new_sat_id=1,
    )
    retry_result = manager.commit_user_task_migration(
        retry_plan,
        handover_delay=0.6,
    )
    assert retry_result["migrated_task_ids"] == [11, 12]
    assert retry_result["failed_task_ids"] == []
    assert old_server.task_queue == []
    assert new_server.task_queue[0]["upload_delay"] == pytest.approx(0.6)
    assert new_server.task_queue[0]["remaining_cycles"] == pytest.approx(0.4e9)
    assert new_server.task_queue[0]["status"] == "queued"
    assert new_server.task_queue[0]["start_processing_time"] is None


def _target_visibility(sat_id: int) -> VisibilityInfo:
    return VisibilityInfo(
        sat_id=sat_id,
        is_visible=True,
        elevation_deg=60.0,
        azimuth_deg=0.0,
        distance_km=700.0,
        rvt_seconds=120.0,
    )


def test_handover_aborts_without_changing_satellite_when_migration_cannot_fit():
    env = LEOSatelliteEnv(EnvConfig(num_users=1, task_arrival_prob=0.0, seed=7))
    try:
        env.reset(seed=7)
        user = env.user_manager.users[0]
        old_sat_id = user.serving_satellite
        new_sat_id = (old_sat_id + 1) % env.num_satellites
        old_server = env.mec_manager.get_server(old_sat_id)
        new_server = env.mec_manager.get_server(new_sat_id)
        assert old_server is not None and new_server is not None
        _enqueue(old_server, user_id=user.user_id, task_id=31, cycles=1e9)
        for task_id in range(new_server.config.max_queue_size):
            _enqueue(new_server, user_id=99, task_id=100 + task_id, cycles=1e9)
        env._check_handover_link_feasibility = lambda *_: (True, None)
        env._is_satellite_visible = lambda *_: True

        reward = env._execute_handover(user, _target_visibility(new_sat_id))

        assert reward < 0.0
        assert user.state == UserState.CONNECTED
        assert user.serving_satellite == old_sat_id
        assert [task["task_id"] for task in old_server.task_queue] == [31]
        assert all(task["user_id"] != user.user_id for task in new_server.task_queue)
        assert env.stats["handover_attempts"] == 1
        assert env.stats["handover_aborted"] == 1
        assert env.stats["handover_committed"] == 0
        assert env.stats["migration_rejections"] == 1
    finally:
        env.close()


def test_handover_commits_association_only_after_all_tasks_are_enqueued():
    env = LEOSatelliteEnv(EnvConfig(num_users=1, task_arrival_prob=0.0, seed=7))
    try:
        env.reset(seed=7)
        user = env.user_manager.users[0]
        old_sat_id = user.serving_satellite
        new_sat_id = (old_sat_id + 1) % env.num_satellites
        old_server = env.mec_manager.get_server(old_sat_id)
        new_server = env.mec_manager.get_server(new_sat_id)
        assert old_server is not None and new_server is not None
        _enqueue(old_server, user_id=user.user_id, task_id=41, cycles=1e9)
        _enqueue(old_server, user_id=user.user_id, task_id=42, cycles=2e9)
        env._check_handover_link_feasibility = lambda *_: (True, None)
        env._is_satellite_visible = lambda *_: True

        env._execute_handover(user, _target_visibility(new_sat_id))

        assert user.state == UserState.CONNECTED
        assert user.serving_satellite == new_sat_id
        assert all(task["user_id"] != user.user_id for task in old_server.task_queue)
        assert [
            task["task_id"] for task in new_server.task_queue
            if task["user_id"] == user.user_id
        ] == [41, 42]
        assert env.stats["handover_committed"] == 1
        assert env.stats["successful_handovers"] == 1
        assert env.stats["handover_aborted"] == 0
    finally:
        env.close()


def test_user_is_blocked_only_when_old_link_is_lost_and_target_rejects_tasks():
    env = LEOSatelliteEnv(EnvConfig(num_users=1, task_arrival_prob=0.0, seed=7))
    try:
        env.reset(seed=7)
        user = env.user_manager.users[0]
        old_sat_id = user.serving_satellite
        new_sat_id = (old_sat_id + 1) % env.num_satellites
        old_server = env.mec_manager.get_server(old_sat_id)
        new_server = env.mec_manager.get_server(new_sat_id)
        assert old_server is not None and new_server is not None
        _enqueue(old_server, user_id=user.user_id, task_id=51, cycles=1e9)
        for task_id in range(new_server.config.max_queue_size):
            _enqueue(new_server, user_id=99, task_id=200 + task_id, cycles=1e9)
        env._check_handover_link_feasibility = lambda *_: (True, None)
        env._is_satellite_visible = lambda *_: False

        env._execute_handover(user, _target_visibility(new_sat_id))

        assert user.state == UserState.BLOCKED
        assert user.serving_satellite == -1
        assert all(task["user_id"] != user.user_id for task in old_server.task_queue)
        assert env.stats["forced_disconnects"] == 1
        assert env.stats["failed_tasks"] == 1
        assert env.stats["handover_aborted"] == 1
    finally:
        env.close()


def test_failed_reconnection_keeps_user_blocked_without_counting_handover():
    env = LEOSatelliteEnv(EnvConfig(num_users=1, task_arrival_prob=0.0, seed=7))
    try:
        env.reset(seed=7)
        user = env.user_manager.users[0]
        old_sat_id = user.serving_satellite
        old_server = env.mec_manager.get_server(old_sat_id)
        assert old_server is not None
        old_server.remove_user(user.user_id)
        user.serving_satellite = -1
        user.state = UserState.BLOCKED
        target_sat_id = (old_sat_id + 1) % env.num_satellites
        env._check_handover_link_feasibility = lambda *_: (False, "snr_below_threshold")

        reward = env._execute_handover(user, _target_visibility(target_sat_id))

        assert reward < 0.0
        assert user.state == UserState.BLOCKED
        assert user.serving_satellite == -1
        assert env.stats["handover_attempts"] == 0
        assert env.stats["reconnection_attempts"] == 1
        assert env.stats["reconnections"] == 0
    finally:
        env.close()


def test_successful_reconnection_is_not_counted_as_handover():
    env = LEOSatelliteEnv(EnvConfig(num_users=1, task_arrival_prob=0.0, seed=7))
    try:
        env.reset(seed=7)
        user = env.user_manager.users[0]
        old_sat_id = user.serving_satellite
        old_server = env.mec_manager.get_server(old_sat_id)
        assert old_server is not None
        old_server.remove_user(user.user_id)
        user.serving_satellite = -1
        user.state = UserState.BLOCKED
        target_sat_id = (old_sat_id + 1) % env.num_satellites
        env._check_handover_link_feasibility = lambda *_: (True, None)

        env._execute_handover(user, _target_visibility(target_sat_id))

        assert user.state == UserState.CONNECTED
        assert user.serving_satellite == target_sat_id
        assert env.stats["reconnection_attempts"] == 1
        assert env.stats["reconnections"] == 1
        assert env.stats["handover_attempts"] == 0
        assert env.stats["successful_handovers"] == 0
    finally:
        env.close()


class _FakeEvalEnv:
    def __init__(self):
        self.reset_seeds = []
        self.step_calls = 0
        self.closed = False

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return None, {}

    def get_pre_handover_mask(self):
        return np.ones(1, dtype=bool)

    def step(self, actions, return_observation=False, return_info=False):
        self.step_calls += 1
        return None, 1.0, False, True, {}

    def get_stats_summary(self):
        return {}

    def close(self):
        self.closed = True


def test_on_policy_evaluation_uses_isolated_env_and_fixed_seed():
    trainer = HANMAPPOTrainer.__new__(HANMAPPOTrainer)
    training_env = _FakeEvalEnv()
    eval_env = _FakeEvalEnv()
    trainer.env = training_env
    trainer.config = SimpleNamespace(
        eval_episodes=1,
        seed=7,
        best_model_metric="reward",
    )
    trainer.total_steps = 10
    trainer.episodes = 2
    trainer.best_reward = float("inf")
    trainer.best_model_score = float("inf")
    trainer.eval_history = []
    trainer.logger = logging.getLogger("test_p1_eval_isolation")
    trainer.han_encoder = nn.Linear(1, 1)
    trainer._cached_han_user_embed = np.asarray([[1.0]], dtype=np.float32)
    trainer._cached_sat_embed = np.asarray([[2.0]], dtype=np.float32)
    trainer._cached_graph_snapshot = object()
    trainer._create_eval_env = lambda: eval_env
    trainer._encode_graph_state = lambda: (
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        np.ones((1, 2), dtype=np.float32),
        np.zeros((1, 1), dtype=np.int64),
    )
    trainer._apply_pre_handover_action_mask = lambda actions, mask: actions
    trainer._process_actions = lambda actions: np.zeros((1, 2), dtype=np.float32)
    trainer._save_checkpoint = lambda best=False, final=False: None
    trainer.mappo = SimpleNamespace(
        actor=nn.Linear(1, 1),
        critic=nn.Linear(1, 1),
        act=lambda *args, **kwargs: (
            {"handover": np.zeros(1), "offload": np.zeros(1)},
            None,
            None,
        ),
    )

    training_cache = trainer._cached_graph_snapshot
    trainer._evaluate()

    assert training_env.reset_seeds == []
    assert training_env.step_calls == 0
    assert eval_env.reset_seeds == [1_000_007]
    assert eval_env.step_calls == 1
    assert eval_env.closed
    assert trainer.env is training_env
    assert trainer._cached_graph_snapshot is training_cache


def test_current_serving_satellite_is_not_a_handover_candidate_or_event():
    env = LEOSatelliteEnv(
        EnvConfig(
            num_users=1,
            max_steps=2,
            task_arrival_prob=0.0,
            randomize_episode_start=False,
            seed=7,
        )
    )
    try:
        env.reset(seed=7)
        user = env.user_manager.users[0]
        current_satellite = user.serving_satellite
        assert current_satellite >= 0

        candidates = env._get_handover_candidates(user)
        assert all(item.sat_id != current_satellite for item in candidates)

        current_visibility = env._get_satellite_visibility(user, current_satellite)
        assert current_visibility is not None
        handovers_before = env.stats["total_handovers"]
        reward = env._execute_handover(user, current_visibility)

        assert reward == pytest.approx(0.0)
        assert env.stats["total_handovers"] == handovers_before
        assert user.serving_satellite == current_satellite
    finally:
        env.close()


def test_terminal_step_flushes_rewards_created_during_environment_update():
    env = LEOSatelliteEnv(
        EnvConfig(
            num_users=1,
            max_steps=1,
            task_arrival_prob=0.0,
            reward_interruption_weight=0.0,
            randomize_episode_start=False,
            seed=7,
        )
    )
    try:
        env.reset(seed=7)
        env._execute_user_handover = lambda user, handover: 1.0
        env._execute_user_task = lambda user, offload, allocation: 0.0

        def create_completion_reward():
            env.pending_rewards[0] = env.pending_rewards.get(0, 0.0) + 2.0

        env._update_environment = create_completion_reward
        _, reward, terminated, truncated, _ = env.step(
            np.asarray([[0.0, 0.0]], dtype=np.float32),
            return_observation=False,
        )

        assert not terminated
        assert truncated
        assert reward == pytest.approx(3.0)
        assert env.last_user_rewards[0] == pytest.approx(3.0)
        assert env.pending_rewards.get(0, 0.0) == pytest.approx(0.0)
    finally:
        env.close()
