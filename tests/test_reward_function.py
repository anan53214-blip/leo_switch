from pathlib import Path

import numpy as np
import pytest

from scripts.train import TrainConfig, compute_model_selection_score
from src.environment.gym_env import EnvConfig, LEOSatelliteEnv
from src.environment.mec import MECConfig


def test_custom_composite_scores_are_not_model_selection_metrics():
    from scripts.train import BEST_MODEL_METRIC_CHOICES

    assert "effective_latency_score" not in BEST_MODEL_METRIC_CHOICES
    assert "latency_priority_score" not in BEST_MODEL_METRIC_CHOICES


def test_task_success_rate_can_be_used_as_selection_metric():
    record = {"task_success_rate": 0.73}

    assert compute_model_selection_score(record, "task_success_rate") == pytest.approx(0.73)


def test_default_local_cpu_matches_constrained_terminal_scenario():
    config = MECConfig()

    assert config.user_cpu_freq_ghz == pytest.approx(0.5)


def test_g1_latency_suite_defaults_use_avg_delay_selection_metric():
    from scripts.run_latency_priority_g1_300k_600s_u10_suite import SuiteConfig

    config = SuiteConfig(run_id="unit")

    assert config.best_model_metric == "avg_delay"
    assert config.compare_ranking_metric == "avg_delay"


def test_g1_u20_latency_suite_defaults_match_run_label():
    from scripts.run_latency_priority_g1_300k_600s_u20_suite import RUN_LABEL, SuiteConfig, build_paths

    config = SuiteConfig(run_id="unit")
    paths = build_paths(project_root=Path("project"), run_id=config.run_id)

    assert RUN_LABEL == "g1_300k_600s_u20"
    assert config.exp_name == "han_mappo_latency_priority_g1_300k_600s_u20"
    assert config.num_users == 20
    assert "u20" in str(paths.system_run_dir)
    assert "u20" in str(paths.compare_output_dir)


def test_g1_u30_new_metrics_suite_defaults_train_han_attn_and_compare_core_methods():
    from scripts.run_latency_priority_g1_300k_600s_u30_new_metrics_suite import (
        DEFAULT_BASELINES,
        RUN_LABEL,
        SuiteConfig,
        build_compare_command,
        build_paths,
        build_train_command,
    )

    config = SuiteConfig(run_id="unit")
    paths = build_paths(project_root=Path("project"), run_id=config.run_id)
    train_command = build_train_command(Path("project"), paths, config)
    compare_command = build_compare_command(Path("project"), paths, config)

    assert RUN_LABEL == "g1_300k_600s_u30_new_metrics"
    assert config.exp_name == "han_attn_latency_priority_g1_300k_600s_u30_new_metrics"
    assert config.algorithm == "han_attn"
    assert config.num_users == 30
    assert DEFAULT_BASELINES == (
        "han_mappo",
        "mappo_no_han",
        "attn_mappo",
        "random",
        "min_distance",
        "full_local",
        "joint_greedy",
    )
    assert train_command[train_command.index("--algorithm") + 1] == "han_attn"
    assert train_command[train_command.index("--num_users") + 1] == "30"
    assert compare_command[compare_command.index("--baselines") + 1 :] == list(DEFAULT_BASELINES)
    assert "u30_new_metrics" in str(paths.system_run_dir)
    assert "u30_new_metrics" in str(paths.compare_output_dir)


def test_g1_u30_cpq_suite_defaults_train_cpq_han_attn_and_compare_core_methods():
    from scripts.run_latency_priority_g1_300k_600s_u30_cpq_suite import (
        DEFAULT_BASELINES,
        RUN_LABEL,
        SuiteConfig,
        build_compare_command,
        build_paths,
        build_train_command,
    )

    config = SuiteConfig(run_id="unit")
    paths = build_paths(project_root=Path("project"), run_id=config.run_id)
    train_command = build_train_command(Path("project"), paths, config)
    compare_command = build_compare_command(Path("project"), paths, config)

    assert RUN_LABEL == "g1_300k_600s_u30_cpq"
    assert config.exp_name == "han_attn_cpq_latency_priority_g1_300k_600s_u30"
    assert config.algorithm == "han_attn_cpq"
    assert config.compare_ranking_metric == "avg_delay"
    assert "han_attn" in DEFAULT_BASELINES
    assert "attn_mappo" in DEFAULT_BASELINES
    assert DEFAULT_BASELINES == (
        "han_attn",
        "attn_mappo",
        "han_mappo",
        "mappo_no_han",
        "min_distance",
        "random",
        "joint_greedy",
        "full_local",
    )
    assert train_command[train_command.index("--algorithm") + 1] == "han_attn_cpq"
    assert compare_command[compare_command.index("--baselines") + 1 :] == list(DEFAULT_BASELINES)
    assert "u30_cpq" in str(paths.system_run_dir)
    assert "u30_cpq" in str(paths.compare_output_dir)


def test_g1_u30_attn_han_final_suite_defaults_train_final_han_attn_only():
    from scripts.run_latency_priority_g1_300k_600s_u30_attn_han_final_suite import (
        DEFAULT_BASELINES,
        RUN_LABEL,
        SuiteConfig,
        build_compare_command,
        build_paths,
        build_train_command,
    )

    config = SuiteConfig(run_id="unit")
    paths = build_paths(project_root=Path("project"), run_id=config.run_id)
    train_command = build_train_command(Path("project"), paths, config)
    compare_command = build_compare_command(Path("project"), paths, config)

    assert RUN_LABEL == "g1_300k_600s_u30_attn_han_final"
    assert config.exp_name == "han_attn_latency_priority_g1_300k_600s_u30_final"
    assert config.algorithm == "han_attn"
    assert "han_attn" not in DEFAULT_BASELINES
    assert "han_attn_legacy" not in DEFAULT_BASELINES
    assert DEFAULT_BASELINES == (
        "attn_mappo",
        "han_mappo",
        "mappo_no_han",
        "min_distance",
        "random",
        "joint_greedy",
        "full_local",
    )
    assert train_command[train_command.index("--algorithm") + 1] == "han_attn"
    assert compare_command[compare_command.index("--baselines") + 1 :] == list(DEFAULT_BASELINES)
    assert "attn_han_final" in str(paths.system_run_dir)
    assert "attn_han_final" in str(paths.compare_output_dir)


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
            'reward_task_success',
            'reward_deadline_slack',
            'reward_service_continuity',
            'reward_handover',
            'reward_load_balance',
            'reward_enqueue',
            'penalty_deadline',
            'penalty_task_failure',
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


def test_task_reward_prioritizes_deadline_success_over_energy_savings():
    env = _build_single_user_env()

    try:
        success_reward, success_terms = env._compute_task_reward(
            total_delay=1.0,
            total_energy=8.0,
            max_delay=2.0,
        )
        late_reward, late_terms = env._compute_task_reward(
            total_delay=2.2,
            total_energy=0.0,
            max_delay=2.0,
        )

        assert success_terms["reward_task_success"] > 0.0
        assert success_terms["reward_deadline_slack"] > 0.0
        assert success_terms["reward_energy"] < 0.0
        assert late_terms["penalty_task_failure"] < 0.0
        assert late_terms["penalty_deadline"] < 0.0
        assert late_reward < 0.0
        assert success_reward > late_reward
    finally:
        env.close()


def test_energy_penalty_only_applies_to_successful_tasks():
    env = _build_single_user_env()

    try:
        _, late_terms = env._compute_task_reward(
            total_delay=3.0,
            total_energy=10.0,
            max_delay=1.0,
        )

        assert late_terms["reward_energy"] == pytest.approx(0.0)
        assert late_terms["penalty_task_failure"] < 0.0
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
        "reward_delay_weight": 0.35,
        "reward_energy_weight": 0.05,
        "reward_handover_weight": 0.10,
        "reward_load_balance_weight": 0.05,
        "reward_qos_weight": 0.40,
        "reward_service_continuity_weight": 0.15,
        "reward_deadline_penalty": 1.00,
        "reward_failed_task_penalty": 0.80,
        "reward_deadline_slack_weight": 0.25,
        "reward_enqueue_bonus": 0.0,
    }

    for key, value in expected.items():
        assert getattr(env_config, key) == pytest.approx(value)
        assert getattr(train_config, key) == pytest.approx(value)


def test_comparison_defaults_use_deadline_priority_reward_weights():
    from scripts.compare_system_baselines import build_default_train_config

    config = build_default_train_config(
        objective="multi_objective",
        seed=42,
        max_steps=600,
        num_users=10,
        best_model_metric="avg_delay",
    )

    assert config["reward_delay_weight"] == pytest.approx(0.35)
    assert config["reward_energy_weight"] == pytest.approx(0.05)
    assert config["reward_handover_weight"] == pytest.approx(0.10)
    assert config["reward_load_balance_weight"] == pytest.approx(0.05)
    assert config["reward_qos_weight"] == pytest.approx(0.40)
    assert config["reward_service_continuity_weight"] == pytest.approx(0.15)
    assert config["reward_deadline_penalty"] == pytest.approx(1.00)
    assert config["reward_failed_task_penalty"] == pytest.approx(0.80)
    assert config["reward_deadline_slack_weight"] == pytest.approx(0.25)
    assert config["reward_enqueue_bonus"] == pytest.approx(0.0)
