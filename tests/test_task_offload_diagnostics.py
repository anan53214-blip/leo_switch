import csv

import numpy as np
import pytest

from scripts.task_offload_diagnostics import (
    build_bimodality_summary,
    build_task_type_offload_summary,
    offload_ratio_bin,
    save_task_offload_diagnostics,
)
from src.environment.gym_env import EnvConfig, LEOSatelliteEnv
from src.environment.mec import MECConfig, MECServer


def test_mec_completion_exposes_queue_and_processing_stage_delays():
    server = MECServer(
        satellite_id=0,
        config=MECConfig(
            satellite_cpu_freq_ghz=1.0,
            satellite_num_cores=1,
            max_queue_size=2,
        ),
    )
    assert server.enqueue_task(
        user_id=0,
        task_id=7,
        offload_cycles=0.25e9,
        offload_data_bits=1e6,
        max_delay=2.0,
        arrival_time=0.0,
        upload_delay=0.1,
        download_delay=0.05,
    )

    completed = server.process_queue(current_time=0.2, time_step=1.0)

    assert len(completed) == 1
    assert completed[0]["queue_wait"] == 0.2
    assert completed[0]["processing_time"] == 0.25
    assert completed[0]["total_delay"] == pytest.approx(0.6)


def test_local_task_trace_records_decision_mode_and_terminal_result():
    env = LEOSatelliteEnv(
        EnvConfig(
            num_planes=1,
            sats_per_plane=4,
            num_users=1,
            max_steps=1,
            task_arrival_prob=1.0,
            randomize_episode_start=False,
            resample_users_on_reset=False,
            enable_task_trace=True,
            seed=3,
        )
    )
    try:
        env.reset(seed=3)
        actions = np.zeros((1, 2), dtype=np.float32)
        env.step(actions)
        records = env.get_task_trace_records()
    finally:
        env.close()

    assert len(records) == 1
    record = records[0]
    assert record["decision_made"] is True
    assert record["requested_offload_ratio"] == 0.0
    assert record["actual_offload_ratio"] == 0.0
    assert record["execution_mode"] == "local"
    assert record["outcome"] in {"completed", "deadline_miss"}
    assert np.isfinite(record["local_compute_delay_sec"])
    assert np.isfinite(record["total_delay_sec"])
    assert np.isfinite(record["task_reward"])


def test_bimodality_summary_and_artifact_export(tmp_path):
    trace = [
        {
            "decision_made": True,
            "task_type": "light",
            "actual_offload_ratio": 0.0,
            "outcome": "completed",
            "success": True,
            "task_reward": 0.8,
            "total_delay_sec": 0.5,
            "total_energy_j": 0.1,
        },
        {
            "decision_made": True,
            "task_type": "light",
            "actual_offload_ratio": 1.0,
            "outcome": "completed",
            "success": True,
            "task_reward": 0.7,
            "total_delay_sec": 0.6,
            "total_energy_j": 0.2,
            "mec_admission_attempted": True,
            "mec_admission_accepted": True,
        },
        {
            "decision_made": True,
            "task_type": "heavy",
            "actual_offload_ratio": 0.5,
            "outcome": "deadline_miss",
            "success": False,
            "task_reward": -1.0,
            "total_delay_sec": 8.0,
            "total_energy_j": 0.3,
            "deadline_miss_reason": "mec_queue_wait",
            "mec_admission_attempted": True,
            "mec_admission_accepted": True,
        },
    ]
    methods = [{"method": "probe", "display_name": "Probe", "task_trace": trace}]

    rows = [
        {**row, "method": "probe", "display_name": "Probe", "offload_bin": offload_ratio_bin(row["actual_offload_ratio"])}
        for row in trace
    ]
    bimodality = build_bimodality_summary(rows)
    task_summary = build_task_type_offload_summary(rows)
    paths = save_task_offload_diagnostics(tmp_path, methods)

    assert bimodality[0]["strict_endpoint_share"] == 2 / 3
    assert any(
        row["task_type"] == "heavy" and row["offload_bin"] == "medium_high"
        for row in task_summary
    )
    assert paths["task_trace"].exists()
    assert paths["bimodality_summary"].exists()
    with paths["task_trace"].open(encoding="utf-8") as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == 3
