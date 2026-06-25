"""Fast satellite load/status features for candidate-attention policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.environment.gym_env import LEOSatelliteEnv


SATELLITE_LOAD_FEATURE_DIM = 8


def build_satellite_load_features(env: "LEOSatelliteEnv", num_agents: int) -> np.ndarray:
    """Build per-satellite load, visibility, link, and demand features."""
    features = np.zeros(
        (env.num_satellites, SATELLITE_LOAD_FEATURE_DIM),
        dtype=np.float32,
    )
    visible_count = np.zeros(env.num_satellites, dtype=np.float32)
    snr_sum = np.zeros(env.num_satellites, dtype=np.float32)
    rvt_sum = np.zeros(env.num_satellites, dtype=np.float32)
    demand_sum = np.zeros(env.num_satellites, dtype=np.float32)

    for user in env.user_manager.users:
        task = env.user_tasks.get(user.user_id)
        task_demand = 0.0
        if task is not None:
            data_score = float(task.data_size) / 1e8
            compute_score = float(task.computation) / 1e10
            task_demand = 0.5 * data_score + 0.5 * compute_score
        for vis in env._get_visible_satellites(user):
            sid = int(vis.sat_id)
            visible_count[sid] += 1.0
            snr_sum[sid] += env.channel.compute_snr_db(
                vis.distance_km,
                vis.elevation_deg,
            )
            rvt_sum[sid] += float(vis.rvt_seconds)
            demand_sum[sid] += task_demand

    for sid in range(env.num_satellites):
        server = env.mec_manager.get_server(sid)
        utilization = 0.0
        queue_ratio = 0.0
        connected_ratio = 0.0
        if server is not None:
            utilization = float(server.utilization)
            queue_ratio = (
                float(server.queue_length) /
                max(float(server.config.max_queue_size), 1.0)
            )
            connected_ratio = (
                float(len(server.connected_users)) /
                max(float(num_agents), 1.0)
            )
        count = max(float(visible_count[sid]), 1.0)
        features[sid] = [
            sid / max(float(env.num_satellites), 1.0),
            np.clip(utilization, 0.0, 1.0),
            np.clip(queue_ratio, 0.0, 1.0),
            np.clip(connected_ratio, 0.0, 1.0),
            np.clip(visible_count[sid] / max(float(num_agents), 1.0), 0.0, 1.0),
            np.clip(demand_sum[sid] / count, 0.0, 1.0),
            np.clip((snr_sum[sid] / count) / 50.0, 0.0, 1.0),
            np.clip((rvt_sum[sid] / count) / 600.0, 0.0, 1.0),
        ]
    return features

