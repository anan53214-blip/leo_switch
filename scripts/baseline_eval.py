"""
基线算法评估脚本

提供多种启发式基线策略，与HAN+MAPPO进行对比。

用法示例:
    python scripts/baseline_eval.py --baseline max_rvt --episodes 5
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.environment.gym_env import LEOSatelliteEnv, EnvConfig


def _select_handover_action(baseline: str, user, env, visible_sats) -> int:
    """
    基线策略选择切换动作（返回动作索引，0=不切换，k=第k个可见卫星）
    """
    if not visible_sats:
        return 0

    if baseline == "random":
        # 随机从[0..K]选择
        return int(env.rng.integers(0, len(visible_sats) + 1))

    if baseline == "stay":
        return 0

    # 选择目标卫星
    if baseline == "max_elev":
        target_idx = int(np.argmax([s.elevation_deg for s in visible_sats]))
    elif baseline == "max_rvt":
        target_idx = int(np.argmax([s.rvt_seconds for s in visible_sats]))
    elif baseline == "min_distance":
        target_idx = int(np.argmin([s.distance_km for s in visible_sats]))
    else:
        raise ValueError(f"未知基线策略: {baseline}")

    # 若目标卫星就是当前服务卫星，则不切换
    if user.serving_satellite == visible_sats[target_idx].sat_id:
        return 0

    # 动作索引从1开始
    return target_idx + 1


def _snapshot_candidates(env: LEOSatelliteEnv) -> Dict:
    """获取每个用户的候选卫星快照（用于论文描述）"""
    snapshot = {}
    for user in env.user_manager.users:
        visible_sats = env._get_visible_satellites(user)
        snapshot[user.user_id] = [
            {
                "sat_id": sat.sat_id,
                "elevation_deg": float(sat.elevation_deg),
                "distance_km": float(sat.distance_km),
                "rvt_seconds": float(sat.rvt_seconds)
            }
            for sat in visible_sats
        ]
    return snapshot


def _run_episode(env: LEOSatelliteEnv, baseline: str, offload_ratio: float) -> Tuple[float, Dict, Dict]:
    """运行单个episode并返回奖励与统计"""
    obs, info = env.reset()
    candidate_snapshot = _snapshot_candidates(env)
    done = False
    episode_reward = 0.0

    while not done:
        actions = np.zeros((env.num_users, 2), dtype=np.float32)

        for user_id, user in enumerate(env.user_manager.users):
            visible_sats = env._get_visible_satellites(user)
            handover_action = _select_handover_action(baseline, user, env, visible_sats)
            actions[user_id, 0] = handover_action
            actions[user_id, 1] = offload_ratio

        obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        episode_reward += float(reward)

    return episode_reward, info, candidate_snapshot


def evaluate_baseline(
    baseline: str,
    episodes: int,
    env_config: EnvConfig,
    offload_ratio: float,
    max_steps: int
) -> Dict:
    """评估基线策略并返回汇总统计"""
    env_config.max_steps = max_steps
    env = LEOSatelliteEnv(env_config)

    rewards: List[float] = []
    delays: List[float] = []
    energies: List[float] = []
    handover_success_rates: List[float] = []
    completion_rates: List[float] = []

    for _ in range(episodes):
        episode_reward, info, candidate_snapshot = _run_episode(env, baseline, offload_ratio)
        rewards.append(episode_reward)

        stats = info.get("stats", {})
        delays.append(stats.get("total_delay", 0.0))
        energies.append(stats.get("total_energy", 0.0))

        total_handovers = stats.get("total_handovers", 0)
        successful_handovers = stats.get("successful_handovers", 0)
        if total_handovers > 0:
            handover_success_rates.append(successful_handovers / total_handovers)
        else:
            handover_success_rates.append(1.0)

        total_tasks = stats.get("total_tasks", 0)
        completed_tasks = stats.get("completed_tasks", 0)
        if total_tasks > 0:
            completion_rates.append(completed_tasks / total_tasks)
        else:
            completion_rates.append(0.0)

    results = {
        "baseline": baseline,
        "episodes": episodes,
        "offload_ratio": offload_ratio,
        "max_steps": max_steps,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_delay": float(np.mean(delays)),
        "mean_energy": float(np.mean(energies)),
        "handover_success_rate": float(np.mean(handover_success_rates)),
        "task_completion_rate": float(np.mean(completion_rates)),
        "candidate_snapshot": candidate_snapshot,
        "env_config": asdict(env_config)
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="基线策略评估")
    parser.add_argument("--baseline", type=str, default="max_rvt",
                        choices=["random", "stay", "max_elev", "max_rvt", "min_distance"],
                        help="基线策略类型")
    parser.add_argument("--episodes", type=int, default=5, help="评估episode数")
    parser.add_argument("--offload_ratio", type=float, default=0.5, help="固定卸载比例")
    parser.add_argument("--max_steps", type=int, default=200, help="每个episode步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output_dir", type=str, default="results/baselines", help="结果保存目录")

    args = parser.parse_args()

    env_config = EnvConfig(
        num_users=5,
        max_steps=args.max_steps,
        seed=args.seed
    )

    results = evaluate_baseline(
        baseline=args.baseline,
        episodes=args.episodes,
        env_config=env_config,
        offload_ratio=args.offload_ratio,
        max_steps=args.max_steps
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"baseline_{args.baseline}_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
