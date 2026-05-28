"""
Generate comparison summaries and plots from already-trained artifacts.

This helper is for interrupted train_compare runs where learned baseline
checkpoints/history files exist, but comparison_summary.json and figures were
not written yet. It evaluates saved checkpoints where needed and reuses
training_history.json records for training curves.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_system_baselines import (
    DEFAULT_SELECTION_METRIC,
    MADDPGAlgorithm,
    MADDPGConfig,
    PRIMARY_COMPARE_METRICS,
    PDQNAlgorithm,
    PDQNConfig,
    annotate_priority_metrics,
    build_policy,
    evaluate_han_offpolicy_checkpoint,
    evaluate_maddpg_policy,
    evaluate_mappo_checkpoint_with_trainer,
    evaluate_pdqn_policy,
    evaluate_policy,
    evaluate_simple_heuristic_with_offload_search,
    evaluate_system_checkpoint,
    no_han_trainer_class_for_objective,
    order_methods,
    plot_additional_metric_curves,
    plot_delay_energy_tradeoff,
    plot_method_comparison,
    plot_paper_dashboard,
    plot_performance_radar,
    plot_reward_distribution,
    plot_step_metric_curves,
    plot_success_continuity_scatter,
    plot_training_curve_vs_baselines,
    primary_metric_leaders,
    resolve_device,
    save_episode_metrics_csv,
    save_results_csv,
    save_results_json,
    setup_publication_style,
)
from scripts.train import HANMADDPGTrainer, HANPDQNTrainer


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize_artifact_config_paths(
    config_data: Dict[str, Any],
    system_run_dir: Path,
    compare_dir: Path,
) -> Dict[str, Any]:
    sanitized = dict(config_data)
    sanitized["save_path"] = str(system_run_dir)
    sanitized["log_path"] = str(compare_dir / "logs")
    return sanitized


def evaluate_maddpg_checkpoint(
    checkpoint: Path,
    config_data: Dict[str, Any],
    objective: str,
    episodes: int,
    seed: int,
    max_steps: int,
    device_name: str,
) -> Dict[str, Any]:
    device = torch.device(resolve_device(device_name))
    payload = torch.load(checkpoint, map_location=device)
    cfg = dict(payload.get("config", {}))
    cfg["device"] = str(device)
    cfg.setdefault("num_agents", int(payload.get("num_agents", config_data.get("num_users", 20))))
    cfg.setdefault("obs_dim", int(payload.get("obs_dim", 69)))
    if "max_candidates" not in cfg and "handover_dim" in payload:
        cfg["max_candidates"] = int(payload["handover_dim"]) - 1
    algo = MADDPGAlgorithm(MADDPGConfig(**cfg))
    algo.actor.load_state_dict(payload["actor_state_dict"])
    algo.critic.load_state_dict(payload["critic_state_dict"])
    algo.target_actor.load_state_dict(payload.get("target_actor_state_dict", payload["actor_state_dict"]))
    algo.target_critic.load_state_dict(payload.get("target_critic_state_dict", payload["critic_state_dict"]))
    algo.train_step = int(payload.get("train_step", 0))
    result = evaluate_maddpg_policy(
        algorithm=algo,
        objective=objective,
        config=config_data,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
    )
    result["checkpoint"] = str(checkpoint)
    result["training_history"] = str(checkpoint.parent / "training_history.json")
    result["source"] = "checkpoint_eval"
    return result


def evaluate_pdqn_checkpoint(
    checkpoint: Path,
    config_data: Dict[str, Any],
    objective: str,
    episodes: int,
    seed: int,
    max_steps: int,
    device_name: str,
) -> Dict[str, Any]:
    device = resolve_device(device_name)
    payload = torch.load(checkpoint, map_location=torch.device(device))
    cfg = dict(payload.get("config", {}))
    cfg["device"] = device
    algo = PDQNAlgorithm(PDQNConfig(**cfg))
    algo.load(checkpoint)
    result = evaluate_pdqn_policy(
        algorithm=algo,
        objective=objective,
        config=config_data,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
    )
    result["checkpoint"] = str(checkpoint)
    result["training_history"] = str(checkpoint.parent / "training_history.json")
    result["source"] = "checkpoint_eval"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-run-dir", required=True)
    parser.add_argument("--compare-dir", required=True)
    parser.add_argument("--objective", default="multi_objective", choices=["multi_objective", "delay_only", "energy_only"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-users", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metric", default=DEFAULT_SELECTION_METRIC)
    parser.add_argument("--plot-window", type=int, default=5)
    args = parser.parse_args()

    setup_publication_style()
    system_run_dir = Path(args.system_run_dir)
    compare_dir = Path(args.compare_dir)
    learned_dir = compare_dir / "learned_baselines"
    system_history = system_run_dir / "training_history.json"
    system_checkpoint = system_run_dir / "best_model.pt"

    config_payload = load_json(system_history)
    config_data = sanitize_artifact_config_paths(
        dict(config_payload.get("config", {})),
        system_run_dir=system_run_dir,
        compare_dir=compare_dir,
    )
    config_data["num_users"] = int(args.num_users)
    config_data["max_steps"] = int(args.max_steps)
    config_data["best_model_metric"] = args.metric

    methods: List[Dict[str, Any]] = []
    methods.append(
        evaluate_system_checkpoint(
            checkpoint=system_checkpoint,
            config_data=config_data,
            objective=args.objective,
            episodes=args.episodes,
            device=resolve_device(args.device),
            max_steps=args.max_steps,
        )
    )
    methods[-1]["source"] = "checkpoint_eval"
    methods[-1]["training_history"] = str(system_history)

    for baseline_name in ("random", "min_distance"):
        result = evaluate_simple_heuristic_with_offload_search(
            strategy=baseline_name,
            objective=args.objective,
            config=config_data,
            episodes=args.episodes,
            seed=args.seed,
            max_steps=args.max_steps,
            offload_grid=[0.0, 0.5, 1.0],
            selection_metric_name=args.metric,
        )
        result["source"] = "heuristic_eval"
        methods.append(result)

    for baseline_name in ("full_local", "joint_greedy"):
        policy = build_policy(
            name=baseline_name,
            objective=args.objective,
            fixed_offload=0.0,
            joint_offload_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
        result = evaluate_policy(
            policy=policy,
            objective=args.objective,
            config=config_data,
            episodes=args.episodes,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        result["source"] = "heuristic_eval"
        methods.append(result)

    methods.append(
        evaluate_maddpg_checkpoint(
            learned_dir / "maddpg" / "maddpg_model.pt",
            config_data,
            args.objective,
            args.episodes,
            args.seed,
            args.max_steps,
            args.device,
        )
    )
    methods.append(
        evaluate_pdqn_checkpoint(
            learned_dir / "pdqn" / "pdqn_model.pt",
            config_data,
            args.objective,
            args.episodes,
            args.seed,
            args.max_steps,
            args.device,
        )
    )
    methods.append(
        evaluate_mappo_checkpoint_with_trainer(
            checkpoint=learned_dir / "mappo_no_han" / "best_model.pt",
            config_data=config_data,
            episodes=args.episodes,
            device=resolve_device(args.device),
            max_steps=args.max_steps,
            trainer_cls=no_han_trainer_class_for_objective(args.objective),
            method_name="mappo_no_han",
            is_system=False,
        )
    )
    methods[-1]["training_history"] = str(learned_dir / "mappo_no_han" / "training_history.json")
    methods[-1]["checkpoint"] = str(learned_dir / "mappo_no_han" / "best_model.pt")

    methods.append(
        evaluate_han_offpolicy_checkpoint(
            checkpoint=learned_dir / "han_maddpg" / "best_model.pt",
            config_data=config_data,
            episodes=args.episodes,
            device=resolve_device(args.device),
            max_steps=args.max_steps,
            trainer_cls=HANMADDPGTrainer,
            method_name="han_maddpg",
        )
    )
    methods[-1]["training_history"] = str(learned_dir / "han_maddpg" / "training_history.json")
    methods[-1]["checkpoint"] = str(learned_dir / "han_maddpg" / "best_model.pt")

    methods.append(
        evaluate_han_offpolicy_checkpoint(
            checkpoint=learned_dir / "han_pdqn" / "best_model.pt",
            config_data=config_data,
            episodes=args.episodes,
            device=resolve_device(args.device),
            max_steps=args.max_steps,
            trainer_cls=HANPDQNTrainer,
            method_name="han_pdqn",
        )
    )
    methods[-1]["training_history"] = str(learned_dir / "han_pdqn" / "training_history.json")
    methods[-1]["checkpoint"] = str(learned_dir / "han_pdqn" / "best_model.pt")

    methods = annotate_priority_metrics(methods, metric_name=args.metric)
    methods = order_methods(methods)
    leaders = primary_metric_leaders(methods)

    json_path = save_results_json(
        compare_dir,
        {
            "generated_at": "from_existing_artifacts",
            "run_mode": "artifact_plot_only",
            "objective": args.objective,
            "best_model_metric": args.metric,
            "compare_ranking_metric": args.metric,
            "primary_compare_metrics": [label for _, label in PRIMARY_COMPARE_METRICS],
            "primary_metric_leaders": leaders,
            "system_run_dir": str(system_run_dir),
            "system_checkpoint": str(system_checkpoint),
            "training_history": str(system_history),
            "total_timesteps": int(config_data.get("total_timesteps", 0)),
            "methods": methods,
        },
    )
    csv_path = save_results_csv(compare_dir, methods)
    episode_csv_path = save_episode_metrics_csv(compare_dir, methods)
    metrics_plot = plot_method_comparison(methods, compare_dir)
    reward_curve_plot = plot_training_curve_vs_baselines(system_history, methods, compare_dir, window=args.plot_window)
    step_metric_plots = plot_step_metric_curves(system_history, compare_dir, window=args.plot_window)
    episode_metric_plot = plot_additional_metric_curves(methods, compare_dir)
    tradeoff_plot = plot_delay_energy_tradeoff(methods, compare_dir)
    reliability_plot = plot_success_continuity_scatter(methods, compare_dir)
    radar_plot = plot_performance_radar(methods, compare_dir)
    reward_distribution_plot = plot_reward_distribution(methods, compare_dir)
    dashboard_plot = plot_paper_dashboard(system_history, methods, compare_dir, window=args.plot_window)

    print(f"Summary JSON saved to: {json_path}")
    print(f"Summary CSV saved to: {csv_path}")
    if episode_csv_path is not None:
        print(f"Episode metrics CSV saved to: {episode_csv_path}")
    for path in [
        metrics_plot,
        reward_curve_plot,
        episode_metric_plot,
        tradeoff_plot,
        reliability_plot,
        radar_plot,
        reward_distribution_plot,
        dashboard_plot,
        *step_metric_plots,
    ]:
        if path is not None:
            print(f"Figure: {path}")


if __name__ == "__main__":
    main()
