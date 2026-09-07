"""Evaluate HAN+PDQN and its fixed-discrete SCA variant in a NEW directory.

No training is performed. Existing checkpoints and source experiment folders
are read-only. Default episode count and evaluation seed offset match the
unified baseline evaluator. Independent test seeds can be supplied explicitly.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Optional, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithm.sca_adapter import FixedDiscreteSCA
from src.algorithm.sca_offload import SCAConfig
from src.environment.gym_env import build_env_config
from scripts import compare_system_baselines as compare
from scripts.train import HANPDQNTrainer, MODEL_SCHEMA_VERSION, GEOMETRY_SCHEMA_VERSION, ENVIRONMENT_SCHEMA_VERSION

SCA_METHOD = "han_pdqn_sca"
DEFAULT_EVAL_EPISODES = compare.DEFAULT_EVAL_EPISODES
DEFAULT_TEST_SEED_OFFSET = compare.EVALUATION_SEED_OFFSET


def load_source_checkpoint(checkpoint: Path) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"HAN+PDQN checkpoint not found: {checkpoint}")
    payload = compare.torch_load_trusted_checkpoint(checkpoint, map_location="cpu")
    for key, expected in (
        ("model_schema_version", MODEL_SCHEMA_VERSION),
        ("geometry_schema_version", GEOMETRY_SCHEMA_VERSION),
        ("environment_schema_version", ENVIRONMENT_SCHEMA_VERSION),
    ):
        if payload.get(key) != expected:
            raise ValueError(f"{key}={payload.get(key)}; required {expected}")
    config = dict(payload.get("config", {}))
    if config.get("algorithm") != "pdqn":
        raise ValueError("SCA requires a HAN+PDQN checkpoint, not a MAPPO or raw PDQN checkpoint")
    for key in ("han_state_dict", "q_net_state_dict", "param_nets_state_dict"):
        if key not in payload:
            raise ValueError(f"Missing checkpoint field {key}")
    return config


def reserve_output_directory(output_dir: Path, checkpoint: Path) -> Path:
    output_dir = output_dir.resolve()
    if output_dir in checkpoint.resolve().parents:
        raise ValueError("Output must not be the checkpoint directory or one of its ancestors")
    # mkdir(exist_ok=False) also prevents races and overwriting any existing run.
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _timing(values: Sequence[float]) -> dict:
    return {
        "mean_ms": float(np.mean(values)) if values else 0.0,
        "p95_ms": float(np.percentile(values, 95)) if values else 0.0,
        "max_ms": float(np.max(values)) if values else 0.0,
        "samples": len(values),
    }


def evaluate_variant(
    checkpoint: Path, source_config: dict, output_dir: Path,
    episode_seeds: Sequence[int], device: str, max_steps: Optional[int],
    sca_config: Optional[SCAConfig],
) -> tuple[dict, dict, dict]:
    method = SCA_METHOD if sca_config is not None else "han_pdqn"
    scratch = output_dir / "runtime" / method
    config = compare.train_config_from_dict(
        source_config, device=device, max_steps=max_steps, episodes=len(episode_seeds),
        save_path=scratch, exp_name=method, load_path=checkpoint,
    )
    config.log_path = str(scratch / "logs")
    trainer = HANPDQNTrainer(config)
    rewards, summaries, actions = [], [], []
    decision_times, refinement_times = [], []
    counts = Counter()
    initial_proxy = final_proxy = 0.0
    episode_diagnostics = []
    gpu = torch.device(config.device).type == "cuda"
    try:
        trainer.load_checkpoint(str(checkpoint))
        # No optimizer/update/replay calls are made in this evaluator.
        for module in (trainer.han_encoder, trainer.algorithm.q_net, trainer.algorithm.param_nets):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        with (output_dir / f"{method}_step_diagnostics.jsonl").open("x", encoding="utf-8") as log:
            for episode_index, seed in enumerate(episode_seeds, start=1):
                trainer._cached_han_user_embed = None
                trainer._cached_sat_embed = None
                trainer.env.reset(seed=int(seed))
                episode_reward = 0.0
                step = 0
                episode_counts = Counter()
                adapter = FixedDiscreteSCA(trainer.env, sca_config) if sca_config else None
                with adapter if adapter else nullcontext():
                    while True:
                        if gpu:
                            torch.cuda.synchronize()
                        start = perf_counter()
                        obs, _, masks, *_ = trainer._encode_graph_state()
                        masks = trainer._apply_pre_handover_action_mask(masks, trainer.env.get_pre_handover_mask())
                        requested, _, _ = trainer._select_eval_action(obs, masks)
                        if gpu:
                            torch.cuda.synchronize()
                        inference_ms = (perf_counter() - start) * 1000
                        kwargs = dict(return_observation=False, return_info=False)
                        if adapter:
                            outcome, executed, diagnostic = adapter.step(requested, **kwargs)
                            assert np.array_equal(executed[:, 0], requested[:, 0])
                            minimum = float(config.min_effective_offload_ratio)
                            assert np.array_equal(executed[:, 1] >= minimum, requested[:, 1] >= minimum)
                            refine_ms = diagnostic["refinement_ms"]
                            refinement_times.append(refine_ms)
                            for key in ("attempted", "changed", "iterations", "converged", "iteration_limit",
                                        "numerical_guard", "fallback", "admission_mismatch"):
                                counts[key] += diagnostic[key]
                                episode_counts[key] += diagnostic[key]
                            initial_proxy += diagnostic["proxy_cost_before"]
                            final_proxy += diagnostic["proxy_cost_after"]
                        else:
                            outcome = trainer.env.step(requested, **kwargs)
                            executed = requested.copy()
                            diagnostic = {}
                            refine_ms = 0.0
                        decision_times.append(inference_ms + refine_ms)
                        actions.append(executed.copy())
                        _, reward, terminated, truncated, _ = outcome
                        episode_reward += trainer._scalar_reward(reward)
                        step += 1
                        log.write(json.dumps({
                            "episode": episode_index, "seed": int(seed), "step": step,
                            "decision_ms": inference_ms + refine_ms,
                            "discrete_changes": 0, "execution_mode_changes": 0,
                            **diagnostic,
                        }, ensure_ascii=False) + "\n")
                        if terminated or truncated:
                            break
                rewards.append(episode_reward)
                summaries.append(trainer.env.get_stats_summary())
                episode_diagnostics.append({"episode": episode_index, "seed": int(seed), **episode_counts})
                print(f"{method}: episode {episode_index}/{len(episode_seeds)}, seed={seed}, reward={episode_reward:.3f}", flush=True)
        extra = compare.action_diagnostics(actions, float(config.min_effective_offload_ratio))
        extra.update({
            "checkpoint": str(checkpoint.resolve()), "source": "frozen_checkpoint_paired_eval",
            "evaluation_seeds": list(map(int, episode_seeds)),
            "decision_time_mean_ms": _timing(decision_times)["mean_ms"],
            "decision_time_p95_ms": _timing(decision_times)["p95_ms"],
        })
        result = compare.summarize_results(method, rewards, summaries, extra, is_system=sca_config is None)
        for record, seed in zip(result["episode_metrics"], episode_seeds):
            record["evaluation_seed"] = int(seed)
        diagnostics = {
            "method": method, "counts": dict(counts), "episodes": episode_diagnostics,
            "decision_timing": _timing(decision_times), "refinement_timing": _timing(refinement_times),
            "proxy_cost_before": initial_proxy, "proxy_cost_after": final_proxy,
            "fallback_rate": counts["fallback"] / max(counts["attempted"], 1),
            "iteration_limit_rate": counts["iteration_limit"] / max(counts["attempted"], 1),
            "discrete_changes": 0, "execution_mode_changes": 0,
        }
        return result, diagnostics, asdict(build_env_config(config))
    finally:
        trainer.env.close()
        for handler in trainer.logger.handlers[:]:
            handler.close()
            trainer.logger.removeHandler(handler)


def run_comparison(
    checkpoint: Path, output_dir: Path, *, episodes: int = DEFAULT_EVAL_EPISODES,
    evaluation_seed_offset: int = DEFAULT_TEST_SEED_OFFSET,
    evaluation_seeds: Optional[Sequence[int]] = None, device: str = "cpu",
    max_steps: Optional[int] = None, sca_config: Optional[SCAConfig] = None,
    plots: bool = True,
) -> Path:
    checkpoint = checkpoint.resolve()
    source_config = load_source_checkpoint(checkpoint)
    if episodes <= 0 or (max_steps is not None and max_steps <= 0):
        raise ValueError("episodes and max_steps must be positive")
    seeds = list(evaluation_seeds) if evaluation_seeds is not None else [
        int(source_config.get("seed", 42)) + evaluation_seed_offset + index for index in range(episodes)
    ]
    if not seeds or len(set(seeds)) != len(seeds) or any(int(s) < 0 for s in seeds):
        raise ValueError("evaluation seeds must be nonempty, unique and nonnegative")
    sca_config = sca_config or SCAConfig()
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output_dir = reserve_output_directory(output_dir, checkpoint)
    provenance = {
        "status": "running", "checkpoint": str(checkpoint), "checkpoint_sha256": checksum,
        "source_training_config": source_config, "evaluation_seeds": seeds,
        "sca_config": asdict(sca_config), "device": compare.resolve_device(device),
        "torch_threads": torch.get_num_threads(), "max_steps_override": max_steps,
        "solver": "MM with analytically solved scalar LPs and full-offload endpoint comparison",
        "local_cpu_wait_proxy_seconds": 0.0,
        "queue_wait_proxy": "floor((observable_queue_depth + admission_rank)/slots) * reference_cycles / frozen_task_rate",
        "objective": "delay + normalized_energy + soft_deadline_penalty; excludes Jain reward and future returns",
        "timing": "HAN/graph + PDQN inference + SCA refinement; excludes env.step except refinement hook",
        "source_files_modified": False,
    }
    manifest = output_dir / "sca_manifest.json"
    manifest.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    methods, diagnostics = [], []
    for variant_config in (None, sca_config):
        try:
            method, diagnostic, env_config = evaluate_variant(
                checkpoint, source_config, output_dir, seeds, device, max_steps, variant_config,
            )
        except Exception as exc:
            provenance.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            manifest.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
            raise
        methods.append(method)
        diagnostics.append(diagnostic)
    compare.save_results_json(output_dir, {
        "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION, "metric_schema_version": 2,
        "run_mode": "paired_sca_evaluation", "objective": "multi_objective",
        "best_model_metric": source_config.get("best_model_metric", "reward"),
        "compare_ranking_metric": "reward", "env_config": env_config,
        "evaluation_seeds": seeds, "methods": methods,
    })
    compare.save_results_csv(output_dir, methods)
    compare.save_episode_metrics_csv(output_dir, methods)
    (output_dir / "sca_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    if plots:
        compare.setup_publication_style()
        compare.plot_method_comparison(methods, output_dir)
        compare.plot_delay_energy_tradeoff(methods, output_dir)
        compare.plot_success_continuity_scatter(methods, output_dir)
        compare.plot_performance_radar(methods, output_dir)
        compare.plot_reward_distribution(methods, output_dir)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checksum:
        raise RuntimeError("Source checkpoint changed during evaluation")
    provenance["status"] = "complete"
    manifest.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted local HAN+PDQN checkpoint")
    parser.add_argument("--output-dir", type=Path, default=None, help="Must be a NEW directory")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES,
                        help="Evaluation episodes per method; shares the unified evaluator default (5).")
    parser.add_argument("--evaluation-seed-offset", type=int, default=DEFAULT_TEST_SEED_OFFSET)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--sca-max-iterations", type=int, default=5)
    parser.add_argument("--sca-tolerance", type=float, default=1e-6)
    parser.add_argument("--sca-deadline-penalty", type=float, default=10.0)
    parser.add_argument("--sca-queue-reference-cycles", type=float, default=2e9)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    output = args.output_dir or PROJECT_ROOT / "results/baseline_compare" / (
        "han_pdqn_sca_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    path = run_comparison(
        args.checkpoint, output, episodes=args.episodes,
        evaluation_seed_offset=args.evaluation_seed_offset, evaluation_seeds=args.evaluation_seeds,
        device=args.device, max_steps=args.max_steps, plots=not args.no_plots,
        sca_config=SCAConfig(args.sca_max_iterations, args.sca_tolerance,
                             args.sca_deadline_penalty, args.sca_queue_reference_cycles),
    )
    print(f"Saved paired HAN+PDQN / SCA comparison: {path}")


if __name__ == "__main__":
    main()
