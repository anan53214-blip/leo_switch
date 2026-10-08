"""Train and plot the U30 HAN+PDQN single-factor hyperparameter experiment.

Run from any directory inside a complete LEO_switch checkout. All new outputs
stay beside this script. Existing default histories are read without copying.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
RATES = (1e-4, 3e-4, 1e-3, 3e-3)
BATCHES = (128, 256, 512, 1024)
DEFAULT_RATE, DEFAULT_BATCH = 1e-3, 512
SEEDS = (42, 43, 44, 45, 46)
SCHEMAS = {"model_schema_version": 5, "geometry_schema_version": 3,
           "environment_schema_version": 13}
RUNTIME_FIELDS = {"exp_name", "device", "save_path", "log_path", "load_path",
                  "pretrained_han_path", "save_interval", "log_interval"}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tag(rate, batch):
    return f"lr{rate:.0e}_bs{batch}"


def settings():
    return [(rate, DEFAULT_BATCH) for rate in RATES] + [
        (DEFAULT_RATE, batch) for batch in BATCHES if batch != DEFAULT_BATCH]


def original_history(seed):
    return ROOT / f"results/multiseed_0907/seed{seed}/u30/learned_baselines/han_pdqn/training_history.json"


def local_path(value):
    path = Path(value)
    if path.is_file():
        return path.resolve()
    parts = str(value).replace("\\", "/").split("/")
    if "results" in parts:
        rebased = ROOT.joinpath(*parts[parts.index("results"):])
        if rebased.is_file():
            return rebased.resolve()
    return path


def curve(payload, window=1):
    points = []
    for record in payload.get("training", []):
        if record.get("partial_episode", False):
            continue
        step = record.get("total_steps", record.get("step"))
        reward = record.get("mean_reward", record.get("reward"))
        if step is None or reward is None or not math.isfinite(float(reward)):
            raise ValueError("Complete training episodes require finite step/reward values")
        points.append((int(step), float(reward)))
    if not points or any(b[0] <= a[0] for a, b in zip(points, points[1:])):
        raise ValueError("Training episode steps must be nonempty and strictly increasing")
    values = [v for _, v in points]
    return {step: sum(values[max(0, i-window+1):i+1]) / min(window, i+1)
            for i, (step, _) in enumerate(points)}


def validate_history(path, reference, seed, rate, batch):
    data = load_json(path)
    for key, value in SCHEMAS.items():
        if data.get(key) != value:
            raise ValueError(f"Schema mismatch in {path}: {key}")
    actual = data.get("config", {})
    expected = {**reference, "seed": seed, "pdqn_lr": rate, "batch_size": batch}
    differences = {k: (actual.get(k), expected.get(k)) for k in set(actual) | set(expected)
                   if k not in RUNTIME_FIELDS and actual.get(k) != expected.get(k)}
    if differences:
        raise ValueError(f"Configuration mismatch in {path}: {differences}")
    if int(data.get("summary", {}).get("total_steps", 0)) != int(reference["total_timesteps"]):
        raise ValueError(f"Incomplete training history: {path}")
    if not data.get("evaluation"):
        raise ValueError(f"Missing validation trajectory: {path}")
    for record in data["evaluation"]:
        if not math.isfinite(float(record.get("eval_mean_reward", float("nan")))):
            raise ValueError(f"Nonfinite validation reward in {path}")
    if int(data["evaluation"][-1].get("total_steps", 0)) != int(reference["total_timesteps"]):
        raise ValueError(f"Missing final validation at the full training budget: {path}")
    curve(data)
    return data


def resolve_encoder(argument, reference):
    candidates = [local_path(argument)] if argument else [
        local_path(reference["pretrained_han_path"]), ROOT / "han_pdqn/best_model_u30.pt"]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("U30 pretrained HAN checkpoint missing; supply --pretrained-han-path")


def validate_encoder(path, reference):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in SCHEMAS.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Encoder checkpoint schema mismatch: {key}")
    if not checkpoint.get("han_state_dict"):
        raise ValueError("Checkpoint has no pretrained HAN weights")
    source = local_path(reference["pretrained_han_path"])
    if source.is_file() and source.resolve() != path.resolve():
        original = torch.load(source, map_location="cpu", weights_only=False)["han_state_dict"]
        supplied = checkpoint["han_state_dict"]
        if set(original) != set(supplied) or any(
                not torch.equal(original[k].cpu(), supplied[k].cpu()) for k in original):
            raise ValueError("Supplied HAN differs from the encoder used by the reused default histories")
    elif not source.is_file():
        declared = checkpoint.get("config", {}).get("pretrained_han_path")
        if str(path) != str(source) and declared != reference["pretrained_han_path"]:
            raise ValueError("Cannot establish encoder provenance for reused default histories")
    digest = hashlib.sha256()
    for key, tensor in sorted(checkpoint["han_state_dict"].items()):
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_plan(reference, fresh_default=False):
    plan = []
    for rate, batch in settings():
        for seed in SEEDS:
            reuse = (rate, batch) == (DEFAULT_RATE, DEFAULT_BATCH) and not fresh_default
            directory = OUTPUT / tag(rate, batch) / f"seed{seed}"
            history = original_history(seed) if reuse else directory / "training_history.json"
            exists = history.is_file()
            if exists:
                validate_history(history, reference, seed, rate, batch)
            if reuse and not exists:
                raise FileNotFoundError(f"Default history missing: {history}; use --fresh-default to train it")
            if not reuse and not exists and directory.exists() and any(directory.iterdir()):
                raise ValueError(f"Incomplete run retained at {directory}; inspect it and move it aside before rerunning")
            plan.append({"setting": tag(rate, batch), "pdqn_lr": rate, "batch_size": batch,
                         "seed": seed, "history": str(history.resolve()),
                         "state": "reused" if reuse else "complete" if exists else "pending"})
    return plan


def write_manifest(reference, encoder, fingerprint, plan, state, window):
    manifest = {"state": state, "updated_at": datetime.now(timezone.utc).isoformat(),
                "method": "han_pdqn", "num_users": 30, "learning_rates": RATES,
                "batch_sizes": BATCHES, "seeds": SEEDS, "fixed_config": reference,
                "encoder_checkpoint": str(encoder), "han_weights_sha256": fingerprint,
                "smoothing_window_episodes": window, "runs": plan,
                "curve_metric": "Mean training episode reward; incomplete final episodes excluded",
                "confidence_interval": "95% percentile bootstrap over five training seeds; 2000 draws",
                "validation_note": "Validation trajectories are for hyperparameter selection, not held-out test performance"}
    (OUTPUT / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train_worker(args, reference):
    from dataclasses import fields
    from scripts.train import HANPDQNTrainer, TrainConfig
    valid = {f.name for f in fields(TrainConfig)}
    unknown = set(reference) - valid
    if unknown:
        raise ValueError(f"Saved configuration has unsupported fields: {unknown}")
    rate, batch, seed = args.worker
    rate, batch, seed = float(rate), int(batch), int(seed)
    directory = OUTPUT / tag(rate, batch) / f"seed{seed}"
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Refusing to overwrite existing run: {directory}")
    encoder = resolve_encoder(args.pretrained_han_path, reference)
    config = TrainConfig(**reference)
    config.seed, config.pdqn_lr, config.batch_size = seed, rate, batch
    config.device = args.device
    config.exp_name = f"han_pdqn_u30_{tag(rate, batch)}_seed{seed}"
    config.pretrained_han_path, config.load_path = str(encoder), None
    config.save_path = config.log_path = str(directory)
    # Keep best/final checkpoints, without redundant intermediate checkpoints.
    config.save_interval = int(config.total_timesteps) + 1
    trainer = HANPDQNTrainer(config)
    trainer.train()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(plan, reference, window):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from scripts.paper_metrics import bootstrap_mean_ci

    curves, raw, aggregate, summary = {}, [], [], []
    for rate, batch in settings():
        name = tag(rate, batch)
        runs = [r for r in plan if r["setting"] == name]
        assert len(runs) == 5
        smooth, end_train, end_validation = [], [], []
        for run in runs:
            data = validate_history(run["history"], reference, run["seed"], rate, batch)
            smooth.append(curve(data, window))
            rewards = list(curve(data).values())
            end_train.append(float(np.mean(rewards[-20:])))
            final = data["evaluation"][-1]
            end_validation.append(float(final["eval_mean_reward"]))
            for kind, records, field in [("training", data["training"], "mean_reward"),
                                         ("validation", data["evaluation"], "eval_mean_reward")]:
                for record in records:
                    if record.get("partial_episode", False):
                        continue
                    raw.append({"setting": name, "pdqn_lr": rate, "batch_size": batch,
                                "seed": run["seed"], "trajectory": kind,
                                "total_steps": record["total_steps"], "mean_reward": record[field],
                                "source_history": run["history"]})
        steps = sorted(set.intersection(*(set(c) for c in smooth)))
        if not steps:
            raise ValueError(f"No shared training steps for {name}")
        means, lows, highs = [], [], []
        for step in steps:
            mean, low, high = bootstrap_mean_ci([c[step] for c in smooth])
            means.append(mean); lows.append(low); highs.append(high)
            aggregate.append({"setting": name, "pdqn_lr": rate, "batch_size": batch,
                              "total_steps": step, "seed_count": 5, "mean_reward": mean,
                              "ci_low": low, "ci_high": high, "smoothing_window": window})
        curves[name] = (steps, means, lows, highs)
        tm, tl, th = bootstrap_mean_ci(end_train)
        vm, vl, vh = bootstrap_mean_ci(end_validation)
        summary.append({"setting": name, "pdqn_lr": rate, "batch_size": batch,
                        "seed_count": 5, "last20_training_reward": tm,
                        "last20_training_ci_low": tl, "last20_training_ci_high": th,
                        "final_validation_reward": vm, "final_validation_ci_low": vl,
                        "final_validation_ci_high": vh})

    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    colors, styles = ("#0072B2", "#009E73", "#D55E00", "#CC79A7"), ("--", "-.", "-", ":")
    for axis, cases, title in [
        (axes[0], [(r, DEFAULT_BATCH) for r in RATES], "(a) Learning rate (batch size = 512)"),
        (axes[1], [(DEFAULT_RATE, b) for b in BATCHES], "(b) Batch size (learning rate = 0.001)")]:
        for index, (rate, batch) in enumerate(cases):
            steps, means, lows, highs = curves[tag(rate, batch)]
            label = f"LR = {rate:.0e}" if axis is axes[0] else f"B = {batch}"
            default = (rate, batch) == (DEFAULT_RATE, DEFAULT_BATCH)
            color = "#D55E00" if default else colors[index]
            axis.plot(steps, means, color=color, linestyle=styles[index],
                      linewidth=2.2 if default else 1.7, label=label)
            axis.fill_between(steps, lows, highs, color=color, alpha=0.14)
        axis.set_title(title); axis.set_xlabel("Training steps")
        axis.set_xlim(0, reference["total_timesteps"])
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:g}k"))
        axis.grid(alpha=0.3, linestyle="--"); axis.legend(frameon=True)
    axes[0].set_ylabel("Mean episode reward")
    fig.tight_layout()
    fig.savefig(OUTPUT / "convergence_u30.png")
    fig.savefig(OUTPUT / "convergence_u30.pdf")
    plt.close(fig)
    write_csv(OUTPUT / "convergence_seed_records.csv", raw)
    write_csv(OUTPUT / "convergence_curves.csv", aggregate)
    write_csv(OUTPUT / "convergence_summary.csv", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--pretrained-han-path")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print plan without training or writing")
    parser.add_argument("--plot-only", action="store_true", help="Require all completed histories and plot")
    parser.add_argument("--fresh-default", action="store_true", help="Train the five default cases instead of reusing")
    parser.add_argument("--window", type=int, default=3, help="Identical trailing episode smoothing for all curves")
    parser.add_argument("--worker", nargs=3, metavar=("LR", "BATCH", "SEED"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    reference = load_json(original_history(43))["config"]
    if reference["num_users"] != 30 or reference["algorithm"] != "pdqn":
        raise ValueError("Reference configuration must be U30 HAN+PDQN")
    if (reference["pdqn_lr"], reference["batch_size"]) != (DEFAULT_RATE, DEFAULT_BATCH):
        raise ValueError("Reference configuration does not match the declared default")
    if args.worker:
        train_worker(args, reference)
        return
    plan = make_plan(reference, args.fresh_default)
    encoder = resolve_encoder(args.pretrained_han_path, reference)
    fingerprint = validate_encoder(encoder, reference)
    pending = [r for r in plan if r["state"] == "pending"]
    print(f"U30 HAN+PDQN: {len(settings())} configurations x 5 seeds = {len(plan)} runs")
    print(f"Reused: {sum(r['state']=='reused' for r in plan)}; new training needed: {len(pending)}")
    print(f"Frozen HAN: {encoder}; output: {OUTPUT}")
    for run in plan:
        print(f"{run['setting']} seed{run['seed']}: {run['state']}")
    if args.dry_run:
        return
    if args.plot_only and pending:
        raise ValueError(f"Cannot plot parameter comparisons: {len(pending)} runs have not been trained")
    if pending and args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run on the GPU server or explicitly use --device cpu")
    manifest = OUTPUT / "experiment_manifest.json"
    if manifest.exists():
        previous = load_json(manifest)
        if previous["han_weights_sha256"] != fingerprint:
            raise ValueError("Encoder changed since experiment began")
    write_manifest(reference, encoder, fingerprint, plan, "running" if pending else "plotting", args.window)
    for run in pending:
        run["state"] = "running"
        write_manifest(reference, encoder, fingerprint, plan, "running", args.window)
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--device", args.device,
                   "--pretrained-han-path", str(encoder), "--worker", str(run["pdqn_lr"]),
                   str(run["batch_size"]), str(run["seed"])]
        try:
            result = subprocess.run(command, cwd=ROOT)
        except KeyboardInterrupt:
            run["state"] = "interrupted"
            write_manifest(reference, encoder, fingerprint, plan, "interrupted", args.window)
            raise
        if result.returncode:
            run["state"] = "failed"
            write_manifest(reference, encoder, fingerprint, plan, "failed", args.window)
            raise RuntimeError(f"Training failed for {run['setting']} seed{run['seed']}; original logs retained")
        validate_history(run["history"], reference, run["seed"], run["pdqn_lr"], run["batch_size"])
        run["state"] = "complete"
        write_manifest(reference, encoder, fingerprint, plan, "running", args.window)
    plot_results(plan, reference, args.window)
    write_manifest(reference, encoder, fingerprint, plan, "complete", args.window)
    print(f"Finished: {OUTPUT / 'convergence_u30.png'}")


if __name__ == "__main__":
    main()
