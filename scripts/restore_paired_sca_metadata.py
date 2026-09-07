"""Restore copied comparison metadata only after a paired control reproduces it."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

METRICS = ("reward", "avg_delay", "avg_success_delay", "p95_success_delay",
           "total_energy", "total_tasks", "completed_tasks", "deadline_violations",
           "total_handovers", "task_success_rate", "service_continuity_rate")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_and_prepare(source: Path, target: Path):
    old = read_json(source / "comparison_summary.json")
    pair = read_json(target / "sca/comparison_summary.json")
    manifest = read_json(target / "sca/sca_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete SCA: {target}")
    for key in ("environment_schema_version", "metric_schema_version", "env_config"):
        if old[key] != pair[key]:
            raise ValueError(f"Historical {key} differs: {target}")
    old_control = next(m for m in old["methods"] if m["method"] == "han_pdqn")
    new_control = next(m for m in pair["methods"] if m["method"] == "han_pdqn")
    original = old_control["episode_metrics"]
    replayed = new_control["episode_metrics"]
    if len(original) != len(replayed) or len(original) != len(pair["evaluation_seeds"]):
        raise ValueError("Episode count mismatch")
    differences = {}
    for metric in METRICS:
        diffs = []
        for left, right in zip(original, replayed):
            a, b = float(left[metric]), float(right[metric])
            if not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"Paired control does not reproduce {target}: {metric} {a} vs {b}")
            diffs.append(abs(a-b))
        differences[metric] = max(diffs)
    # Verify the selected CSV really came from this historical comparison.
    with (target / "comparison_seed_records.csv").open(encoding="utf-8", newline="") as handle:
        selected = list(csv.DictReader(handle))
    methods = []
    for row in selected:
        method = deepcopy(next(m for m in old["methods"] if m["method"] == row["method"]))
        for metric in ("mean_reward", "avg_delay", "total_energy", "task_success_rate"):
            if not math.isclose(float(row[metric]), float(method[metric]), rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"Selected CSV does not match historical source: {row['method']} {metric}")
        method["source_is_system"] = method["is_system"]
        method["is_system"] = method["method"] == "han_pdqn"
        history = target / "learned_baselines" / method["method"] / "training_history.json"
        if history.exists():
            method["training_history"] = str(history.resolve())
        methods.append(method)
    checkpoint = Path(manifest["checkpoint"])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after evaluation")
    restored = deepcopy(old)
    restored.update(methods=methods, evaluation_seeds=pair["evaluation_seeds"],
                    system_checkpoint=str(checkpoint), system_checkpoint_sha256=digest,
                    system_run_dir=str(checkpoint.parent),
                    training_history=str(target / "learned_baselines/han_pdqn/training_history.json"))
    restored["metadata_recovery"] = {
        "source_summary": str((source / "comparison_summary.json").resolve()),
        "source_sha256": hashlib.sha256((source / "comparison_summary.json").read_bytes()).hexdigest(),
        "evaluation_seed_evidence": "Paired HAN+PDQN replay reproduced historical per-episode metrics in order",
        "episode_metric_max_absolute_differences": differences,
        "relative_tolerance": 1e-6, "absolute_tolerance": 1e-6,
        "historical_method_values_preserved": True,
    }
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--user-counts", type=int, nargs="+", default=[20, 25, 30, 35, 40])
    args = parser.parse_args()
    pending = []
    for users in args.user_counts:
        target = args.seed_dir / f"u{users}"
        destination = target / "comparison_summary.json"
        if destination.exists():
            raise FileExistsError(f"Will not overwrite metadata: {destination}")
        pending.append((destination, verify_and_prepare(args.source / f"u{users}", target)))
    # All groups must validate before any metadata is written.
    for destination, payload in pending:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"Verified and restored: {destination}")


if __name__ == "__main__":
    main()
