"""Matched PDQN / from-scratch HAN+PDQN experiments (protocol 2026-11-06).

Run from the repository root, for example::

    python scripts/run_matched_pdqn_suite.py --dry-run
    python scripts/run_matched_pdqn_suite.py --device cuda

This is a new experiment entry point. Historical training entry points and
results retain their original meanings. No pretrained checkpoint is required.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import platform
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_system_baselines import (
    per_user_training_rewards, scalar_reward_value, summarize_results,
)
from scripts.paper_metrics import bootstrap_mean_ci
from scripts.train import (
    ENVIRONMENT_SCHEMA_VERSION, GEOMETRY_SCHEMA_VERSION, MODEL_SCHEMA_VERSION,
    HANMAPPOTrainer, TrainConfig,
)
from src.algorithm.pdqn import PDQNAlgorithm, PDQNConfig
from src.environment.gym_env import LEOSatelliteEnv, EnvConfig, build_env_config
from src.graph.builder import HeteroGraphBuilder, HeteroGraphData
from src.model.hetero_gnn import HANConfig, HANEncoder

PROTOCOL_VERSION = "matched_pdqn_v1_20261106"
METHODS = {"pdqn_matched": "PDQN (matched)", "han_pdqn_joint": "HAN+PDQN (joint)"}
METRICS = ("mean_reward", "avg_success_delay", "deadline_violation_rate",
           "energy_per_successful_task", "task_success_rate", "blocked_time_ratio",
           "handovers_per_user_minute", "total_energy")
LOGGER = logging.getLogger("matched_pdqn")


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path: Path, data) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def training_seed(seed: int, episode: int) -> int:
    # Three disjoint namespaces, not overlapping sliding windows of seeds.
    if not 0 <= seed < 1000 or not 0 <= episode < 1_000_000:
        raise ValueError("Training seeds must be in [0,999] and episode index < 1,000,000")
    return seed * 1_000_000 + episode


@dataclass(frozen=True)
class Snapshot:
    raw: np.ndarray
    masks: np.ndarray
    graph: HeteroGraphData | None

    @classmethod
    def capture(cls, env, builder=None):
        graph = builder.build(env) if builder is not None else None
        if graph is not None:
            # Own every array: replay must not reference mutable simulator state.
            graph = HeteroGraphData(
                node_features={k: np.asarray(v, dtype=np.float32).copy() for k, v in graph.node_features.items()},
                edge_index={k: tuple(np.asarray(x, dtype=np.int64).copy() for x in v) for k, v in graph.edge_index.items()},
                edge_features={k: np.asarray(v, dtype=np.float32).copy() for k, v in graph.edge_features.items()},
                num_nodes=dict(graph.num_nodes),
            )
        return cls(np.asarray(env._get_observation(), dtype=np.float32).copy(),
                   env.get_handover_action_mask(env.max_visible_sats, apply_pre_handover_gate=True).astype(bool).copy(),
                   graph)


class SnapshotReplay:
    """O(1) ring access; adjacent transitions share immutable graph snapshots."""
    def __init__(self, capacity: int, seed: int):
        self.capacity, self.position = capacity, 0
        self.data = []
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.data)

    def add(self, state, actions, rewards, next_state, terminated):
        item = (state, np.asarray(actions, dtype=np.float32).copy(),
                np.asarray(rewards, dtype=np.float32).copy(), next_state, bool(terminated))
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, count):
        return [self.data[i] for i in self.rng.sample(range(len(self.data)), count)]

    def state_dict(self):
        states, ids, transitions = [], {}, []
        def index(state):
            key = id(state)
            if key not in ids:
                ids[key] = len(states)
                states.append({"raw": state.raw, "masks": state.masks, "graph": state.graph})
            return ids[key]
        for state, actions, rewards, nxt, terminated in self.data:
            transitions.append((index(state), actions, rewards, index(nxt), terminated))
        return {"capacity": self.capacity, "position": self.position, "rng": self.rng.getstate(),
                "states": states, "transitions": transitions}

    def load_state_dict(self, payload):
        if payload["capacity"] != self.capacity:
            raise ValueError("Replay capacity changed")
        states = [Snapshot(**item) for item in payload["states"]]
        self.data = [(states[a], action, reward, states[b], done)
                     for a, action, reward, b, done in payload["transitions"]]
        self.position = payload["position"]
        self.rng.setstate(payload["rng"])


class MatchedPDQN(PDQNAlgorithm):
    """Same PDQN update for both methods; optional trainable HAN and target HAN.

    HAN accumulates gradients from TD and parameter losses, then takes ONE Adam
    step. It is not in either head optimizer. Microbatches accumulate the full
    replay-batch mean; they do not increase the update-to-data ratio.
    """
    def __init__(self, config: PDQNConfig, han_config: HANConfig | None, encoder_lr: float, microbatch: int):
        super().__init__(config)
        self.han_config = han_config
        self.encoder = HANEncoder(han_config).to(self.device) if han_config else None
        self.target_encoder = copy.deepcopy(self.encoder).eval() if self.encoder else None
        if self.target_encoder is not None:
            self.target_encoder.requires_grad_(False)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr) if self.encoder else None
        self.microbatch = microbatch

    def encode(self, states, target=False):
        raw = torch.as_tensor(np.stack([s.raw for s in states]), device=self.device)
        encoder = self.target_encoder if target else self.encoder
        if encoder is None:
            return raw.flatten(0, 1)
        if any(s.graph is None for s in states):
            raise ValueError("Joint HAN training requires raw graph snapshots")
        # Reuse the existing graph batching contract, without constructing a
        # MAPPO trainer. Its per-graph semantic attention prevents cross-sample mixing.
        context = SimpleNamespace(device=self.device, han_encoder=encoder)
        embeddings = HANMAPPOTrainer._encode_graph_snapshots(context, [s.graph for s in states])
        return torch.cat((raw, torch.stack(embeddings["user"])), dim=-1).flatten(0, 1)

    def act_state(self, state, epsilon=0.0):
        with torch.no_grad():
            obs = self.encode([state]).cpu().numpy()
        return self.act(obs, state.masks, epsilon=epsilon)

    def _batch(self, records):
        states, actions, rewards, next_states, terminal = zip(*records)
        count = len(records) * self.num_agents
        tensor = lambda value, dtype=torch.float32: torch.as_tensor(np.asarray(value), dtype=dtype, device=self.device)
        actions = tensor(actions).reshape(count, self.action_feature_dim)
        indices = actions[:, :self.handover_dim].argmax(-1)
        return (states, next_states, tensor(rewards).reshape(count),
                tensor(terminal).repeat_interleave(self.num_agents), indices,
                F.one_hot(indices, self.handover_dim).float(), actions[:, -1],
                tensor([s.masks for s in states], torch.bool).reshape(count, self.handover_dim),
                tensor([s.masks for s in next_states], torch.bool).reshape(count, self.handover_dim))

    def update(self, replay):
        if len(replay) < self.config.batch_size:
            return {}
        samples = replay.sample(self.config.batch_size)
        chunks = [samples[i:i + self.microbatch] for i in range(0, len(samples), self.microbatch)]
        self.q_optimizer.zero_grad(set_to_none=True)
        if self.encoder_optimizer:
            self.encoder_optimizer.zero_grad(set_to_none=True)
        q_total = param_total = 0.0
        for records in chunks:
            states, nxt, rewards, terminal, indices, one_hot, offload, masks, next_masks = self._batch(records)
            obs = self.encode(states)
            with torch.no_grad():
                online_next = self.encode(nxt)
                target_next = self.encode(nxt, target=True)
                online_q, _ = self._all_action_q(self.q_net, self.param_nets, online_next)
                next_actions = online_q.masked_fill(~next_masks, -1e9).argmax(-1)
                next_params = self.target_param_nets(target_next)
                selected_params = next_params.gather(1, next_actions[:, None]).squeeze(1)
                target_q = self.target_q_net(target_next, F.one_hot(next_actions, self.handover_dim).float(), selected_params)
                target = rewards + self.config.gamma * (1.0 - terminal) * target_q
            loss = F.mse_loss(self.q_net(obs, one_hot, offload), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite TD loss")
            weight = len(records) / len(samples)
            (weight * loss).backward()
            q_total += weight * loss.item()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.grad_clip_norm, error_if_nonfinite=True)
        self.q_optimizer.step()
        self.q_net.requires_grad_(False)
        self.param_optimizer.zero_grad(set_to_none=True)
        try:
            for records in chunks:
                states, _, _, _, indices, _, offload, masks, _ = self._batch(records)
                obs = self.encode(states)
                q_values, params = self._all_action_q(self.q_net, self.param_nets, obs)
                objective = q_values.masked_fill(~masks, -1e9).max(-1).values.mean()
                bc_loss = F.mse_loss(params.gather(1, indices[:, None]).squeeze(1), offload)
                loss = -self.config.param_loss_coef * objective + self.config.bc_loss_coef * bc_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite parameter loss")
                weight = len(records) / len(samples)
                (weight * loss).backward()
                param_total += weight * loss.item()
            nn.utils.clip_grad_norm_(self.param_nets.parameters(), self.config.grad_clip_norm, error_if_nonfinite=True)
            self.param_optimizer.step()
            encoder_norm = 0.0
            if self.encoder_optimizer:
                encoder_norm = float(nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config.grad_clip_norm, error_if_nonfinite=True))
                self.encoder_optimizer.step()
        finally:
            self.q_net.requires_grad_(True)
        self.train_step += 1
        if self.train_step % self.config.target_update_interval == 0:
            self.q_net_to_targets()
        return {"q_loss": q_total, "param_loss": param_total, "han_grad_norm": encoder_norm,
                "epsilon": self.current_epsilon(), "optimizer_steps": self.train_step}

    def q_net_to_targets(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_param_nets.load_state_dict(self.param_nets.state_dict())
        if self.encoder:
            self.target_encoder.load_state_dict(self.encoder.state_dict())

    def state_dict(self):
        return {"pdqn_config": asdict(self.config),
                "han_config": asdict(self.han_config) if self.han_config else None,
                "modules": {name: getattr(self, name).state_dict() for name in
                            ("q_net", "target_q_net", "param_nets", "target_param_nets")},
                "encoder": self.encoder.state_dict() if self.encoder else None,
                "target_encoder": self.target_encoder.state_dict() if self.encoder else None,
                "optimizers": {"q": self.q_optimizer.state_dict(), "param": self.param_optimizer.state_dict(),
                               "encoder": self.encoder_optimizer.state_dict() if self.encoder_optimizer else None},
                "train_step": self.train_step, "action_rng": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, data):
        expected = asdict(self.config); actual = dict(data["pdqn_config"])
        expected.pop("device"); actual.pop("device")
        if expected != actual or data["han_config"] != (asdict(self.han_config) if self.han_config else None):
            raise ValueError("Checkpoint architecture/effective settings differ")
        for name, state in data["modules"].items():
            getattr(self, name).load_state_dict(state)
        self.q_optimizer.load_state_dict(data["optimizers"]["q"])
        self.param_optimizer.load_state_dict(data["optimizers"]["param"])
        if self.encoder:
            self.encoder.load_state_dict(data["encoder"])
            self.target_encoder.load_state_dict(data["target_encoder"])
            self.encoder_optimizer.load_state_dict(data["optimizers"]["encoder"])
        self.train_step = data["train_step"]
        self.rng.bit_generator.state = data["action_rng"]


@contextmanager
def isolated_evaluation(learner):
    global_rng = capture_rng()
    action_rng = copy.deepcopy(learner.rng.bit_generator.state)
    was_training = learner.encoder.training if learner.encoder else False
    if learner.encoder:
        learner.encoder.eval()
    try:
        yield
    finally:
        restore_rng(global_rng)
        learner.rng.bit_generator.state = action_rng
        if learner.encoder:
            learner.encoder.train(was_training)


def evaluate(learner, env_config, seeds, method):
    rewards, summaries = [], []
    with isolated_evaluation(learner):
        env = LEOSatelliteEnv(env_config)
        builder = HeteroGraphBuilder() if learner.encoder else None
        try:
            for seed in seeds:
                env.reset(seed=seed)
                reward_sum = 0.0
                while True:
                    state = Snapshot.capture(env, builder)
                    actions, _, _ = learner.act_state(state, epsilon=0.0)
                    _, reward, terminated, truncated, _ = env.step(actions, return_observation=False, return_info=False)
                    reward_sum += scalar_reward_value(reward)
                    if terminated or truncated:
                        break
                rewards.append(reward_sum)
                summaries.append(env.get_stats_summary())
        finally:
            env.close()
    result = summarize_results(method, rewards, summaries)
    result.update(display_name=METHODS[method], evaluation_seeds=list(seeds))
    for row, seed in zip(result["episode_metrics"], seeds):
        row["scenario_seed"] = seed
    return result


def make_learner(protocol, method, seed, users, device):
    env_data = dict(protocol["environment"], num_users=users, seed=seed)
    env_config = EnvConfig(**env_data)
    env = LEOSatelliteEnv(env_config)
    raw_dim, max_candidates = env.user_obs_dim, env.max_visible_sats
    env.close()
    settings = dict(protocol["pdqn"])
    settings.update(num_agents=users, obs_dim=raw_dim + (protocol["han"]["out_dim"] if method == "han_pdqn_joint" else 0),
                    max_candidates=max_candidates, seed=seed, device=device)
    han = HANConfig(**protocol["han"]) if method == "han_pdqn_joint" else None
    return MatchedPDQN(PDQNConfig(**settings), han, protocol["encoder_lr"], protocol["microbatch_size"]), env_config


def load_checkpoint(path, protocol_hash):
    # Only load this program's trusted local outputs. Recovery includes numpy
    # arrays / RNG tuples and cannot be loaded with weights_only=True.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol_hash") != protocol_hash:
        raise ValueError(f"Checkpoint protocol mismatch: {path}")
    return payload


def train_one(protocol, method, seed, users, destination, device, resume=False):
    protocol_hash = fingerprint(protocol)
    state_path = destination / "run_state.json"
    if state_path.exists():
        previous = read_json(state_path)
        if previous["protocol_hash"] != protocol_hash:
            raise ValueError(f"Different protocol already exists: {destination}")
        if previous["status"] == "trained":
            for name, expected in previous["sha256"].items():
                if digest(destination / name) != expected:
                    raise ValueError(f"Completed output changed: {destination / name}")
            LOGGER.info("Verified completed training; skipping %s", destination)
            return
        if not resume:
            raise FileExistsError(f"Incomplete run: {destination}. Use --resume to restore its full replay/RNG checkpoint.")
    elif destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Unrecognized nonempty directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    learner, env_config = make_learner(protocol, method, seed, users, device)
    env = LEOSatelliteEnv(env_config)
    builder = HeteroGraphBuilder() if learner.encoder else None
    replay = SnapshotReplay(protocol["pdqn"]["replay_size"], seed)
    progress = {"total_steps": 0, "episodes": 0, "best_reward": None, "best_step": None,
                "training_seconds": 0.0, "training": [], "validation": []}
    recovery = destination / "resume.pt"
    def checkpoint():
        return {"protocol_hash": protocol_hash, "method": method, "seed": seed, "num_users": users,
                "total_steps": progress["total_steps"], "training_device": device,
                "torch_version": torch.__version__, "learner": learner.state_dict()}
    def save_recovery():
        best = load_checkpoint(destination / "best_model.pt", protocol_hash) if progress["best_step"] is not None else None
        atomic_torch(recovery, {**checkpoint(), "progress": progress, "replay": replay.state_dict(),
                                "rng": capture_rng(), "best_checkpoint": best})
    if state_path.exists() and resume:
        payload = load_checkpoint(recovery, protocol_hash)
        if payload["training_device"] != device or payload["torch_version"] != torch.__version__:
            raise ValueError("Resume training on the original device type and PyTorch version; evaluation may use another device")
        learner.load_state_dict(payload["learner"])
        replay.load_state_dict(payload["replay"])
        progress = payload["progress"]
        if payload["best_checkpoint"] is not None:
            atomic_torch(destination / "best_model.pt", payload["best_checkpoint"])
        restore_rng(payload["rng"])
        LOGGER.info("Resuming %s at step %d with %d replay transitions", method, progress["total_steps"], len(replay))
    else:
        save_recovery()  # Exact initial state, even if interrupted before first episode.
    atomic_json(state_path, {"status": "running", "protocol_hash": protocol_hash, "method": method,
                             "seed": seed, "num_users": users, "device": device})
    atomic_json(destination / "effective_config.json", {
        "protocol_hash": protocol_hash, "environment": asdict(env_config),
        "pdqn": asdict(learner.config), "han": asdict(learner.han_config) if learner.han_config else None,
        "encoder_lr": protocol["encoder_lr"] if learner.encoder else None,
        "pretrained_han_path": None, "encoder_update": "TD plus parameter loss; one encoder optimizer step" if learner.encoder else None,
        "parameter_count": sum(p.numel() for module in (learner.q_net, learner.param_nets, learner.encoder) if module is not None for p in module.parameters()),
    })
    start = time.perf_counter()
    LOGGER.info("Starting %s U%d seed%d on %s: batch=%d, replay=%d, target interval=%d", method, users, seed,
                device, learner.config.batch_size, learner.config.replay_size, learner.config.target_update_interval)
    carried_seconds = progress["training_seconds"]
    last_recovery_step = progress["total_steps"]
    steps = protocol["total_timesteps"]
    try:
        while progress["total_steps"] < steps:
            episode_index = progress["episodes"]
            scenario = training_seed(seed, episode_index)
            env.reset(seed=scenario)
            state = Snapshot.capture(env, builder)
            reward_sum, length, stats = 0.0, 0, {}
            while progress["total_steps"] < steps:
                if progress["total_steps"] < learner.config.warmup_steps:
                    actions, features, _ = learner.random_actions(state.masks)
                else:
                    actions, features, _ = learner.act_state(state, learner.current_epsilon())
                _, reward, terminated, truncated, _ = env.step(actions, return_observation=False, return_info=False)
                if truncated and not terminated:
                    # The simulator suppresses next-decision arrivals at the
                    # reporting horizon. Restore that continuation ONLY for
                    # training bootstrap snapshots, never for test accounting.
                    env._generate_tasks()
                nxt = Snapshot.capture(env, builder)
                replay.add(state, features, per_user_training_rewards(env, reward), nxt, terminated)
                if len(replay) >= max(learner.config.batch_size, learner.config.warmup_steps):
                    stats = learner.update(replay)
                state = nxt
                reward_sum += scalar_reward_value(reward)
                length += 1
                progress["total_steps"] += 1
                current = progress["total_steps"]
                if current % protocol["eval_interval"] == 0 or current == steps:
                    validation = evaluate(learner, env_config, protocol["validation_seeds"], method)
                    score = validation["mean_reward"]
                    progress["validation"].append({"total_steps": current, "mean_reward": score,
                                                   "episode_rewards": [x["reward"] for x in validation["episode_metrics"]]})
                    if progress["best_reward"] is None or score > progress["best_reward"]:
                        progress["best_reward"], progress["best_step"] = score, current
                        atomic_torch(destination / "best_model.pt", checkpoint())
                    LOGGER.info("%s U%d seed%d validation step=%d reward=%.4f best=%.4f", method, users, seed, current, score, progress["best_reward"])
                if terminated or truncated or current == steps:
                    record = {"total_steps": current, "episode": episode_index + 1, "scenario_seed": scenario,
                              "mean_reward": reward_sum, "length": length,
                              "partial_episode": not (terminated or truncated), **stats}
                    progress["training"].append(record)
                    progress["episodes"] += 1
                    progress["training_seconds"] = carried_seconds + time.perf_counter() - start
                    if terminated or truncated:
                        if current - last_recovery_step >= protocol["save_interval"]:
                            save_recovery()
                            last_recovery_step = current
                    if progress["episodes"] % protocol["log_interval_episodes"] == 0 or current == steps:
                        LOGGER.info("%s U%d seed%d step=%d/%d episode_reward=%.4f replay=%d", method, users, seed, current, steps, reward_sum, len(replay))
                    break
        atomic_torch(destination / "final_model.pt", checkpoint())
        progress["training_seconds"] = carried_seconds + time.perf_counter() - start
        atomic_json(destination / "training_history.json", {
            "protocol_hash": protocol_hash, "method": method, "seed": seed, "num_users": users,
            "config": read_json(destination / "effective_config.json"),
            "training": progress["training"], "validation": progress["validation"],
            "summary": {k: v for k, v in progress.items() if k not in ("training", "validation")},
        })
        atomic_json(state_path, {"status": "trained", "protocol_hash": protocol_hash,
                                "method": method, "seed": seed, "num_users": users, "device": device,
                                "sha256": {name: digest(destination / name) for name in
                                           ("best_model.pt", "final_model.pt", "training_history.json", "effective_config.json")}})
    except BaseException as exc:
        atomic_json(state_path, {"status": "interrupted", "protocol_hash": protocol_hash,
                                "method": method, "seed": seed, "num_users": users, "device": device,
                                "recovery_checkpoint": str(recovery), "error": repr(exc)})
        raise
    finally:
        env.close()


def test_one(protocol, method, seed, users, destination, device):
    protocol_hash = fingerprint(protocol)
    run = read_json(destination / "run_state.json")
    if run["status"] != "trained" or run["protocol_hash"] != protocol_hash:
        raise ValueError(f"Training not complete: {destination}")
    model = destination / "best_model.pt"
    model_hash = digest(model)
    if model_hash != run["sha256"]["best_model.pt"]:
        raise ValueError(f"Selected checkpoint changed: {model}")
    result_path = destination / "test_results.json"
    if result_path.exists():
        old = read_json(result_path)
        if old["protocol_hash"] != protocol_hash or old["checkpoint_sha256"] != model_hash:
            raise ValueError(f"Existing test result belongs to another checkpoint/protocol: {result_path}")
        return old
    set_seed(seed)
    learner, env_config = make_learner(protocol, method, seed, users, device)
    payload = load_checkpoint(model, protocol_hash)
    learner.load_state_dict(payload["learner"])
    result = evaluate(learner, env_config, protocol["test_seeds"], method)
    result.update(protocol_hash=protocol_hash, seed=seed, num_users=users,
                  checkpoint_sha256=model_hash, checkpoint_step=payload["total_steps"],
                  split="test", training_history=str((destination / "training_history.json").resolve()))
    atomic_json(result_path, result)
    LOGGER.info("%s U%d seed%d TEST reward=%.4f", method, users, seed, result["mean_reward"])
    return result


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_name(path.name + ".tmp")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def aggregate(protocol, output, methods, users, seeds):
    records, summaries = [], []
    for n in users:
        for method in methods:
            group = []
            for seed in seeds:
                path = output / f"seed{seed}" / f"u{n}" / "learned_baselines" / method / "test_results.json"
                row = read_json(path)  # Never silently average an incomplete subset.
                if (row["protocol_hash"] != fingerprint(protocol) or row["evaluation_seeds"] != protocol["test_seeds"]
                        or row["method"] != method or row["seed"] != seed or row["num_users"] != n):
                    raise ValueError(f"Incompatible test result: {path}")
                group.append(row)
                records.append({"method": method, "num_users": n, "seed": seed,
                                **{k: row[k] for k in METRICS}})
            summary = {"method": method, "num_users": n, "seed_count": len(group)}
            for key in METRICS:
                mean, low, high = bootstrap_mean_ci([row[key] for row in group])
                summary.update({key: mean, key + "_ci_low": low, key + "_ci_high": high})
            summaries.append(summary)
    write_csv(output / "multiuser_seed_records.csv", records)
    write_csv(output / "multiuser_summary.csv", summaries)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    for axis, metric, label in zip(axes, METRICS[1:4], ("Successful-task delay (s)", "Deadline violation rate", "Energy per successful task (J)")):
        for method in methods:
            group = [r for r in summaries if r["method"] == method]
            x = [r["num_users"] for r in group]; y = np.array([r[metric] for r in group])
            low = np.array([r[metric + "_ci_low"] for r in group]); high = np.array([r[metric + "_ci_high"] for r in group])
            axis.errorbar(x, y, yerr=np.maximum(0, np.array([y-low, high-y])), marker="o", capsize=3, label=METHODS[method])
        axis.set(xlabel="Users", ylabel=label, xticks=users); axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.savefig(output / "matched_core_metrics.png", dpi=200); plt.close(fig)
    fig, axes = plt.subplots(1, len(users), figsize=(4 * len(users), 3.4), squeeze=False, constrained_layout=True)
    for axis, n in zip(axes.flat, users):
        for method in methods:
            histories = [read_json(output / f"seed{s}" / f"u{n}" / "learned_baselines" / method / "training_history.json") for s in seeds]
            curves = [{r["total_steps"]: r["mean_reward"] for r in h["training"] if not r["partial_episode"]} for h in histories]
            if not curves or not curves[0]:
                continue
            if any(set(curve) != set(curves[0]) for curve in curves):
                raise ValueError("Training curves have different complete-episode positions")
            x = sorted(curves[0])
            points = np.asarray([bootstrap_mean_ci([curve[t] for curve in curves]) for t in x])
            axis.plot(np.asarray(x) / protocol["environment"]["max_steps"], points[:, 0], label=METHODS[method])
            axis.fill_between(np.asarray(x) / protocol["environment"]["max_steps"], points[:, 1], points[:, 2], alpha=.15)
        axis.set(title=f"{n} users", xlabel="Complete training episodes", ylabel="Mean episode reward")
        axis.grid(alpha=.25)
    axes.flat[0].legend(fontsize=8)
    fig.savefig(output / "matched_training_convergence.png", dpi=200); plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/multiseed_joint_20261106")
    parser.add_argument("--user-counts", type=int, nargs="+", default=[30])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=list(METHODS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--total-timesteps", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--pdqn-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--microbatch-size", type=int, default=16, help="Gradient accumulation size; effective batch size is unchanged")
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=.5)
    parser.add_argument("--eval-interval", type=int, default=25_000)
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--validation-seed-start", type=int, default=1_000_000_000)
    parser.add_argument("--test-seed-start", type=int, default=2_000_000_000)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--save-interval", type=int, default=25_000, help="Full recovery checkpoint at first episode boundary after this many steps")
    parser.add_argument("--log-interval-episodes", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--env-config", type=Path, help="EnvConfig JSON, or a saved history/summary containing config/env_config")
    parser.add_argument("--stage", choices=("all", "train", "test", "aggregate"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extend-suite", action="store_true", help="Allow adding users/seeds/methods to an unchanged protocol; completed runs are verified and reused")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Tiny real-environment integration run in a separate directory")
    args = parser.parse_args(argv)
    if args.smoke_test:
        args.user_counts, args.seeds = [2], [42]
        args.total_timesteps, args.max_steps = 12, 4
        args.batch_size, args.warmup_steps, args.replay_size = 4, 4, 32
        args.microbatch_size, args.target_update_interval = 2, 3
        args.eval_interval, args.save_interval = 6, 4
        args.validation_episodes, args.test_episodes = 2, 2
        args.log_interval_episodes = 1
        args.output_dir = args.output_dir / "smoke_test"
    return args


def build_protocol(args):
    positive = ("total_timesteps", "batch_size", "microbatch_size", "replay_size", "warmup_steps",
                "target_update_interval", "eval_interval", "validation_episodes", "test_episodes",
                "max_steps", "save_interval", "log_interval_episodes", "torch_threads")
    if any(getattr(args, key) <= 0 for key in positive) or any(n <= 0 for n in args.user_counts):
        raise ValueError("Counts/intervals must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.user_counts)) != len(args.user_counts) or len(set(args.methods)) != len(args.methods):
        raise ValueError("Duplicate seeds, user counts or methods")
    for seed in args.seeds:
        training_seed(seed, math.ceil(args.total_timesteps / args.max_steps))
    if args.replay_size < max(args.batch_size, args.warmup_steps):
        raise ValueError("Replay capacity must cover batch and warmup")
    if args.total_timesteps < max(args.batch_size, args.warmup_steps):
        raise ValueError("Budget would perform no learning updates")
    if any(not math.isfinite(v) or v <= 0 for v in (args.pdqn_lr, args.encoder_lr, args.grad_clip)):
        raise ValueError("Learning rates and gradient limit must be finite and positive")
    validation = list(range(args.validation_seed_start, args.validation_seed_start + args.validation_episodes))
    test = list(range(args.test_seed_start, args.test_seed_start + args.test_episodes))
    if min(validation + test) < 1_000_000_000 or max(validation + test) >= 2**32 - 100_000 or set(validation) & set(test):
        raise ValueError("Validation/test seeds must be disjoint and outside the training namespace")
    base = asdict(TrainConfig())
    if args.env_config:
        payload = read_json(args.env_config)
        base.update(payload.get("environment", payload.get("env_config", payload.get("config", payload))))
    environment = asdict(build_env_config(base, max_steps=args.max_steps))
    environment.update(seed=0, num_users=0, enable_task_trace=False)
    pdqn = asdict(PDQNConfig())
    for key in ("num_agents", "obs_dim", "max_candidates", "device", "seed"):
        pdqn.pop(key)
    pdqn.update(batch_size=args.batch_size, lr=args.pdqn_lr, replay_size=args.replay_size,
                warmup_steps=args.warmup_steps, target_update_interval=args.target_update_interval,
                grad_clip_norm=args.grad_clip, epsilon_decay_steps=max(int(args.total_timesteps * .25), args.warmup_steps + 1))
    source_files = [Path(__file__).resolve(), ROOT / "scripts/train.py", ROOT / "scripts/compare_system_baselines.py", ROOT / "scripts/paper_metrics.py"]
    source_files += sorted((ROOT / "src").rglob("*.py"))
    return {"version": PROTOCOL_VERSION, "model_schema_version": MODEL_SCHEMA_VERSION,
            "geometry_schema_version": GEOMETRY_SCHEMA_VERSION, "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
            "environment": environment, "pdqn": pdqn, "han": asdict(HANConfig(dropout=0.0)),
            "encoder_lr": args.encoder_lr, "microbatch_size": args.microbatch_size,
            "total_timesteps": args.total_timesteps, "eval_interval": args.eval_interval,
            "validation_seeds": validation, "test_seeds": test, "save_interval": args.save_interval,
            "log_interval_episodes": args.log_interval_episodes, "torch_threads": args.torch_threads,
            "selection_metric": "validation mean reward; earliest checkpoint on ties",
            "replay_terminal": "terminated only; bootstrap at time-limit truncation",
            "time_limit_next_state": "generate next decision arrivals for training bootstrap only",
            "action_mask": "environment feasibility mask including link, migration capacity and pre-handover gate",
            "updates_per_transition": 1, "training_seed_rule": "training_seed * 1000000 + episode_index",
            "encoder_gradient": "sum of TD and parameter losses; one Adam step; separate target HAN",
            # Stable across Windows/Linux path separators and text line endings.
            "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest()
                              for p in source_files}}


def main(argv=None):
    args = parse_args(argv)
    protocol = build_protocol(args)
    output = args.output_dir.resolve()
    plan = {"output_dir": str(output), "methods": args.methods, "user_counts": args.user_counts,
            "seeds": args.seeds, "runs": len(args.methods) * len(args.user_counts) * len(args.seeds),
            "protocol_hash": fingerprint(protocol), "protocol": protocol}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu or run on the GPU server")
    torch.set_num_threads(args.torch_threads)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "suite_manifest.json"
    if manifest.exists():
        previous = read_json(manifest)
        if previous["protocol_hash"] != plan["protocol_hash"]:
            raise ValueError("Output directory already belongs to a different protocol/code version; choose a new --output-dir")
        changes = [key for key in ("methods", "user_counts", "seeds") if previous[key] != plan[key]]
        if changes:
            if not args.extend_suite or any(not set(previous[key]).issubset(plan[key]) for key in changes):
                raise ValueError("Experiment matrix changed; use --extend-suite only to add runs under the same protocol")
            history = previous.get("matrix_history", []) + [{key: previous[key] for key in ("methods", "user_counts", "seeds")}]
            atomic_json(manifest, {**previous, **plan, "matrix_history": history})
    else:
        if any(output.iterdir()):
            raise FileExistsError(f"Refusing to use a nonempty directory without a suite manifest: {output}")
        atomic_json(manifest, {**plan, "python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "initial_device": device})
    LOGGER.setLevel(logging.INFO); LOGGER.propagate = False
    handlers = [logging.StreamHandler(), logging.FileHandler(output / "suite.log", encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s")); LOGGER.addHandler(handler)
    try:
        for users in args.user_counts:
            for seed in args.seeds:
                for method in args.methods:
                    destination = output / f"seed{seed}" / f"u{users}" / "learned_baselines" / method
                    if args.stage in ("all", "train"):
                        train_one(protocol, method, seed, users, destination, device, args.resume)
                    if args.stage in ("all", "test"):
                        test_one(protocol, method, seed, users, destination, device)
        if args.stage in ("all", "test", "aggregate"):
            aggregate(protocol, output, args.methods, args.user_counts, args.seeds)
        LOGGER.info("Requested stage complete: %s", args.stage)
    finally:
        for handler in handlers:
            LOGGER.removeHandler(handler); handler.close()


if __name__ == "__main__":
    main()
