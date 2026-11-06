import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import run_matched_pdqn_suite as suite
from src.algorithm.pdqn import PDQNAlgorithm


@pytest.fixture
def protocol():
    torch.set_num_threads(1)
    args = suite.parse_args(["--smoke-test", "--device", "cpu"])
    return suite.build_protocol(args)


def populated(protocol, method="han_pdqn_joint"):
    suite.set_seed(42)
    learner, cfg = suite.make_learner(protocol, method, 42, 2, "cpu")
    env = suite.LEOSatelliteEnv(cfg)
    env.reset(seed=suite.training_seed(42, 0))
    builder = suite.HeteroGraphBuilder() if learner.encoder else None
    replay = suite.SnapshotReplay(32, 42)
    state = suite.Snapshot.capture(env, builder)
    for _ in range(4):
        actions, features, _ = learner.random_actions(state.masks)
        _, reward, terminal, truncated, _ = env.step(actions)
        if truncated and not terminal:
            env._generate_tasks()
        nxt = suite.Snapshot.capture(env, builder)
        replay.add(state, features, suite.per_user_training_rewards(env, reward), nxt, terminal)
        state = nxt
    env.close()
    return learner, cfg, replay


def test_protocol_disjoint_scenarios_and_invalid_settings(protocol):
    train = {suite.training_seed(seed, ep) for seed in range(42, 47) for ep in range(300)}
    assert len(train) == 1500
    assert train.isdisjoint(protocol["validation_seeds"])
    assert train.isdisjoint(protocol["test_seeds"])
    assert set(protocol["validation_seeds"]).isdisjoint(protocol["test_seeds"])
    args = suite.parse_args(["--test-seed-start", "1000000000"])
    with pytest.raises(ValueError, match="disjoint"):
        suite.build_protocol(args)
    args = suite.parse_args(["--replay-size", "100"])
    with pytest.raises(ValueError, match="Replay capacity"):
        suite.build_protocol(args)


def test_effective_pdqn_settings_match(protocol):
    raw, _ = suite.make_learner(protocol, "pdqn_matched", 42, 2, "cpu")
    joint, _ = suite.make_learner(protocol, "han_pdqn_joint", 42, 2, "cpu")
    a, b = suite.asdict(raw.config), suite.asdict(joint.config)
    assert b.pop("obs_dim") == a.pop("obs_dim") + protocol["han"]["out_dim"]
    assert a == b


def test_graph_batching_has_no_cross_snapshot_attention(protocol):
    learner, _, replay = populated(protocol)
    states = [item[0] for item in replay.data]
    with torch.no_grad():
        batch = learner.encode(states)
        individually = torch.cat([learner.encode([state]) for state in states])
    torch.testing.assert_close(batch, individually, rtol=2e-5, atol=2e-6)


def test_joint_gradients_target_sync_and_replay_contains_raw_graphs(protocol):
    learner, _, replay = populated(protocol)
    before = copy.deepcopy(learner.encoder.state_dict())
    target_before = copy.deepcopy(learner.target_encoder.state_dict())
    first_raw = replay.data[0][0].raw.copy()
    result = learner.update(replay)
    assert result["han_grad_norm"] > 0
    assert any(not torch.equal(before[k], v) for k, v in learner.encoder.state_dict().items())
    assert all(torch.equal(target_before[k], v) for k, v in learner.target_encoder.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in learner.target_encoder.parameters())
    assert replay.data[0][0].graph is not None
    assert replay.data[0][0].raw.shape[-1] < learner.obs_dim
    np.testing.assert_array_equal(replay.data[0][0].raw, first_raw)
    learner.update(replay); learner.update(replay)
    for key, value in learner.encoder.state_dict().items():
        torch.testing.assert_close(value, learner.target_encoder.state_dict()[key], rtol=0, atol=0)


def test_replay_roundtrip_preserves_sampling_and_shared_snapshots(protocol):
    _, _, replay = populated(protocol)
    restored = suite.SnapshotReplay(32, 999)
    restored.load_state_dict(replay.state_dict())
    assert restored.data[0][3] is restored.data[1][0]
    for a, b in zip(replay.sample(4), restored.sample(4)):
        np.testing.assert_array_equal(a[0].raw, b[0].raw)
        np.testing.assert_array_equal(a[1], b[1])
    assert not restored.data[-1][-1]  # A time limit must not suppress bootstrap.
    assert restored.data[-1][3].masks.any(axis=1).all()


def test_evaluation_does_not_change_training_rng_or_weights(protocol):
    learner, cfg, replay = populated(protocol)
    before = suite.capture_rng()
    action_before = copy.deepcopy(learner.rng.bit_generator.state)
    weights_before = copy.deepcopy(learner.encoder.state_dict())
    suite.evaluate(learner, cfg, protocol["validation_seeds"], "han_pdqn_joint")
    after = suite.capture_rng()
    assert before["python"] == after["python"]
    np.testing.assert_array_equal(before["numpy"][1], after["numpy"][1])
    assert before["numpy"][2:] == after["numpy"][2:]
    assert torch.equal(before["torch"], after["torch"])
    assert action_before == learner.rng.bit_generator.state
    assert all(torch.equal(weights_before[k], v) for k, v in learner.encoder.state_dict().items())


def test_raw_update_matches_existing_pdqn_math(protocol):
    learner, _, replay = populated(protocol, "pdqn_matched")
    reference = PDQNAlgorithm(copy.deepcopy(learner.config))
    for name in ("q_net", "target_q_net", "param_nets", "target_param_nets"):
        getattr(reference, name).load_state_dict(getattr(learner, name).state_dict())
    replay_rng = replay.rng.getstate()
    samples = replay.sample(4)
    replay.rng.setstate(replay_rng)
    class FixedReplay:
        def __len__(self): return 4
        def sample(self, count):
            tensor = lambda rows, dtype=torch.float32: torch.tensor(np.asarray(rows), dtype=dtype)
            return {"obs": tensor([s[0].raw for s in samples]), "next_obs": tensor([s[3].raw for s in samples]),
                    "actions": tensor([s[1] for s in samples]), "rewards": tensor([s[2] for s in samples]),
                    "dones": tensor([s[4] for s in samples]),
                    "masks": tensor([s[0].masks for s in samples], torch.bool),
                    "next_masks": tensor([s[3].masks for s in samples], torch.bool)}
    expected = reference.update(FixedReplay())
    actual = learner.update(replay)
    assert actual["q_loss"] == pytest.approx(expected["q_loss"], rel=2e-5, abs=1e-7)
    assert actual["param_loss"] == pytest.approx(expected["param_loss"], rel=2e-5, abs=1e-7)
    for name in ("q_net", "param_nets"):
        for key, value in getattr(learner, name).state_dict().items():
            torch.testing.assert_close(value, getattr(reference, name).state_dict()[key], rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("method", ["pdqn_matched", "han_pdqn_joint"])
def test_interrupted_resume_matches_uninterrupted_training(protocol, tmp_path, monkeypatch, method):
    uninterrupted, interrupted = tmp_path / "full", tmp_path / "resume"
    suite.train_one(protocol, method, 42, 2, uninterrupted, "cpu")
    original_save = suite.atomic_torch
    def stop_after_boundary(path, data):
        original_save(path, data)
        if path.name == "resume.pt" and data["total_steps"] == 8:
            raise KeyboardInterrupt("simulated power interruption after atomic checkpoint")
    monkeypatch.setattr(suite, "atomic_torch", stop_after_boundary)
    with pytest.raises(KeyboardInterrupt):
        suite.train_one(protocol, method, 42, 2, interrupted, "cpu")
    monkeypatch.setattr(suite, "atomic_torch", original_save)
    with pytest.raises(FileExistsError, match="Incomplete"):
        suite.train_one(protocol, method, 42, 2, interrupted, "cpu")
    suite.train_one(protocol, method, 42, 2, interrupted, "cpu", resume=True)
    for name in ("best_model.pt", "final_model.pt"):
        a = suite.load_checkpoint(uninterrupted / name, suite.fingerprint(protocol))
        b = suite.load_checkpoint(interrupted / name, suite.fingerprint(protocol))
        assert a["total_steps"] == b["total_steps"]
        for module in a["learner"]["modules"]:
            for key, value in a["learner"]["modules"][module].items():
                torch.testing.assert_close(value, b["learner"]["modules"][module][key], rtol=0, atol=0)
        if a["learner"]["encoder"]:
            for key, value in a["learner"]["encoder"].items():
                torch.testing.assert_close(value, b["learner"]["encoder"][key], rtol=0, atol=0)
    history = suite.read_json(interrupted / "training_history.json")
    assert [r["total_steps"] for r in history["validation"]] == [6, 12]
    result = suite.test_one(protocol, method, 42, 2, interrupted, "cpu")
    assert result["evaluation_seeds"] == protocol["test_seeds"]
    assert result["checkpoint_step"] in [6, 12]
    assert suite.test_one(protocol, method, 42, 2, interrupted, "cpu") == result
    changed = copy.deepcopy(protocol); changed["pdqn"]["batch_size"] = 8
    with pytest.raises(ValueError, match="Different protocol"):
        suite.train_one(changed, method, 42, 2, interrupted, "cpu", resume=True)


def test_aggregate_refuses_missing_seed(protocol, tmp_path):
    with pytest.raises(FileNotFoundError):
        suite.aggregate(protocol, tmp_path, ["pdqn_matched"], [30], [42, 43])


def test_dry_run_has_no_output_side_effects(tmp_path, capsys):
    destination = tmp_path / "preview"
    suite.main(["--dry-run", "--output-dir", str(destination)])
    plan = json.loads(capsys.readouterr().out)
    assert plan["runs"] == 10
    assert not destination.exists()
    assert plan["protocol"]["pdqn"]["batch_size"] == 512


def test_suite_matrix_can_only_be_explicitly_extended(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(suite, "train_one", lambda *args: calls.append(args[1:4]))
    command = ["--stage", "train", "--device", "cpu", "--output-dir", str(tmp_path / "suite")]
    suite.main(command + ["--seeds", "42"])
    with pytest.raises(ValueError, match="matrix changed"):
        suite.main(command + ["--seeds", "42", "43"])
    suite.main(command + ["--seeds", "42", "43", "--extend-suite"])
    manifest = suite.read_json(tmp_path / "suite/suite_manifest.json")
    assert manifest["seeds"] == [42, 43]
    assert manifest["matrix_history"][0]["seeds"] == [42]
    with pytest.raises(ValueError, match="matrix changed"):
        suite.main(command + ["--seeds", "43", "--extend-suite"])
