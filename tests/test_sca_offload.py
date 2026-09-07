from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog
import torch

from src.algorithm.sca_offload import (
    OffloadProblem, SCAConfig, SCAResult, majorizer_cost, solve_offload, split_breakpoints,
)
from src.algorithm.sca_adapter import FixedDiscreteSCA
from src.environment.gym_env import EnvConfig, LEOSatelliteEnv


def problem(**overrides):
    base = OffloadProblem(4.2, -4.0, 0.4, 1.6, 4.0, 0.6, 3.0)
    return replace(base, **overrides)


def test_analytic_subproblem_matches_independent_highs_lp():
    rng = np.random.default_rng(31)
    cfg = SCAConfig()
    for _ in range(40):
        local = rng.uniform(0.1, 10)
        p = problem(local_intercept=local + rng.uniform(0, 2), local_slope=-local,
                    remote_intercept=rng.uniform(0, 3), remote_slope=rng.uniform(0.1, 10),
                    local_energy=rng.uniform(0, 6), upload_energy=rng.uniform(0, 6),
                    deadline=rng.uniform(0.2, 8))
        reference = rng.uniform(0.05, 1)
        energy0 = p.energy(reference)
        derivative = p.energy_reference / (energy0 + p.energy_reference) ** 2
        c = [p.energy_weight * derivative * (p.upload_energy - p.local_energy),
             p.delay_weight / p.deadline, cfg.deadline_penalty / p.deadline]
        upper = float(np.nextafter(np.float32(1), np.float32(0)))
        result = linprog(c, A_ub=[[p.local_slope, -1, 0], [p.remote_slope, -1, 0], [0, 1, -1]],
                         b_ub=[-p.local_intercept, -p.remote_intercept, p.deadline],
                         bounds=[(p.minimum_ratio, upper), (0, None), (0, None)], method="highs")
        assert result.success
        constant = p.energy_weight * (energy0 / (energy0 + p.energy_reference)
                                      + derivative * (p.local_energy - energy0))
        analytic = min(majorizer_cost(p, cfg, x, reference) for x in split_breakpoints(p))
        assert analytic == pytest.approx(result.fun + constant, abs=1e-8)


def test_energy_majorizer_is_tight_and_upper_bounding():
    p, cfg = problem(), SCAConfig()
    for reference in (0.05, 0.3, 0.8, 1.0):
        assert majorizer_cost(p, cfg, reference, reference) == pytest.approx(p.cost(reference, cfg))
        for ratio in np.linspace(0.05, 1, 50):
            assert majorizer_cost(p, cfg, float(ratio), reference) >= p.cost(float(ratio), cfg) - 1e-12


def test_mm_is_monotone_and_returns_executable_ratio():
    rng = np.random.default_rng(9)
    cfg = SCAConfig(max_iterations=8)
    for _ in range(60):
        p = problem(upload_energy=float(rng.uniform(0.1, 10)), deadline=float(rng.uniform(0.3, 6)))
        initial = float(np.float32(rng.uniform(0.05, 1)))
        result = solve_offload(p, initial, cfg)
        assert p.minimum_ratio <= result.ratio <= 1
        assert result.ratio == float(np.float32(result.ratio))
        assert result.final_cost <= result.initial_cost
        assert all(b <= a for a, b in zip(result.cost_history, result.cost_history[1:]))


def test_full_offload_drops_local_branch_and_infeasible_deadlines_still_solve():
    p = problem(local_intercept=30, local_slope=-4, remote_intercept=0.1, remote_slope=0.1)
    result = solve_offload(p, 0.5, SCAConfig())
    assert result.ratio == 1
    assert p.delay(1) == pytest.approx(0.2)
    impossible = replace(p, deadline=0.01)
    result = solve_offload(impossible, 0.5, SCAConfig())
    assert np.isfinite(result.final_cost)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        problem(upload_energy=float("nan"))
    with pytest.raises(ValueError):
        solve_offload(problem(), 0.0, SCAConfig())
    with pytest.raises(ValueError):
        SCAConfig(max_iterations=0)


def make_env():
    return LEOSatelliteEnv(EnvConfig(num_users=6, max_steps=5, task_arrival_prob=1.0, seed=42))


def legal_actions(env):
    masks = env.get_handover_action_mask(max_candidates=env.max_visible_sats, apply_pre_handover_gate=True)
    actions = np.zeros((env.num_users, 2), dtype=np.float32)
    for user_id, row in enumerate(masks):
        # Exercise real handover/migration code whenever a non-stay action is legal.
        choices = np.flatnonzero(row)
        actions[user_id, 0] = choices[-1]
        actions[user_id, 1] = 0.0 if user_id == 0 else 0.6
    return actions


def test_identity_adapter_matches_unmodified_environment(monkeypatch):
    import src.algorithm.sca_adapter as adapter_module

    def identity(p, ratio, cfg):
        cost = p.cost(ratio, cfg)
        return SCAResult(ratio, "converged", 1, cost, cost, (cost,))

    monkeypatch.setattr(adapter_module, "solve_offload", identity)
    original, wrapped = make_env(), make_env()
    original.reset(seed=1_000_042)
    wrapped.reset(seed=1_000_042)
    attempted = 0
    try:
        with FixedDiscreteSCA(wrapped, SCAConfig()) as adapter:
            for _ in range(5):
                requested = legal_actions(original)
                before = requested.copy()
                expected = original.step(requested, return_observation=False, return_info=False)
                actual, executed, diagnostic = adapter.step(requested, return_observation=False, return_info=False)
                assert np.array_equal(requested, before)
                assert np.array_equal(executed, requested)
                assert actual[1] == pytest.approx(expected[1])
                for key in ("completed_tasks", "total_tasks", "total_energy", "total_handovers", "failed_tasks"):
                    assert wrapped.stats[key] == pytest.approx(original.stats[key])
                attempted += diagnostic["attempted"]
        assert attempted > 0
        assert "_plan_ofdma_uplink_allocations" not in vars(wrapped)
    finally:
        original.close()
        wrapped.close()


def test_refinement_preserves_handover_mode_and_restores_hook_on_exception(monkeypatch):
    env = make_env()
    env.reset(seed=1_000_042)
    attempted = changed = 0
    try:
        with FixedDiscreteSCA(env, SCAConfig()) as adapter:
            for _ in range(5):
                requested = legal_actions(env)
                _, executed, diag = adapter.step(requested, return_observation=False, return_info=False)
                assert np.array_equal(executed[:, 0], requested[:, 0])
                assert np.array_equal(executed[:, 1] >= 0.05, requested[:, 1] >= 0.05)
                assert executed[0, 1] == 0
                assert diag["proxy_cost_after"] <= diag["proxy_cost_before"]
                attempted += diag["attempted"]
                changed += diag["changed"]
        assert attempted > 0 and changed > 0
        with pytest.raises(RuntimeError, match="test exception"):
            with FixedDiscreteSCA(env, SCAConfig()):
                raise RuntimeError("test exception")
        assert "_plan_ofdma_uplink_allocations" not in vars(env)
    finally:
        env.close()


def test_bad_coefficients_fall_back_to_original_ratio(monkeypatch):
    env = make_env()
    env.reset(seed=1_000_042)
    attempted = 0
    try:
        with FixedDiscreteSCA(env, SCAConfig()) as adapter:
            def fail(*args):
                raise ValueError("invalid coefficient fixture")
            monkeypatch.setattr(adapter, "_problem", fail)
            for _ in range(5):
                requested = legal_actions(env)
                _, executed, diag = adapter.step(requested, return_observation=False, return_info=False)
                assert np.array_equal(executed, requested)
                assert diag["fallback"] == diag["attempted"]
                attempted += diag["attempted"]
        assert attempted > 0
    finally:
        env.close()


def test_paired_runner_checkpoint_and_output_protection(tmp_path):
    from scripts.train import HANMAPPOTrainer, HANPDQNTrainer, TrainConfig
    from scripts.evaluate_sca_baseline import run_comparison

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    trainers = []
    try:
        common = dict(device="cpu", num_users=4, max_steps=8, task_arrival_prob=1.0,
                      replay_size=16, batch_size=2, log_path=str(tmp_path / "logs"))
        # These random weights are synthetic fixtures, NOT experimental results.
        encoder = HANMAPPOTrainer(TrainConfig(**common, algorithm="mappo", save_path=str(tmp_path / "encoder")))
        trainers.append(encoder)
        encoder._save_checkpoint(best=True)
        source = HANPDQNTrainer(TrainConfig(**common, algorithm="pdqn", exp_name="han_pdqn_fixture",
                                           save_path=str(tmp_path / "source"),
                                           pretrained_han_path=str(tmp_path / "encoder/best_model.pt")))
        trainers.append(source)
        source._save_checkpoint(best=True)
        checkpoint = tmp_path / "source/best_model.pt"
        checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        path = run_comparison(checkpoint, tmp_path / "paired", evaluation_seeds=[1_000_042, 1_000_043],
                              device="cpu", max_steps=8, plots=False)
        summary = json.loads((path / "comparison_summary.json").read_text())
        methods = summary["methods"]
        assert [m["display_name"] for m in methods] == ["HAN+PDQN", "SCA"]
        assert all(m["evaluation_seeds"] == [1_000_042, 1_000_043] for m in methods)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checksum
        diagnostics = json.loads((path / "sca_diagnostics.json").read_text())[-1]
        assert diagnostics["counts"]["attempted"] > 0
        assert diagnostics["discrete_changes"] == diagnostics["execution_mode_changes"] == 0
        manifest = json.loads((path / "sca_manifest.json").read_text())
        assert manifest["status"] == "complete"
        before = (path / "comparison_summary.json").read_bytes()
        with pytest.raises(FileExistsError):
            run_comparison(checkpoint, path, plots=False)
        assert (path / "comparison_summary.json").read_bytes() == before
    finally:
        for trainer in trainers:
            trainer.env.close()
            for handler in trainer.logger.handlers[:]:
                handler.close()
                trainer.logger.removeHandler(handler)
        torch.set_num_threads(old_threads)
