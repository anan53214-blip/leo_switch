"""Evaluation-only SCA adapter for the existing environment's two-phase step.

The instance-local hook runs AFTER the original handover transition and atomic
admission planning, BEFORE task execution. It changes only the parsed ratio
array, keeps the same upload activation set and returns the original allocation
objects. No shared environment class, policy network, reward or scheduler is
modified. The hook is removed on context exit, including exceptions.
"""

from collections import Counter
from time import perf_counter

import numpy as np

from src.algorithm.sca_offload import OffloadProblem, SCAConfig, solve_offload


class FixedDiscreteSCA:
    def __init__(self, env, config: SCAConfig):
        self.env = env
        self.config = config
        self.last_ratios = None
        self.last_diagnostics = {}
        self._active = False

    def __enter__(self):
        if self._active or "_plan_ofdma_uplink_allocations" in vars(self.env):
            raise RuntimeError("SCA requires an unwrapped, private evaluation environment")
        self._original_plan = self.env._plan_ofdma_uplink_allocations
        self.env._plan_ofdma_uplink_allocations = self._plan_and_refine
        self._active = True
        return self

    def __exit__(self, *exc):
        del self.env._plan_ofdma_uplink_allocations
        self._active = False

    def _problem(self, user_id, allocation, rank):
        env = self.env
        task = env.user_tasks[user_id]
        user = env.user_manager.users[user_id]
        server = env.mec_manager.get_server(user.serving_satellite)
        vis = env._get_satellite_visibility(user, user.serving_satellite)
        if server is None or vis is None or not vis.is_visible:
            raise ValueError("missing serving MEC or visible link")
        # Queue depth is an input feature. Do not inspect task_queue contents,
        # exact remaining cycles or the private local CPU reservation timeline.
        slots = max(int(server.config.mec_max_concurrent_tasks), 1)
        depth = int(server.queue_length)
        active = min(max(depth + int(allocation.concurrent_users), 1), slots)
        frequency = float(server.total_capacity_ghz) * 1e9 / active
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("invalid MEC rate")
        queue_wait = ((depth + rank) // slots) * self.config.queue_reference_cycles / frequency
        age = float(np.clip(env.current_time - task.creation_time, 0.0, 2.0 * task.max_delay))
        local_compute = env.offload_calc.compute_local_delay(task.computation)
        local_energy = env.offload_calc.compute_local_energy(task.computation)
        link = (vis.distance_km, vis.elevation_deg)
        propagation = sum(env.offload_calc.compute_transmission_delay(
            0.0, *link, bandwidth_mhz=allocation.bandwidth_mhz))
        full_transmission = sum(env.offload_calc.compute_transmission_delay(
            task.data_size, *link, bandwidth_mhz=allocation.bandwidth_mhz))
        upload_energy = env.offload_calc.compute_transmission_energy(
            task.data_size, *link, bandwidth_mhz=allocation.bandwidth_mhz)
        return OffloadProblem(
            local_intercept=age + local_compute,
            local_slope=-local_compute,
            remote_intercept=age + queue_wait + propagation
                + float(env._step_handover_interruption_seconds[user_id]),
            remote_slope=max(full_transmission - propagation, 0.0) + task.computation / frequency,
            local_energy=local_energy, upload_energy=upload_energy,
            deadline=float(task.max_delay),
            minimum_ratio=float(env.config.min_effective_offload_ratio),
            delay_weight=float(env.config.reward_delay_weight),
            energy_weight=float(env.config.reward_energy_weight),
            energy_reference=float(env.config.reward_energy_reference_j),
        )

    def _plan_and_refine(self, ratios):
        # The original environment has already expired tasks and executed all
        # handovers, including migrations/failures. No speculative second step.
        allocations = self._original_plan(ratios)
        start = perf_counter()
        original = ratios.copy()
        proposed = original.copy()
        ranks = Counter()
        statuses = Counter()
        attempted = changed = iterations = 0
        initial_cost = final_cost = 0.0
        failure_messages = []
        self._admitted_tasks = {}
        for user_id, allocation in allocations.items():
            if not allocation.admission_allowed:
                continue
            task = self.env.user_tasks.get(user_id)
            if task is None:
                continue
            self._admitted_tasks[user_id] = task
            sid = int(self.env.user_manager.users[user_id].serving_satellite)
            rank = ranks[sid]
            ranks[sid] += 1
            attempted += 1
            try:
                problem = self._problem(user_id, allocation, rank)
                result = solve_offload(problem, float(original[user_id]), self.config)
                proposed[user_id] = result.ratio
                statuses[result.status] += 1
                iterations += result.iterations
                initial_cost += result.initial_cost
                final_cost += result.final_cost
                changed += int(abs(float(proposed[user_id]) - float(original[user_id])) > self.config.tolerance)
            except (ValueError, ArithmeticError) as exc:
                # Preserve the original ratio; no ad-hoc change to handover or mode.
                statuses["fallback"] += 1
                failure_messages.append(f"user {user_id}: {exc}")
        minimum = float(self.env.config.min_effective_offload_ratio)
        if (not np.all(np.isfinite(proposed)) or np.any(proposed < 0) or np.any(proposed > 1)
                or not np.array_equal(proposed >= minimum, original >= minimum)):
            raise RuntimeError("SCA violated the ratio/mode invariant")
        ratios[:] = proposed
        self.last_ratios = proposed.copy()
        self.last_diagnostics = {
            "attempted": attempted, "changed": changed, "iterations": iterations,
            "converged": statuses["converged"], "iteration_limit": statuses["iteration_limit"],
            "numerical_guard": statuses["numerical_guard"], "fallback": statuses["fallback"],
            "proxy_cost_before": initial_cost, "proxy_cost_after": final_cost,
            "refinement_ms": (perf_counter() - start) * 1000,
            "failure_messages": failure_messages,
        }
        return allocations

    def step(self, actions, **kwargs):
        if not self._active:
            raise RuntimeError("Use FixedDiscreteSCA as a context manager")
        requested = np.asarray(actions, dtype=np.float32).copy()
        self.last_ratios = None
        outcome = self.env.step(requested, **kwargs)
        if self.last_ratios is None:
            raise RuntimeError("Environment did not invoke the atomic upload planner")
        executed = requested.copy()
        executed[:, 1] = self.last_ratios
        assert np.array_equal(executed[:, 0], requested[:, 0])
        self.last_diagnostics["admission_mismatch"] = sum(
            float(task.offload_ratio) < float(self.env.config.min_effective_offload_ratio)
            for task in self._admitted_tasks.values()
        )
        return outcome, executed, dict(self.last_diagnostics)
