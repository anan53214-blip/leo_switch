"""Fixed-discrete offload refinement by majorization-minimization (SCA).

Each split-branch subproblem is an LP with variables (ratio, delay, slack).
Eliminating delay and slack leaves a convex piecewise-linear scalar objective;
its minimum is attained at an endpoint or breakpoint. This solves the LP
exactly without a general-purpose solver. Full offload is evaluated separately
because its local branch disappears. Guarantees concern this frozen proxy,
not the simulator's FCFS dynamics or discontinuous completion reward.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SCAConfig:
    max_iterations: int = 5
    tolerance: float = 1e-6
    deadline_penalty: float = 10.0
    queue_reference_cycles: float = 2e9

    def __post_init__(self):
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        for name in ("tolerance", "deadline_penalty", "queue_reference_cycles"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class OffloadProblem:
    local_intercept: float
    local_slope: float
    remote_intercept: float
    remote_slope: float
    local_energy: float
    upload_energy: float
    deadline: float
    minimum_ratio: float = 0.05
    delay_weight: float = 0.6
    energy_weight: float = 0.4
    energy_reference: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(float(v)) for v in vars(self).values()):
            raise ValueError("non-finite SCA coefficient")
        if not 0 < self.minimum_ratio <= 1:
            raise ValueError("minimum_ratio must be in (0, 1]")
        if self.deadline <= 0 or self.energy_reference <= 0:
            raise ValueError("deadline and energy_reference must be positive")
        if min(self.local_intercept, self.remote_intercept, self.remote_slope,
               self.local_energy, self.upload_energy, self.delay_weight, self.energy_weight) < 0:
            raise ValueError("negative physical coefficient or weight")
        if self.local_slope > 0 or self.local_intercept + self.local_slope < -1e-9:
            raise ValueError("invalid local delay branch")

    def delay(self, ratio: float) -> float:
        remote = self.remote_intercept + self.remote_slope * ratio
        local = 0.0 if ratio == 1.0 else self.local_intercept + self.local_slope * ratio
        return max(local, remote, 0.0)

    def energy(self, ratio: float) -> float:
        return (1.0 - ratio) * self.local_energy + ratio * self.upload_energy

    def cost(self, ratio: float, config: SCAConfig) -> float:
        delay = self.delay(ratio)
        energy = self.energy(ratio)
        return (
            self.delay_weight * delay / self.deadline
            + self.energy_weight * energy / (energy + self.energy_reference)
            + config.deadline_penalty * max(delay - self.deadline, 0.0) / self.deadline
        )


@dataclass(frozen=True)
class SCAResult:
    ratio: float
    status: str
    iterations: int
    initial_cost: float
    final_cost: float
    cost_history: tuple[float, ...]


def split_breakpoints(problem: OffloadProblem) -> list[float]:
    """LP breakpoints on the executable float32 split interval (lambda < 1)."""
    lower = problem.minimum_ratio
    upper = float(np.nextafter(np.float32(1.0), np.float32(0.0)))
    if lower > upper:
        return []
    points = [lower, upper]
    branches = [
        (problem.local_intercept, problem.local_slope),
        (problem.remote_intercept, problem.remote_slope),
    ]
    denominator = problem.local_slope - problem.remote_slope
    if denominator != 0:
        points.append((problem.remote_intercept - problem.local_intercept) / denominator)
    for intercept, slope in branches:
        if slope != 0:
            points.append((problem.deadline - intercept) / slope)
    return sorted({float(x) for x in points if lower <= x <= upper})


def majorizer_cost(problem: OffloadProblem, config: SCAConfig, ratio: float, reference: float) -> float:
    """Tight global tangent upper bound of the concave normalized energy term."""
    energy0 = problem.energy(reference)
    derivative = problem.energy_reference / (energy0 + problem.energy_reference) ** 2
    energy_tangent = (
        energy0 / (energy0 + problem.energy_reference)
        + derivative * (problem.energy(ratio) - energy0)
    )
    delay = problem.delay(ratio)
    return (problem.delay_weight * delay / problem.deadline
            + problem.energy_weight * energy_tangent
            + config.deadline_penalty * max(delay - problem.deadline, 0.0) / problem.deadline)


def solve_offload(problem: OffloadProblem, initial_ratio: float, config: SCAConfig) -> SCAResult:
    """Solve successive LPs and compare the full-offload endpoint; never worsen the proxy."""
    if not math.isfinite(initial_ratio) or not problem.minimum_ratio <= initial_ratio <= 1:
        raise ValueError("initial ratio must preserve the offload mode")
    current = float(initial_ratio)
    costs = [problem.cost(current, config)]
    candidates = split_breakpoints(problem) + [1.0]
    for iteration in range(1, config.max_iterations + 1):
        # Including the current point handles degeneracy without arbitrary jumps.
        next_ratio = min(candidates + [current], key=lambda ratio: (
            majorizer_cost(problem, config, ratio, current), abs(ratio - current)
        ))
        # Return only float32-representable actions; recheck after rounding.
        next_ratio = float(np.float32(next_ratio))
        if next_ratio < problem.minimum_ratio:
            next_ratio = float(np.nextafter(np.float32(problem.minimum_ratio), np.float32(1.0)))
        next_cost = problem.cost(next_ratio, config)
        if next_cost > costs[-1]:
            return SCAResult(current, "numerical_guard", iteration, costs[0], costs[-1], tuple(costs))
        change = abs(next_ratio - current)
        improvement = costs[-1] - next_cost
        current = next_ratio
        costs.append(next_cost)
        if change <= config.tolerance or improvement <= config.tolerance:
            return SCAResult(current, "converged", iteration, costs[0], costs[-1], tuple(costs))
    return SCAResult(current, "iteration_limit", config.max_iterations, costs[0], costs[-1], tuple(costs))
