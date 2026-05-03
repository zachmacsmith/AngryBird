"""
Phase 4b — simulation harness for multi-strategy comparison.

These components substitute for real drones at hackathon time and enable the
four-way PERR comparison. None of this ships with the production system.

(Build-order test: "would you ship it with real drones?" → No → simulation/)

Quick start:
    from angrybird.simulation import (
        generate_ground_truth,
        SimulatedObserver,
        SimulationRunner,
    )

    truth   = generate_ground_truth(terrain, seed=42)
    runner  = SimulationRunner(
        orchestrator, truth, fire_engine,
        strategies=["greedy", "qubo", "uniform", "fire_front"],
    )
    reports = runner.run_comparison(fire_states)

    # Headline metric: per-drone entropy reduction rate (PERR)
    for cycle_report in reports:
        for name, eval_ in cycle_report.evaluations.items():
            print(f"Cycle {cycle_report.cycle_id} | {name}: PERR={eval_.perr:.4f}")
"""

from .evaluator import CounterfactualEvaluator
from .ground_truth import GroundTruth, generate_ground_truth
from .observer import ObservationSource, SimulatedObserver
from .runner import SimulationRunner

__all__ = [
    "SimulationRunner",
    "CounterfactualEvaluator",
    "GroundTruth",
    "generate_ground_truth",
    "ObservationSource",
    "SimulatedObserver",
]
