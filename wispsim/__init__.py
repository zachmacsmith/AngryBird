"""
WISPsim — WISP Simulated Internal Model.

Simulation harness for the WISP system (Wildfire Intelligence Surveillance
Package).  Runs the angrybird (AB Protocol) package in a synthetic environment,
substituting real drones, terrain, and weather with simulated equivalents for
development and evaluation.  Nothing in this package ships with the production
system.

Quick start (cycle-based strategy comparison):

    from wispsim import CycleRunner, generate_ground_truth

    truth  = generate_ground_truth(terrain, ignition_cell=(150, 40), seed=42)
    runner = CycleRunner(
        orchestrator, truth, fire_engine,
        strategies=["greedy", "qubo", "uniform", "fire_front"],
    )
    reports = runner.run_comparison(fire_states)

Quick start (clock-based simulation video):

    from wispsim import SimulationRunner, SimulationConfig, hilly_heterogeneous

    terrain, truth, config = hilly_heterogeneous()
    runner = SimulationRunner(config, terrain, truth, orchestrator)
    reports = runner.run()          # writes out/sim_hilly/frame_*.png + MP4
"""

from .drone_sim import DroneState, NoiseConfig
from .simple_fire import SimpleFire
from .evaluator import CounterfactualEvaluator
from .fire_oracle import GroundTruthFire
from angrybird.fire_engines import GPUFireEngine
from .ground_truth import GroundTruth, WindEvent, generate_ground_truth
from .observation_buffer import ObservationBuffer
from .observer import ObservationSource, SimulatedObserver
from .renderer import FrameRenderer
from .runner import CycleRunner, SimulationConfig, SimulationRunner
from .scenario import FireReport, Scenario, WeatherPrior
from .scenarios import crown_fire_risk, dual_ignition, flat_homogeneous, hilly_heterogeneous, wind_shift

__all__ = [
    # runners
    "CycleRunner",
    "SimulationRunner",
    "SimulationConfig",
    # fire engines
    "GPUFireEngine",
    "SimpleFire",
    # ground truth
    "GroundTruth",
    "WindEvent",
    "generate_ground_truth",
    "GroundTruthFire",
    # drones
    "DroneState",
    "NoiseConfig",
    # observation
    "ObservationBuffer",
    "ObservationSource",
    "SimulatedObserver",
    # rendering
    "FrameRenderer",
    # evaluation
    "CounterfactualEvaluator",
    # scenario types
    "FireReport",
    "Scenario",
    "WeatherPrior",
    # scenario factories
    "hilly_heterogeneous",
    "wind_shift",
    "flat_homogeneous",
    "crown_fire_risk",
    "dual_ignition",
]
