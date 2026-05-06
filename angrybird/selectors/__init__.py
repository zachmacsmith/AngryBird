"""
Selector registry — plug-and-play measurement location selection.

Quick start:
    from angrybird.selectors import registry

    # Run one strategy
    result = registry.run("greedy", info_field, gp, ensemble, k=5)

    # Run all strategies (for comparison)
    results = registry.run_all(info_field, gp, ensemble, k=5)

    # List registered strategies
    print(registry.names())   # ['greedy', 'qubo', 'uniform', 'fire_front']

    # Swap in a custom selector
    registry.register(MyCustomSelector())

    # Remove one from comparisons
    registry.unregister("qubo")
"""

from .base import Selector, SelectorRegistry, spacing_mask
from .baselines import FireFrontSelector, UniformSelector
from .correlation_path import CorrelationPathSelector
from .greedy import GreedySelector
from .heuristics import FireFrontOrbitSelector, LawnmowerSelector
from .qubo import QUBOSelector, build_qubo, extract_candidates, solve_qubo

# Global registry — the orchestrator and scripts import this directly.
registry = SelectorRegistry()
registry.register(GreedySelector())
registry.register(QUBOSelector())
registry.register(UniformSelector())
registry.register(FireFrontSelector())
registry.register(CorrelationPathSelector())
registry.register(LawnmowerSelector())
registry.register(FireFrontOrbitSelector())

__all__ = [
    "registry",
    "SelectorRegistry",
    "Selector",
    "GreedySelector",
    "QUBOSelector",
    "UniformSelector",
    "FireFrontSelector",
    "CorrelationPathSelector",
    "LawnmowerSelector",
    "FireFrontOrbitSelector",
    "build_qubo",
    "extract_candidates",
    "solve_qubo",
    "spacing_mask",
]
