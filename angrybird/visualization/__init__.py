"""
angrybird.visualization
=======================

All visualization functions for the IGNIS wildfire drone planning system.

§1  Operational (core.py)
    plot_fire_prediction_map
    plot_information_field
    plot_gp_uncertainty
    plot_drone_placement
    plot_mission_queue_table

§2  Evaluation (evaluation.py)
    plot_ensemble_spread
    plot_arrival_distributions
    plot_strategy_comparison
    plot_entropy_convergence
    plot_drone_value_curve
    plot_qubo_greedy_overlap
    plot_placement_stability
    plot_ground_truth_reveal
    plot_innovation_tracking

§3  Presentation (presentation.py)
    plot_observation_gap
    plot_architecture
    plot_before_after

Shared helpers
    save_or_show        — save figure to file or display
    VIZ_CONFIG          — global style dict
    STRATEGY_STYLES     — per-strategy color / marker / linestyle
    DRONE_COLORS        — list of up to 10 drone assignment colours

Backward compatibility
    plot_phase2_summary — original phase-2 summary plot (from _visualization_old)
    plot_gp_prior       — GP prior heatmaps  (from _visualization_old)
    plot_ensemble       — ensemble overview   (from _visualization_old)
    plot_sensitivity    — sensitivity panel   (from _visualization_old)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# §1 operational
# ---------------------------------------------------------------------------
from .core import (
    plot_fire_prediction_map,
    plot_information_field,
    plot_gp_uncertainty,
    plot_drone_placement,
    plot_mission_queue_table,
)

# ---------------------------------------------------------------------------
# §2 evaluation
# ---------------------------------------------------------------------------
from .evaluation import (
    plot_ensemble_spread,
    plot_arrival_distributions,
    plot_strategy_comparison,
    plot_entropy_convergence,
    plot_drone_value_curve,
    plot_qubo_greedy_overlap,
    plot_placement_stability,
    plot_ground_truth_reveal,
    plot_innovation_tracking,
)

# ---------------------------------------------------------------------------
# §3 presentation
# ---------------------------------------------------------------------------
from .presentation import (
    plot_observation_gap,
    plot_architecture,
    plot_before_after,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
from ._style import save_or_show, VIZ_CONFIG, STRATEGY_STYLES, DRONE_COLORS

# ---------------------------------------------------------------------------
# Backward compatibility — scripts/demo_phase2.py imports these directly
# from angrybird.visualization
# ---------------------------------------------------------------------------
try:
    from .._visualization_old import (  # type: ignore[import]
        plot_phase2_summary,
        plot_gp_prior,
        plot_ensemble,
        plot_sensitivity,
    )
except Exception:
    # _visualization_old may not exist in all deployments; degrade gracefully
    pass

__all__ = [
    # §1
    "plot_fire_prediction_map",
    "plot_information_field",
    "plot_gp_uncertainty",
    "plot_drone_placement",
    "plot_mission_queue_table",
    # §2
    "plot_ensemble_spread",
    "plot_arrival_distributions",
    "plot_strategy_comparison",
    "plot_entropy_convergence",
    "plot_drone_value_curve",
    "plot_qubo_greedy_overlap",
    "plot_placement_stability",
    "plot_ground_truth_reveal",
    "plot_innovation_tracking",
    # §3
    "plot_observation_gap",
    "plot_architecture",
    "plot_before_after",
    # helpers
    "save_or_show",
    "VIZ_CONFIG",
    "STRATEGY_STYLES",
    "DRONE_COLORS",
    # backward compat
    "plot_phase2_summary",
    "plot_gp_prior",
    "plot_ensemble",
    "plot_sensitivity",
]
