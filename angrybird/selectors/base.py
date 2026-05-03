"""
Selector protocol and registry.

Every selector is a class with a `name: str` attribute and a `select()` method.
Register selectors with the global `registry` object, then call them by name:

    from angrybird.selectors import registry

    result  = registry.run("greedy",  info_field, gp, ensemble, k=5)
    results = registry.run_all(info_field, gp, ensemble, k=5)  # all registered
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..gp import IGNISGPPrior
    from ..types import EnsembleResult, InformationField, SelectionResult


@runtime_checkable
class Selector(Protocol):
    """
    Duck-typed interface every selector must satisfy.
    Implement `name` and `select()`; everything else is optional.
    """

    name: str

    def select(
        self,
        info_field: "InformationField",
        gp: "IGNISGPPrior",
        ensemble: "EnsembleResult",
        k: int,
    ) -> "SelectionResult": ...


class SelectorRegistry:
    """
    Registry of named selectors.  Plug-and-play: swap strategies by name.

    Usage:
        registry.register(MySelector())
        result = registry.run("my_selector", info_field, gp, ensemble, k=5)
        all_results = registry.run_all(info_field, gp, ensemble, k=5)
    """

    def __init__(self) -> None:
        self._selectors: dict[str, Selector] = {}

    def register(self, selector: Selector) -> Selector:
        """Add a selector.  Returns the selector so it can be used as a decorator."""
        self._selectors[selector.name] = selector
        return selector

    def unregister(self, name: str) -> None:
        self._selectors.pop(name, None)

    def __getitem__(self, name: str) -> Selector:
        if name not in self._selectors:
            available = list(self._selectors)
            raise KeyError(f"Unknown selector {name!r}. Available: {available}")
        return self._selectors[name]

    def __contains__(self, name: str) -> bool:
        return name in self._selectors

    def names(self) -> list[str]:
        return list(self._selectors)

    def run(
        self,
        name: str,
        info_field: "InformationField",
        gp: "IGNISGPPrior",
        ensemble: "EnsembleResult",
        k: int,
    ) -> "SelectionResult":
        """Run one selector by name."""
        return self[name].select(info_field, gp, ensemble, k)

    def run_all(
        self,
        info_field: "InformationField",
        gp: "IGNISGPPrior",
        ensemble: "EnsembleResult",
        k: int,
    ) -> "dict[str, SelectionResult]":
        """Run every registered selector; return {name: SelectionResult}."""
        return {
            name: sel.select(info_field, gp, ensemble, k)
            for name, sel in self._selectors.items()
        }

    def __repr__(self) -> str:
        return f"SelectorRegistry({list(self._selectors)})"


# ---------------------------------------------------------------------------
# Shared spatial helper used by all selectors
# ---------------------------------------------------------------------------

def spacing_mask(
    shape: tuple[int, int],
    selected: list[tuple[int, int]],
    min_cells: float,
) -> "import numpy; numpy.ndarray":
    """
    Return bool[rows, cols] where True = too close to an already-selected cell.
    Uses Euclidean distance so the exclusion zone is a circle, not a square.
    """
    import numpy as np

    mask = np.zeros(shape, dtype=bool)
    rows, cols = shape
    for sr, sc in selected:
        r_lo = max(0, int(sr - min_cells) - 1)
        r_hi = min(rows, int(sr + min_cells) + 2)
        c_lo = max(0, int(sc - min_cells) - 1)
        c_hi = min(cols, int(sc + min_cells) + 2)
        rr = np.arange(r_lo, r_hi)[:, None]
        cc = np.arange(c_lo, c_hi)[None, :]
        dist = np.sqrt((rr - sr) ** 2 + (cc - sc) ** 2)
        mask[r_lo:r_hi, c_lo:c_hi] |= dist <= min_cells
    return mask
