"""
Visualise the correlation-domain segmentation on real LANDFIRE terrain.

Shows three panels side-by-side:
  Left   – Felzenszwalb segmentation (terrain-adaptive, may fall back)
  Centre – Regular-grid segmentation  (fixed 2 km × 2 km tiles)
  Right  – Elevation hillshade for reference

Each segmentation is drawn as a random-colour label map with domain boundaries
overlaid, plus a summary of domain-count and size statistics.

Usage
-----
    python scripts/visualize_domains.py
    python scripts/visualize_domains.py --corr-len 2000 --save out/domains.png
    python scripts/visualize_domains.py --scale-factor 20   # tune Felzenszwalb
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from angrybird.landfire import load_from_directory
from angrybird.selectors.correlation_path import (
    _terrain_features,
    _felzenszwalb_label_map,
    _regular_grid_label_map,
    _is_pathological,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_cmap(n: int, seed: int = 42) -> mcolors.ListedColormap:
    rng = np.random.default_rng(seed)
    colors = rng.uniform(0.2, 0.95, (n, 3))
    return mcolors.ListedColormap(colors)


def _domain_stats(label_map: np.ndarray, resolution_m: float) -> str:
    sizes = np.bincount(label_map.ravel())
    n = len(sizes)
    med_cells = float(np.median(sizes))
    med_m = np.sqrt(med_cells) * resolution_m
    max_pct = sizes.max() / label_map.size * 100
    return (
        f"{n} domains  |  median {med_cells:.0f} cells ({med_m:.0f} m side)  |  "
        f"largest {max_pct:.1f}% of map"
    )


def _draw_boundaries(ax: plt.Axes, label_map: np.ndarray) -> None:
    """Overlay domain boundary edges in white."""
    from skimage.segmentation import find_boundaries
    bounds = find_boundaries(label_map, mode="outer")
    overlay = np.zeros((*bounds.shape, 4), dtype=np.float32)
    overlay[bounds] = [1, 1, 1, 0.6]
    ax.imshow(overlay, interpolation="nearest")


def _plot_segmentation(
    ax: plt.Axes,
    label_map: np.ndarray,
    title: str,
    resolution_m: float,
    pathological: bool,
) -> None:
    n_domains = int(label_map.max()) + 1
    cmap = _rand_cmap(n_domains)
    # Normalise labels to [0, 1] for the cmap
    norm = mcolors.BoundaryNorm(np.arange(n_domains + 1) - 0.5, n_domains)
    ax.imshow(label_map, cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
    _draw_boundaries(ax, label_map)

    rows, cols = label_map.shape
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_xlabel("East (cells) →", fontsize=8)
    ax.set_ylabel("↑ North (cells)", fontsize=8)
    ax.tick_params(labelsize=7)

    flag = "  ⚠ PATHOLOGICAL → fallback" if pathological else ""
    ax.set_title(f"{title}{flag}\n{_domain_stats(label_map, resolution_m)}",
                 fontsize=8, pad=4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise correlation domains on LANDFIRE terrain")
    parser.add_argument("--cache",        default="landfire_cache")
    parser.add_argument("--corr-len",     type=float, default=2000.0,
                        help="Correlation length in metres (default 2000)")
    parser.add_argument("--min-cells",    type=int,   default=15,
                        help="min_domain_cells for Felzenszwalb (default 15)")
    parser.add_argument("--scale-factor", type=float, default=0.5,
                        help="scale = corr_len/res * scale_factor (default 0.5 → scale=10)")
    parser.add_argument("--save",         default=None,
                        help="Save to file instead of displaying")
    args = parser.parse_args()

    print(f"Loading terrain from {args.cache} …")
    terrain = load_from_directory(args.cache, resolution_m=100.0)
    res = terrain.resolution_m
    print(f"  Shape: {terrain.shape[0]}×{terrain.shape[1]}  res={res:.0f} m")

    print("Computing terrain features …")
    features = _terrain_features(terrain)

    # ── Felzenszwalb ────────────────────────────────────────────────────────
    print(f"Running Felzenszwalb  (scale_factor={args.scale_factor}, "
          f"corr_len={args.corr_len:.0f} m, min_cells={args.min_cells}) …")
    scale = args.corr_len / res * args.scale_factor
    print(f"  → Felzenszwalb scale = {scale:.1f}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        from skimage.segmentation import felzenszwalb
        fz_labels = felzenszwalb(
            features, scale=scale, sigma=0.8,
            min_size=args.min_cells, channel_axis=-1,
        )
    fz_labels = fz_labels.astype(np.int32)
    fz_patho  = _is_pathological(fz_labels)

    sizes = np.bincount(fz_labels.ravel())
    print(f"  Domains: {len(sizes)}  |  largest: {sizes.max()} cells "
          f"({sizes.max()/fz_labels.size*100:.1f}%)  |  "
          f"pathological: {fz_patho}")

    # Also test with a better scale
    scale_fixed = args.corr_len / res * 20   # scale_factor=20 → scale=400
    print(f"\nRunning Felzenszwalb  (scale_factor=20, scale={scale_fixed:.0f}) …")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        fz_fixed = felzenszwalb(
            features, scale=scale_fixed, sigma=0.8,
            min_size=args.min_cells, channel_axis=-1,
        ).astype(np.int32)
    fz_fixed_patho = _is_pathological(fz_fixed)
    sizes_fixed = np.bincount(fz_fixed.ravel())
    print(f"  Domains: {len(sizes_fixed)}  |  largest: {sizes_fixed.max()} cells "
          f"({sizes_fixed.max()/fz_fixed.size*100:.1f}%)  |  "
          f"pathological: {fz_fixed_patho}")

    # ── Regular grid ────────────────────────────────────────────────────────
    print("\nBuilding regular grid …")
    rg_labels = _regular_grid_label_map(terrain.shape, args.corr_len, res)
    rg_sizes  = np.bincount(rg_labels.ravel())
    print(f"  Domains: {len(rg_sizes)}  |  "
          f"domain tile: {int(args.corr_len/res)}×{int(args.corr_len/res)} cells "
          f"= {args.corr_len/1000:.1f}×{args.corr_len/1000:.1f} km")

    # ── Elevation reference ─────────────────────────────────────────────────
    elev = terrain.elevation.astype(np.float32)
    hillshade = np.clip((elev - elev.min()) / (elev.max() - elev.min() + 1e-6), 0, 1)

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(
        f"Correlation-domain segmentation  |  LANDFIRE {terrain.shape[0]}×{terrain.shape[1]} "
        f"@ {res:.0f} m  |  corr_len={args.corr_len:.0f} m",
        fontsize=10,
    )

    _plot_segmentation(axes[0], fz_labels,
                       f"Felzenszwalb  (scale={scale:.0f}, factor={args.scale_factor})",
                       res, fz_patho)

    _plot_segmentation(axes[1], fz_fixed,
                       f"Felzenszwalb  (scale={scale_fixed:.0f}, factor=20)",
                       res, fz_fixed_patho)

    _plot_segmentation(axes[2], rg_labels,
                       f"Regular grid  ({int(args.corr_len/res)}×{int(args.corr_len/res)} cells)",
                       res, False)

    axes[3].imshow(hillshade, cmap="gray", origin="upper")
    axes[3].set_title("Elevation (reference)", fontsize=8, pad=4)
    axes[3].set_xlabel("East (cells) →", fontsize=8)
    axes[3].tick_params(labelsize=7)

    plt.tight_layout()

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved → {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
