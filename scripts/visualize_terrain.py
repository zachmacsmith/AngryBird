"""
Render LANDFIRE terrain from a local cache directory.

Usage
-----
    python scripts/visualize_terrain.py
    python scripts/visualize_terrain.py --save out/terrain.png
    python scripts/visualize_terrain.py --cache path/to/other/cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from angrybird.landfire import load_from_directory
from angrybird.visualization.terrain import plot_terrain_overview


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize LANDFIRE terrain from a local cache directory."
    )
    parser.add_argument(
        "--save", metavar="PATH",
        help="Save the figure to this path instead of displaying it.",
    )
    parser.add_argument(
        "--cache", default="landfire_cache",
        help="Directory of LANDFIRE GeoTIFFs (default: landfire_cache).",
    )
    args = parser.parse_args()

    print(f"Loading terrain from {args.cache}…\n")
    terrain = load_from_directory(args.cache)

    print(f"Terrain loaded: {terrain.shape[0]}×{terrain.shape[1]} cells "
          f"at {terrain.resolution_m:.0f} m resolution")
    print(f"Elevation range: {terrain.elevation.min():.0f} – "
          f"{terrain.elevation.max():.0f} m")

    save_path = Path(args.save) if args.save else None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_terrain_overview(terrain, save_path=save_path)

    if save_path is None:
        plt.show()


if __name__ == "__main__":
    main()
