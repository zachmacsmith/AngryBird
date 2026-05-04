"""
Download LANDFIRE terrain for a bounding box and display a birds-eye overview.

Usage
-----
    python scripts/visualize_terrain.py

The script prompts for two lat/lon coordinates that define the bounding box,
downloads the terrain from LANDFIRE, and shows the 8-panel overview.
Optionally saves to a file with --save.

    python scripts/visualize_terrain.py --save out/terrain.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the package is importable when run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from angrybird.landfire import download_terrain, load_from_directory
from angrybird.visualization.terrain import plot_terrain_overview


def _prompt_coord(label: str) -> tuple[float, float]:
    while True:
        raw = input(f"  {label} (lat lon, space-separated): ").strip()
        parts = raw.split()
        if len(parts) != 2:
            print("  Please enter exactly two numbers separated by a space.")
            continue
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except ValueError:
            print("  Could not parse as numbers. Try again.")
            continue
        if not (-90 <= lat <= 90):
            print("  Latitude must be between -90 and 90.")
            continue
        if not (-180 <= lon <= 180):
            print("  Longitude must be between -180 and 180.")
            continue
        return lat, lon


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize LANDFIRE terrain for a bounding box."
    )
    parser.add_argument(
        "--save", metavar="PATH",
        help="Save the figure to this path instead of displaying it.",
    )
    parser.add_argument(
        "--cache", default="landfire_cache",
        help="Directory of LANDFIRE GeoTIFFs (default: landfire_cache).",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Load directly from --cache directory (no download).",
    )
    parser.add_argument(
        "--coord1", nargs=2, type=float, metavar=("LAT", "LON"),
        help="First corner as LAT LON (skips interactive prompt).",
    )
    parser.add_argument(
        "--coord2", nargs=2, type=float, metavar=("LAT", "LON"),
        help="Second corner as LAT LON (skips interactive prompt).",
    )
    args = parser.parse_args()

    if args.local:
        print(f"Loading terrain from {args.cache}…\n")
        terrain = load_from_directory(args.cache)
    else:
        if args.coord1 and args.coord2:
            lat1, lon1 = args.coord1
            lat2, lon2 = args.coord2
        else:
            print("Enter two corners of the bounding box (WGS84).")
            print("Tip: SW corner first, then NE corner.")
            print()
            lat1, lon1 = _prompt_coord("Corner 1")
            lat2, lon2 = _prompt_coord("Corner 2")

        min_lat = min(lat1, lat2)
        max_lat = max(lat1, lat2)
        min_lon = min(lon1, lon2)
        max_lon = max(lon1, lon2)

        bbox = (min_lon, min_lat, max_lon, max_lat)
        print(f"\nBounding box: lon [{min_lon:.4f}, {max_lon:.4f}]  "
              f"lat [{min_lat:.4f}, {max_lat:.4f}]")
        print("Downloading terrain from LANDFIRE…\n")
        terrain = download_terrain(bbox=bbox, out_dir=args.cache)

    print(f"\nTerrain loaded: {terrain.shape[0]}×{terrain.shape[1]} cells "
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
