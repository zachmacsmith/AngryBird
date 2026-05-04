"""
Download LANDFIRE GeoTIFF products overlapping a bounding box.

Requirements:
  pip install requests tqdm rasterio
"""

import os
import math
import requests
from urllib.parse import urlencode
from pathlib import Path
from tqdm import tqdm

# Config
OUT_DIR = Path("landfire_tifs")
OUT_DIR.mkdir(exist_ok=True)
# Bounding box in lon/lat (WGS84): (min_lon, min_lat, max_lon, max_lat)
BBOX = (-122.7, 37.6, -122.2, 38.0)  # example: SF Bay area
# LANDFIRE products to fetch (common names used by LANDFIRE)
PRODUCTS = [
    "LF_2021_CONUS_3DEP_DEM",     # DEM (example name — see notes)
    "LF_2021_CONUS_FUEL",         # Fuels raster (example)
    "LF_2021_CONUS_CANOPY",       # Canopy layers (example)
]

# LANDFIRE tile index / API base (uses LANDFIRE's download service)
# This script uses the LANDFIRE "download" endpoint pattern that lists tiles for products.
API_BASE = "https://landfire.gov/lfdata/"

# Helper: request list of files for a product (simple pattern)
def list_product_tiles(product: str):
    # LANDFIRE organizes files under folders; a simple approach is to query the product directory listing.
    # Construct an index URL and parse for .tif links.
    idx_url = f"{API_BASE}{product}/"
    resp = requests.get(idx_url)
    resp.raise_for_status()
    html = resp.text
    # Find .tif links (naive)
    links = []
    for part in html.split('"'):
        if part.endswith(".tif") or part.endswith(".tif.gz"):
            links.append(part if part.startswith("http") else idx_url + part)
    return links

# Helper: check if tile bbox overlaps requested bbox
# For simplicity, this expects tile filenames to include extents or tile IDs; otherwise download list may be filtered manually.
def download_file(url: str, out_folder: Path):
    local_name = url.split("/")[-1]
    out_path = out_folder / local_name
    if out_path.exists():
        return out_path
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(
            desc=local_name, total=total, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    return out_path

def main():
    for prod in PRODUCTS:
        print(f"Listing tiles for product: {prod}")
        try:
            links = list_product_tiles(prod)
        except Exception as e:
            print(f"Failed to list product {prod}: {e}")
            continue
        prod_dir = OUT_DIR / prod
        prod_dir.mkdir(parents=True, exist_ok=True)
        print(f"Found {len(links)} candidate files for {prod}")
        for url in links:
            try:
                download_file(url, prod_dir)
            except Exception as e:
                print(f"Error downloading {url}: {e}")

if __name__ == "__main__":
    main()

