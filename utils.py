"""
utils.py

General-purpose utility functions for file management, numerical operations,
and geospatial transformations.

Note:
    This module is intended for internal use within the library,
    but may be imported elsewhere as needed.

Author: Travis Alongi (talongi@usgs.gov)
"""

import os
import gzip
import glob
import shutil
import requests
import subprocess

import numpy as np
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info


def setup_output_directory(output_dir, delete_existing=False):
    """
    Ensure output directory exists and optionally clear existing files.

    Args:
        output_dir (str or Path-like): Directory where data will be stored.
        delete_existing (bool, optional): If True, delete all files in the output directory.
                                          Defaults to False.

    Example:
        setup_output_directory("results/")  # creates the dir if needed, doesn't delete
        setup_output_directory("results/", delete_existing=True)  # deletes files first
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if delete_existing:
        for file in output_dir.iterdir():
            if file.is_file():
                file.unlink()


def midpoint(array):
    """
    Calculates midpoints between adjacent items in an array.

    This function returns the midpoints of adjacent values in the input array.

    Args:
        array (numpy.ndarray): A 1D array of numerical values (floats or ints).

    Returns:
        array: A 1D array of midpoints between adjacent values.

    Example:
        ```python
        import numpy as np
        values = np.array([0, 10, 20, 30])
        midpoints = midpoint(values)
        print(midpoints)  # Output: [5. 15. 25.]
        ```
    """
    y = (array[1:] + array[:-1]) / 2
    return y


def convert_latlon_to_utm(lat, lon,return_crs=False):
    """
    Converts latitude and longitude arrays to UTM coordinates.

    This function transforms latitude and longitude values to UTM (Universal
    Transverse Mercator) coordinates using the appropriate UTM projection.

    Args:
        lat (array-like): Latitude values (in degrees).
        lon (array-like): Longitude values (in degrees).

    Returns:
        tuple: A tuple of UTM X and Y coordinates.

    Example:
        ```python
        import numpy as np
        latitudes = np.array([37.0, 37.1])
        longitudes = np.array([-121.9, -121.8])
        utm_x, utm_y = convert_latlon_to_utm(latitudes, longitudes)
        print(utm_x, utm_y)
        ```
    """
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            south_lat_degree=lat.min(),
            north_lat_degree=lat.max(),
            east_lon_degree=lon.max(),
            west_lon_degree=lon.min(),
        ),
    )
    utm_crs = CRS.from_epsg(utm_crs_list[0].code)
    # print(f'Converted Lat/Lon to {utm_crs}')
    transformer = Transformer.from_proj("epsg:4326", utm_crs, always_xy=True)
    if return_crs is True:
        return transformer.transform(lon, lat), utm_crs
    else:
        return transformer.transform(lon, lat)


def download_file(url: str, dest_path: str) -> str:
    """
    Download a file from a URL if it doesn't already exist.

    Args:
        url: The URL to download from.
        dest_path: Full path to save the file (including filename).

    Returns:
        Path to the downloaded file, or None if failed.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path):
        print(f"{os.path.basename(dest_path)} already exists. Skipping download.")
        return dest_path

    print(f"Downloading {os.path.basename(dest_path)}...")

    try:
        response = requests.get(url, stream=True, timeout=10,verify=False)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {os.path.basename(dest_path)}: {e}")
        return None

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with (
        open(dest_path, "wb") as file,
        tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=os.path.basename(dest_path),
        ) as bar,
    ):
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

    print(f"Downloaded to {dest_path}")
    return dest_path


def decompress_gz_file(
    gz_path: str, dest_path: str | None = None, overwrite: bool = False
) -> str:
    """Decompress a .gz file to the destination path."""
    if dest_path is None:
        dest_path = gz_path[:-3]  # Remove .gz

    if os.path.exists(dest_path) and not overwrite:
        print(f"{dest_path} already exists. Skipping decompression.")
        return dest_path

    print(f"Decompressing {gz_path}...")

    with gzip.open(gz_path, "rb") as f_in, open(dest_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    print(f"Decompressed to {dest_path}")
    return dest_path


def convert_dd_catalog_to_csv(
    input_txt: str, output_csv: str, skip_lines: int = 97, header_names_list=None
):
    """
    Convert a Northern California DD catalog .txt file to a CSV.

    Parameters:
    ----------
    input_txt : str
        Path to the downloaded DD catalog file.
    output_csv : str
        Path to output CSV file.
    """
    if header_names_list is None:
        header_names = [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "lat",
            "lon",
            "depth_km",
            "eh1_km",
            "eh2_km",
            "azimuth_deg",
            "ez_km",
            "mag",
            "event_id",
        ]
        header_line = ",".join(header_names)

    else:
        header_names = header_names_list
        header_line = ",".join(header_names)

    # Awk is just easier for this than python
    awk_command = (
        f"awk 'NR > {skip_lines} {{ "
        f"for (i = 1; i <= NF; i++) {{ "
        f'printf "%s%s", $i, (i==NF ? "\\n" : ",") '
        f"}} "
        f"}}' {input_txt}"
    )

    # Get the total number of lines in the input file to use with tqdm
    total_lines = subprocess.check_output(f"wc -l < {input_txt}", shell=True)
    total_lines = int(total_lines.strip())

    # Open output file, write the header, then run awk with tqdm progress bar
    with open(output_csv, "w") as out:
        out.write(header_line + "\n")  # Write header row
        # Run awk with a subprocess, updating tqdm with each line processed
        with subprocess.Popen(awk_command, shell=True, stdout=subprocess.PIPE) as proc:
            # Use tqdm to track progress based on the total number of lines
            with tqdm(total=total_lines - 97, desc="Processing file") as pbar:
                for line in proc.stdout:
                    out.write(line.decode("utf-8"))  # Write output to CSV
                    pbar.update(1)

    print(f"CSV with headers written to {output_csv}")


def remove_temp_files(directory: str = "./input", pattern: str = "*temp*"):
    """Removes temporary files"""
    files_to_remove = glob.glob(os.path.join(directory, pattern))
    for file in files_to_remove:
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except Exception as e:
            print(f"Failed to remove {file}: {e}")
