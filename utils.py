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
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.spatial.transform import Rotation as R
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


def convert_latlon_to_utm(lat, lon):
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
    transformer = Transformer.from_proj("epsg:4326", utm_crs, always_xy=True)
    return transformer.transform(lon, lat)
