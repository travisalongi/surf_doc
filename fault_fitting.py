"""
fault_fitting.py

This module contains functions to process the merged seismicity
and fit the fault surfaces using Support Vector Regression (SVR),
calculate fault orientations, and prepare data for further analysis
and visualization.

Author: Travis Alongi (talongi@usgs.gov) 
"""

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.spatial.transform import Rotation as R


def stdip_from_norm(array):
    """
    Calculates strike and dip from a normal vector (array).

    The function calculates the strike and dip based on the normal vector,
    following the right-hand rule. The strike is measured from the positive
    x-axis, and the dip is the angle between the surface and the horizontal plane.

    Args:
        array (numpy.ndarray): A 3D vector representing the normal vector (x, y, z).

    Returns:
        tuple:
            - strike (float): The strike of the fault surface in degrees (0 to 360).
            - dip (float): The dip of the fault surface in degrees (0 to 90).
    Example:
        ```python
        import numpy as np
        normal_vector = np.array([0.5, 0.5, -1])
        strike, dip = stdip_from_norm(normal_vector)
        print(f"Strike: {strike:.2f}, Dip: {dip:.2f}")
        ```
    """
    rhr = 0

    # Check that array is correct shape
    if array.shape[0] != 3:
        raise ValueError(f"Expected input of shape (3,), got {array.shape}")

    # Checks if vector is pointing up and adjusts calculation to RHR
    if array[-1] > 0:
        array = -1 * array
        rhr = 180
    r = np.linalg.norm(array)  # Magnitude
    x, y, z = array / r  # Normalized
    st = (-np.rad2deg(np.arctan2(y + 0.0000001, x + 0.0000001))) % 360
    dip = rhr - np.rad2deg(np.arccos(z))
    if dip < 0:
        st = (st + 180) % 360
        dip = 180 + dip
    return st, dip


def fit_fault_surface(
    file,
    n_iter,
    xstrap_fraction,
    grid_spacing,
    svr_cross_validation_distribution={
        "C": uniform(1, 100),
        "epsilon": uniform(50, 600),
    },
    max_subsurface_depth_meters = -15000,
):
    """
    Fit a surface from fault data using Support Vector Regression (SVR).

    This function reads fault data from a CSV file, calculates the fault's
    strike and dip, fits a surface using SVR, and extrapolates the surface
    beyond the observed data. It returns the gridded surface coordinates and
    fault orientation.

    Args:
        file (str or path-like): The file containing fault data with x, y, depth_m columns. 
            x and y are assumed to be in UTM coordinates
        n_iter (int): The number of iterations for cross-validation to determine the best SVR parameters.
        xstrap_fraction (float): The fraction of the fault's dimensions to extrapolate beyond.
        grid_spacing (float or int): The desired spacing between points in the output grid.
        svr_cross_validation_distribution (dict): A dictionary of C and epsilon values for SVR cross-validation.
        max_subsurface_depth_meters (float or in): The maximum depth in meters to build faults too (default: -15,000 meters or -15 km)

    Returns:
        tuple:
            - numpy.ndarray: The gridded surface data (x, y, z columns).
            - float: The strike of the surface in degrees.
            - float: The dip of the surface in degrees.
            - dict: The best SVR parameters (C and epsilon).
            - pandas.DataFrame: 

    Example:
        ```python
        svr_params = {"C": [1, 10, 100], "epsilon": [0.01, 0.1, 1]}
        result = fit_fault_surface(
            "fault_data.csv", svr_params, n_iter=10, xstrap_fraction=0.2, grid_spacing=100
        )
        grid_data, strike, dip, best_params = result
        print(f"Strike: {strike:.2f}, Dip: {dip:.2f}")
        print(f"Best SVR Params: {best_params}")
        ```
    """
    df = pd.read_csv(file)
    fault_id = Path(file).stem

    # x,y are assumed to be utm coordinates
    coords = np.array([df.x, df.y, df.depth_m]).T

    centroid = np.mean(coords, axis=0)
    coords_centered = coords - centroid

    cov_matrix = np.cov(coords_centered.T)
    _, _, Vt = np.linalg.svd(cov_matrix)
    normal_vector = Vt[-1]

    strike, dip = stdip_from_norm(normal_vector)
    print(f"FaultID--{fault_id} Strike = {strike} & Dip = {dip}")

    z_axis = np.array([0, 0, 1])
    vert_angle = np.arccos(
        np.dot(normal_vector, z_axis) / np.linalg.norm(normal_vector)
    )

    axis_of_rotation = np.cross(normal_vector, z_axis)
    axis_of_rotation = axis_of_rotation / np.linalg.norm(axis_of_rotation)

    rotation_vert = R.from_rotvec(vert_angle * axis_of_rotation)

    coords_xy_plane = rotation_vert.apply(coords_centered)

    cov_matrix = np.cov(coords_xy_plane.T)
    _, _, Vt = np.linalg.svd(cov_matrix)

    v1 = Vt[1]
    x_axis = np.array([1, 0, 0])
    hor_angle = np.arccos(np.dot(v1, x_axis) / np.linalg.norm(v1))

    rotation_hor = R.from_rotvec(hor_angle * z_axis)
    coords_xy_aligned_yaxis = rotation_hor.apply(coords_xy_plane)

    x, y, z = (
        coords_xy_aligned_yaxis[:, 0],
        coords_xy_aligned_yaxis[:, 1],
        coords_xy_aligned_yaxis[:, 2],
    )

    svr = RandomizedSearchCV(
        SVR(kernel="rbf", gamma="scale"),
        svr_cross_validation_distribution,
        n_iter=n_iter,
    )
    svr_rbf = svr.fit(np.array([x, y]).T, z)
    print(f"{fault_id} -- {svr.best_params_}")

    xdist = x.max() - x.min()
    xtrap_xdist = xdist * xstrap_fraction
    ydist = y.max() - y.min()
    xtrap_ydist = ydist * xstrap_fraction

    x_range = np.arange(x.min() - xtrap_xdist, x.max() + xtrap_xdist, grid_spacing)
    y_range = np.arange(y.min() - xtrap_ydist, y.max() + xtrap_ydist, grid_spacing)
    X, Y = np.meshgrid(x_range, y_range)
    Z = svr_rbf.predict(np.array([X.flatten(), Y.flatten()]).T)

    curve_data = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    inverse_vert_rotation = rotation_vert.inv()
    inverse_hor_rotation = rotation_hor.inv()

    curve_data_irot = (
        inverse_vert_rotation.apply(inverse_hor_rotation.apply(curve_data)) + centroid
    )

    fault_depth = curve_data_irot[:, -1]
    depth_mask = np.logical_and(fault_depth < 0, fault_depth > max_subsurface_depth_meters)
    curve_data_irot_masked = curve_data_irot[depth_mask]

    cat_z = coords_xy_aligned_yaxis[:, -1]

    # Calculate distances from surface to each point
    z_prediction = svr_rbf.predict(coords_xy_aligned_yaxis[:, :2])
    z_hat = np.abs(cat_z - z_prediction)
    df["fault_dist"] = z_hat

    return curve_data_irot_masked, strike, dip, svr.best_params_, df


def wrapper_fit_fault_surface(args):
    """
    Wrapper for fit_fault_surface used for parallel processing.

    Args:
        args (tuple): A tuple containing the arguments for `fit_fault_surface`.

    Returns:
        tuple: The same output as `fit_fault_surface`.

    Example:
        ```python
        from multiprocessing import Pool

        fault_files = ["fault1.csv", "fault2.csv"]
        svr_params = {"C": [1, 10, 100], "epsilon": [0.01, 0.1, 1]}
        args_list = [(f, svr_params, 10, 0.2, 100) for f in fault_files]

        with Pool() as pool:
            results = pool.map(wrapper_process_single_fault, args_list)

        for res in results:
            print(f"Strike: {res[1]:.2f}, Dip: {res[2]:.2f}")
        ```
    """
    return fit_fault_surface(*args)



