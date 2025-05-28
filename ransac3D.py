"""
Stochastic Plane Fitting

Code adapted from https://github.com/leomariga/pyRANSAC-3D

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import random
import numpy as np

import warnings

warnings.filterwarnings("ignore")

def plane_fit_vectorized(pts, thresh=100, maxIteration=1000, seed=None):
    """
    Find the best-fitting plane to a 3D point cloud using RANSAC (vectorized version).
    Runs faster than plane_fit version below but requires more ram.

    Args:
        pts (numpy.ndarray): Array of shape (N, 3) where each row is a 3D point.
        thresh (float): Distance threshold to consider a point as an inlier (default: 100).
        maxIteration (int): Number of random plane hypotheses to test (default: 1000).

    Returns:
        tuple:
            - equation (numpy.ndarray): Plane equation parameters [A, B, C, D].
            - inliers (numpy.ndarray): Indices of the inlier points.
    """

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected input of shape (N, 3), got {pts.shape}")

    if seed is not None:
        np.random.seed(seed)

    n_points = pts.shape[0]

    # Step 1: Generate maxIteration sets of 3 unique indices
    triplet_indices = np.stack([
        np.random.choice(n_points, size=(3,), replace=False)
        for _ in range(maxIteration)
    ])

    # Step 2: Extract triplets of shape (maxIteration, 3, 3)
    triplets = pts[triplet_indices]  # shape (maxIter, 3, 3)

    # Step 3: Compute normal vectors via cross product of (pt2 - pt1) x (pt3 - pt1)
    vecA = triplets[:, 1, :] - triplets[:, 0, :]
    vecB = triplets[:, 2, :] - triplets[:, 0, :]
    normals = np.cross(vecA, vecB)

    # Normalize normals
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norm[:, 0] > 1e-8  # Filter degenerate planes
    normals = normals[valid]
    triplets = triplets[valid]
    norm = norm[valid]

    normals = normals / norm  # unit normals (M, 3)

    # Step 4: Compute D from plane equation Ax + By + Cz + D = 0
    D = -np.einsum("ij,ij->i", normals, triplets[:, 0, :])  # dot(n, p0)

    # Step 5: Compute distances from all pts to all planes
    # (M, N) = dot(normals, pts.T) + D[:, None]
    dist = (normals @ pts.T + D[:, None])  # signed distance from each pt to each plane

    # Step 6: Count inliers (abs distance <= thresh)
    inlier_mask = np.abs(dist) <= thresh
    inlier_counts = inlier_mask.sum(axis=1)

    # Step 7: Pick the plane with the most inliers
    best_idx = np.argmax(inlier_counts)
    best_eq = np.concatenate([normals[best_idx], [D[best_idx]]])
    best_inliers = np.where(inlier_mask[best_idx])[0]

    return best_eq, best_inliers



def plane_fit(pts, thresh=100, maxIteration=1000):
    """
    Find the best-fitting plane to a 3D point cloud using RANSAC (In serial).
    Slower -- but uses less ram if that's a consideration

    This function applies the RANSAC algorithm to find the plane that best fits a set of 3D points,
    iterating over random samples of points to estimate the plane equation.

    Args:
        pts (numpy.ndarray): 3D point cloud as a `np.array (N, 3)` where each row is a point [x, y, z].
        thresh (int or float): Threshold distance (in meters) from the plane considered as inliers. Points within this distance
                        from the plane are considered inliers (default: 100).
        maxIteration (int): Maximum number of iterations for RANSAC (default: 1000).

    Returns:
        tuple:
            - equation (numpy.ndarray): Plane equation parameters [A, B, C, D] for the plane equation
              `Ax + By + Cz + D = 0` as a `np.array (1, 4)`.
            - inliers (numpy.ndarray): Indices of the points from `pts` that are considered inliers.

    Note:
        ** Modified from https://github.com/leomariga/pyRANSAC-3D/ **

    Example:
        ```python
        import pandas as pd
        file = 'catalog.csv' # some rows x,y,depth_m all in meters
        data = pd.read_csv(file)
        cluster_coords = data[["x", "y", "depth_m"]].values
        plane_params, inliers = plane_fit(
            cluster_coords,
            thresh=250,
            maxIteration=1000,
        )
        ```
    """

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected input of shape (N, 3) got {pts.shape}")

    n_points = pts.shape[0]
    best_eq = []
    best_inliers = []

    for it in range(maxIteration):
        # Samples 3 random points
        id_samples = random.sample(range(n_points), 3)
        pt_samples = pts[id_samples]

        # We have to find the plane equation described by those 3 points
        vecA = pt_samples[1, :] - pt_samples[0, :]
        vecB = pt_samples[2, :] - pt_samples[0, :]

        # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
        vecC = np.cross(vecA, vecB)

        # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
        # We have to use a point to find k
        vecC = vecC / np.linalg.norm(vecC)
        k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
        plane_eq = [vecC[0], vecC[1], vecC[2], k]

        # Distance from a point to a plane
        # https://mathworld.wolfram.com/Point-PlaneDistance.html
        pt_id_inliers = []  # list of inliers ids
        dist_pt = (
            plane_eq[0] * pts[:, 0]
            + plane_eq[1] * pts[:, 1]
            + plane_eq[2] * pts[:, 2]
            + plane_eq[3]
        ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

        # Select indexes where distance is biggers than the threshold
        pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
        if len(pt_id_inliers) > len(best_inliers):
            best_eq = plane_eq
            best_inliers = pt_id_inliers

    return np.array(best_eq), np.array(best_inliers)


