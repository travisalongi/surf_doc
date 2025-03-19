"""
Stochastic Plane Fitting
"""

import random
import numpy as np

def plane_fit(pts, thresh=100, maxIteration=1000):
    """
    Find the best-fitting plane to a 3D point cloud using RANSAC.

    This function applies the RANSAC algorithm to find the plane that best fits a set of 3D points, 
    iterating over random samples of points to estimate the plane equation.

    Args:
        pts (NumPy array): 3D point cloud as a `np.array (N, 3)` where each row is a point [x, y, z].
        thresh (float): Threshold distance from the plane considered as inliers. Points within this distance 
                        from the plane are considered inliers (default: 100).
        maxIteration (int): Maximum number of iterations for RANSAC (default: 1000).

    Returns:
        tuple: 
            - equation (NumPy array): Plane equation parameters [A, B, C, D] for the plane equation 
              `Ax + By + Cz + D = 0` as a `np.array (1, 4)`.
            - inliers (list): Indices of the points from `pts` that are considered inliers.

    Note:
        ** Modified from https://github.com/leomariga/pyRANSAC-3D/ **
    """
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
            plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
        ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

        # Select indexes where distance is biggers than the threshold
        pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
        if len(pt_id_inliers) > len(best_inliers):
            best_eq = plane_eq
            best_inliers = pt_id_inliers

    return best_eq, best_inliers
