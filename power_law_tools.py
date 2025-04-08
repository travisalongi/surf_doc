"""
Power Law Tools for Fitting and Estimation

This module provides functions to fit a power-law distribution to data using
Maximum Likelihood Estimation (MLE) and evaluate the goodness of fit. It includes
tools for computing histograms, calculating relative errors, and performing the
MLE optimization.

Author: Travis Alongi (talongi@usgs.gov)
"""

import numpy as np
from scipy.optimize import minimize
from surf.utils import midpoint

import warnings

warnings.filterwarnings("ignore")


def pj_func(x, v0, d, gamma):
    """
    Modified power-law (Powers & Jordan, 2010)
    See original manuscript:
        Powers, P. M., & Jordan, T. H. (2010). Distribution of seismicity
        across strike‐slip faults in California. 
        Journal of Geophysical Research: Solid Earth, 115(B5).

    This function computes the probability density for a power-law distribution,
    given the parameters v0 (scale), d (cutoff), and gamma (exponent).

    Args:
        x (np.ndarray or list): Input values for which to compute the power-law probability.
        v0 (float): Scale parameter.
        d (float): Cutoff parameter (typically the characteristic length scale).
        gamma (float): Exponent parameter of the power-law distribution.

    Returns:
        array: Computed power-law values for the input `x`.

    Example:
        ```python
        import numpy as np
        x_values = np.linspace(1, 10, 100)
        v0, d, gamma = 1.0, 2.0, 1.5
        y_values = pj_func(x_values, v0, d, gamma)
        ```
    """
    v_x = v0 * (d**2 / ((x ** 2) + (d**2))) ** (gamma / 2)
    return v_x


def log_likelihood(params, x, n_obs):
    """
    Log-likelihood function for Poisson-distributed counts.

    This function calculates the log-likelihood of the observed data, assuming
    the data follows a Poisson distribution based on a power-law model (Powers & Jordan, 2010).

    Args:
        params (np.ndarray or list): Parameters [v0, d, gamma] for the power-law model.
        x (np.ndarray or list): Input values for which to compute the power-law probability.
        n_obs (np.ndarray or list): Observed data (counts for each bin).

    Returns:
        float: Negative log-likelihood for the given data and model parameters.

    Example:
        ```python
        import numpy as np
        params = [1.0, 2.0, 1.5]
        x_values = np.linspace(1, 10, 100)
        n_obs = np.random.poisson(lam=pj_func(x_values, *params))
        log_L = log_likelihood(params, x_values, n_obs)
        ```
    """
    phi_0, d, gamma = params
    n_pred = pj_func(x, phi_0, d, gamma)
    # Log-likelihood for Poisson-distributed counts
    log_L = np.sum(n_obs * np.log(n_pred) - n_pred)
    return -log_L  # Negative for minimization


def maximum_likelihood_estimation(bins, n_obs, initial_params, solver_method="Nelder-Mead"):
    """
    Maximum Likelihood Estimation for power-law parameters.

    This function estimates the parameters of a power-law distribution by
    minimizing the negative log-likelihood using the given initial parameters.

    Args:
        bins (np.ndarray or list): The bins or values for which the power-law is fitted.
        n_obs (array-like/int): The observed counts for each bin.
        initial_params (np.ndarray or list): Initial guesses for the parameters [phi_0, d, gamma].
        method (str, optional): The optimization method to use (default is 'Nelder-Mead'). see scipy.optimize

    Returns:
        result: Optimization result from `scipy.optimize.minimize`, containing the estimated parameters.

    Example:
        ```python
        bins = np.linspace(1, 10, 10)
        n_obs = np.array([50, 40, 30, 20, 15, 10, 8, 5, 3])
        initial_params = [1.0, 2.0, 1.5]
        result = maximum_likelihood_estimation(bins, n_obs, initial_params)
        print(result.x)  # Estimated [v0, d, gamma]
        ```
    """
    result = minimize(log_likelihood, initial_params, args=(bins, n_obs), method=solver_method)
    return result


def mean_relative_error(obs, fit):
    """
    Mean relative error between observed and fitted data.

    This function calculates the mean relative error, which is the sum of absolute
    errors normalized by the observed values, divided by the number of observations.

    Args:
        obs (np.ndarray or list): Observed data (counts or measurements).
        fit (np.ndarray or list): Fitted data (predicted values from a model).

    Returns:
        float: Mean relative error between the observed and fitted data.

    Example:
        ```python
        obs = np.array([10, 20, 30, 40, 50])
        fit = np.array([9, 19, 31, 42, 48])
        error = mean_relative_error(obs, fit)
        print(f"Mean Relative Error: {error:.4f}")
        ```
    """
    rel_error = np.mean(np.abs(obs - fit) / obs)
    return rel_error


def compute_histogram(dist_data, bins):
    """
    Compute histogram of fault distances and normalize.

    This function computes a histogram for the given distances and normalizes 
    the counts by the bin width. It also removes bins with zero counts.

    Args:
        dist_data (np.ndarray or list): Fault distances or other data to be histogrammed.
        bins (np.ndarray or list): The bin edges for the histogram.

    Returns:
        tuple: 
            - xmids (np.ndarray or list): Midpoints of the bins (excluding zero-count bins).
            - n_obs_norm_no_zeros (np.ndarray or list): Normalized counts, excluding bins with zero counts.

    Example:
        ```python
        dist_data = np.random.exponential(scale=5, size=1000)
        bins = np.linspace(0, 20, 21)
        xmids, c
        ```
    """
    H = np.histogram(np.abs(dist_data), bins=bins, density=False)
    n_obs_norm = H[0] / np.diff(H[1])  # Normalize by bin width
    n_obs_norm_no_zeros = n_obs_norm[n_obs_norm != 0]
    xmids = midpoint(H[1])[n_obs_norm != 0]
    return xmids, n_obs_norm_no_zeros


def fit_power_law(dist_data, bins, init_params,solver_method="Nelder-Mead"):
    """
    Fit a power-law distribution using Maximum Likelihood Estimation.

    This function fits a power-law model to the provided data (using MLE) and 
    calculates the mean relative error between the observed and fitted values.

    Args:
        dist_data (np.ndarray or list): Fault distances or other data to be histogrammed.
        bins (np.ndarray or list): The bin edges for the histogram.
        init_params (np.ndarray or list): Initial guesses for the power-law parameters [v0, d, gamma].

    Returns:
        tuple: 
            - estimated_params (np.ndarray or list): The estimated power-law parameters [v0, d, gamma].
            - mle_fit (np.ndarray or list): The fitted values based on the estimated parameters.
            - mre (float): The mean relative error between the observed and fitted values.

    """
    xmids,n_obs_norm_no_zeros = compute_histogram(dist_data,bins)
    result = maximum_likelihood_estimation(xmids, n_obs_norm_no_zeros, init_params, solver_method=solver_method)
    estimated_params = result.x
    mle_fit = pj_func(xmids, *estimated_params)
    mre = mean_relative_error(n_obs_norm_no_zeros, mle_fit)
    return estimated_params, mle_fit, mre
