import numpy as np
from scipy.interpolate import interp1d


def _set_weights(w, n):
    """Return w if it is a valid weight vector of size n, and a vector of n 1s
    otherwise.
    """
    if w is None or len(w) != n:
        w = np.ones((n, ))
    return w


def _init_avg(TS, barycenter_size):
    #if size of barycenter matches size of series dimension, return avarage
    if TS.shape[1] == barycenter_size:
        return np.nanmean(TS, axis=0)
    #otherwise: calculate avarage on TS, then sample x-axis to intervals equal barycenter dimension and interpolate
    else:
        TS_avg = np.nanmean(TS, axis=0)
        xnew = np.linspace(0, 1, barycenter_size)
        f = interp1d(np.linspace(0, 1, TS_avg.shape[0]), TS_avg,
                     kind="linear", axis=0)
        return f(xnew)


def dtw_barycenter_(TS,barycenter_size=None,init_barycenter=None,
                    max_iter = 50, eps = 0.00001,weights=None,
                    metric_params = None, verbose = False):
    """
    DTW Barycenter Averaging (DBA) method estimated through
    Expectation-Maximization algorithm.
    :param TS: array-like,
        shape(samples x series)
    :param barycenter_size: <int or None>
        size of barycenter to generate; if None -> same dimension as input series
    :param init_barycenter: <array or None>
        initial barycenter in optimization process; default None
    :param max_iter: <int>
        number of max iteration in Expectation-Maximization algorithm.
    :param eps: <float>
        used for early stopping of Expectation-Maximization algorithm if cost decrease is lower than eps.
    :param weights: <array or None>
        weight of each TS[i]; if None -> uniform weights are assumed
    :param metric_params: <dict or None>
        parameters that controls constrains in DTW computation
    :param verbose: <boolean>
        controls if each iteration result are printed or not.
    :return: <numpy.array>
        barycenter of the provided TS
    """

    if barycenter_size is None:
        barycenter_size  = TS.shape[1]
    weights = _set_weights(weights, TS.shape[0])
    if init_barycenter is None:
        barycenter = _init_avg(TS, barycenter_size)
    else:
        barycenter_size = init_barycenter.shape[0]
        barycenter = init_barycenter
    cost_prev, cost = np.inf, np.inf

    for i in range(max_iter):
        assign = "pa"
        cost = "pc"
