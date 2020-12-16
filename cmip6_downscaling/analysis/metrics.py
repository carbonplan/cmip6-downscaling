import numpy as np
from esda.moran import Moran


def weighted_mean(ds, *args, **kwargs):
    weights = ds.time.dt.days_in_month
    return ds.weighted(weights).mean(dim='time')


def time_mean(obj):
    obj = obj.resample(time='AS').map(weighted_mean, dim='time')
    return obj.mean('time')


def seasonal_cycle_mean(obj):
    return obj.mean(('x', 'y'))


def seasonal_cycle_std(obj):
    return obj.mean(('x', 'y'))


def moransI(obj, weights=None):
    """
    Inputs:
    obj: xarray object for which you want the autocorrelation
    weights: same shape as obj, but with weights which will be applied to
    obj to scale the autocorrelation. Simplest could be all ones.
    """
    if not weights:
        weights = np.ones_like(obj.values)

    return Moran(obj.values, weights)


# def extreme_scaling(obj, factors=[1, 2, 4, 8, 16], q=0.02):

#     for factor in factors:

#         obj_coarse = obj.coarsen(x=factor, y=factor).mean()
#         obj_quantile = obj_coarse.quantile(q=q).mean(('x', 'y'))


def calc(obj, compute=False):
    """
    This function takes an object and then calculates a
    series of metrics for that object. It returns it
    as a dictionary object of xarray objects. If the compute flag
    is turned on these objects will be in memory. Otherwise the
    computations will be lazy.
    """

    metrics = {}
    if compute:
        metrics['time_mean'] = time_mean(obj).load()
        metrics['seasonal_cycle_mean'] = seasonal_cycle_mean(obj).load()
    else:
        metrics['time_mean'] = time_mean(obj)
        metrics['seasonal_cycle_mean'] = seasonal_cycle_mean(obj)

    return metrics
