import numpy as np
from esda.moran import Moran
import xarray as xr

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

def days_temperature_threshold(ds, threshold_direction, value):
    if threshold_direction == 'over':
        return (ds > value).groupby('time.year').sum().mean(dim='year')
    elif threshold_direction == 'under':
        return (ds < value).groupby('time.year').sum().mean(dim='year')

def is_wet_day(ds : xr.Dataset, threshold : float = 0.01):
    """[summary]

    Parameters
    ----------
    ds : xr.Dataset
        [description]
    threshold : float, optional
        the threshold for precipitation (mm/day) above which a day is consisdered wet, by default 0.01
    """
    return  ds > threshold


def wet_day_stat(ds : xr.Dataset, method : str = 'mean', threshold : float = 0.01):
    """Extract days that received precipitation above a given threshold
    and then do statistics on them.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset including precipitation at a daily temporal scale.
    method : str, optional
        statistic you want to assess of wet days, by default 'mean'
    threshold : float, optional
        the threshold for precipitation (mm/day) above which a day is consisdered wet, by default 0.01

    Returns
    -------
    xr.Dataset
        

    Raises
    ------
    NotImplementedError
        [description]
    """
    assert 'pr' in ds,'Precipitation not in dataset'
    wet_day = ds.where(is_wet_day(ds, threshold=threshold))
    skipna = True
    if method=='mean':
        return wet_day.mean(dim='time', skipna=skipna)
    elif method=='median':
        return wet_day.median(dim='time', skipna=skipna)
    elif method=='std':
        return wet_day.std(dim='time', skipna=skipna)
    elif 'percentile' in method:
        # parse the percentile
        percentile = float(method.split('percentile')[1])/100
        wet_day = wet_day.chunk({'time': -1})
        return wet_day.quantile(percentile, dim='time', skipna=skipna)
    else:
        raise NotImplementedError

def probability_two_consecutive_days(ds : xr.Dataset, kind_of_day : str = 'wet', threshold : float = 0.01):
    """Contitional probability of a day being the same as the day before.
    So, given the fact that it was wet yesterday, what is the probability it will be wet today.
    Or, if it was dry yesterday, what is probability it will be dry today.

    Parameters
    ----------
    ds : xr.Dataset
        dataset including daily precipitation
    kind_of_day : str, optional
        The kind of day you are interested in, by default 'wet'
    threshold : float, optional
        Threshold to determine whether day is wet or dry, by default 0.01

    Returns
    -------
    xr.Dataset
        Probability of a day being the same as the day before it
    """
    if kind_of_day == 'wet':
        valid_days = is_wet_day(ds, threshold=threshold)
    elif kind_of_day == 'dry':
         valid_days = ~is_wet_day(ds, threshold=threshold)
    yesterday_valid = valid_days.roll({'time': 1})
    return valid_days.where(yesterday_valid).mean(dim='time', skipna=True)
    
def probability_wet(ds : xr.Dataset, threshold: float = 0.01):
    """
    Unconditional probability of a day being wet

    Parameters
    ----------
    ds : xr.Dataset
        Daily dataset with precipitation
    threshold : float, optional
        Threshold to determine whether day is wet or dry, by default 0.01
    """
    return is_wet_day(ds, threshold=threshold).mean(dim='time')

def spell_length_stat(arr, method='mean'):
    s = ''.join( [str(int(i)) for i in arr] )
    parts = s.split('0')
    spells = [len(p) for p in parts if len(p) > 0]
    if method=='mean':
        return np.mean(spells)
    elif method=='std':
        return np.std(spells)
    elif 'percentile' in method:
        # parse the percentile
        percentile = float(method.split('percentile')[1])/100
        return np.quantile(spells, percentile)

def monthly_variability(ds, method='sum'):
    if method=='sum':
        return ds.groupby('time.month').sum().std(dim='time')
    elif method=='mean':
        return ds.groupby('time.month').sum().std(dim='time')
