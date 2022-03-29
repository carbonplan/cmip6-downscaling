from __future__ import annotations

import numpy as np
import xarray as xr


def weighted_mean(ds, *args, **kwargs):
    weights = ds.time.dt.days_in_month
    return ds.weighted(weights).mean(dim='time')


def days_temperature_threshold(
    ds: xr.Dataset, threshold_direction: str, value: float
) -> xr.Dataset:
    """Calculate whether your day is over a threshold

    Parameters
    ----------
    ds : xr.Dataset
        Daily dataset
    threshold_direction : str
        either 'over' or 'under'
    value: float
        threshold to be used for comparison

    Returns
    -------
    xr.Dataset
        boolean mask of the annual average days over/under a threshold
    """
    if threshold_direction == 'over':
        return (ds > value).groupby('time.year').sum().mean(dim='year')
    elif threshold_direction == 'under':
        return (ds < value).groupby('time.year').sum().mean(dim='year')
    else:
        raise NotImplementedError


def is_wet_day(ds: xr.Dataset, threshold: float = 0.01) -> xr.Dataset:
    """Calculate whether your day is over a threshold

    Parameters
    ----------
    ds : xr.Dataset
        Dailiy dataset of precipitation
    threshold : float, optional
        the threshold for precipitation (mm/day) above which a day is consisdered wet, by default 0.01

    Returns
    -------
    xr.Dataset
        boolean mask of when the dataset is over the threshold
    """
    return ds > threshold


def metric_calc(ds: xr.Dataset, metric: str, dim: str = 'time', skipna: bool = False) -> xr.Dataset:
    """Calculate metric along a dimension in a dataset (most
    likely you'll want to analyze along time)

    Parameters
    ----------
    ds : xr.Dataset
        dataset of climate information
    metric : str
        the metric of interest (e.g. 'mean')
    dim : str
        dimension to calculate your metric along, by default 'time'
    skipna : bool
        whether to skip nans in your metric calculations

    Returns
    -------
    xr.Dataset
        dataset collapsed along the dimension `dim`
    """
    if metric in {'mean', 'median', 'std'}:
        return getattr(ds, metric)(dim=dim, skipna=skipna)
    elif 'percentile' in metric:
        # parse the percentile
        percentile = float(metric.split('percentile')[1]) / 100
        ds = ds.chunk({dim: -1})
        return ds.quantile(percentile, dim=dim, skipna=skipna)
    else:
        raise NotImplementedError


def wet_day_amount(ds: xr.Dataset, method: str = 'mean', threshold: float = 0.01) -> xr.Dataset:
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
        Metric describing statistics of wet days
    """
    assert 'pr' in ds, 'Precipitation not in dataset'
    wet_day = ds.where(is_wet_day(ds, threshold=threshold))
    return metric_calc(wet_day, method, skipna=True)


def probability_two_consecutive_days(
    ds: xr.Dataset, kind_of_day: str = 'wet', threshold: float = 0.01
) -> xr.Dataset:
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


def probability_wet(ds: xr.Dataset, threshold: float = 0.01) -> xr.Dataset:
    """
    Unconditional probability of a day being wet

    Parameters
    ----------
    ds : xr.Dataset
        Daily dataset with precipitation
    threshold : float, optional
        Threshold to determine whether day is wet or dry, by default 0.01

    Returns
    -------
    xr.Dataset
        Probability of having a wet day (collapsed along time dimension)
    """
    return is_wet_day(ds, threshold=threshold).mean(dim='time')


def spell_length_stat(timeseries: np.array, method: str = 'mean') -> float:
    """
    Statistic about length of a spell of boolean values.

    Parameters
    ----------
    timeseries : array_like
        Array of booleans denoting a day being (for example) wet or dry.
    method : str
        String corresponding to the final statistic you want (mean, std, or percentile)

    Returns
    -------
    float
        Statistic (e.g. mean) of a length of consecutive trues
    """
    # add a 0 to help with spells at beginning or end of series
    timeseries = np.append(timeseries, [0])
    yesterday = np.roll(timeseries, 1)
    changes = timeseries - yesterday
    spell_starts = np.argwhere(changes == 1)
    spell_ends = np.argwhere(changes == -1)
    spells = spell_ends - spell_starts
    if method == 'mean':
        return np.mean(spells)
    elif method == 'std':
        return np.std(spells)
    elif 'percentile' in method:
        # parse the percentile
        percentile = float(method.split('percentile')[1]) / 100
        return np.quantile(spells, percentile)
    else:
        raise NotImplementedError


def apply_spell_length(wet_days: xr.Dataset, metric: str = 'mean') -> xr.Dataset:
    """
    Calculate statistic about wet spell length

    Parameters
    ----------
    wet_days : xr.Dataset
        Boolean dataset of whether days were wet or not
    method : str
        String corresponding to the final statistic you want (mean, std, or percentile)

    Returns
    -------
    xr.Dataset
        Dataset collapsed along time dimension with statistic of spell length
    """
    wet_spell_length = xr.apply_ufunc(
        spell_length_stat, wet_days, metric, input_core_dims=[['time'], []], vectorize=True
    )
    return wet_spell_length


def monthly_variability(ds: xr.Dataset, method: str = 'sum') -> xr.Dataset:
    """
    Calculate monthly variability

    Parameters
    ----------
    ds : xr.Dataset
        Dataset of daily data
    method : str
        String corresponding to the final statistic you want, by default 'sum'

    Returns
    -------
    xr.Dataset
        Statistics of monthly variability along time dimension
    """
    if method not in {'sum', 'mean'}:
        raise ValueError('Must select sum or mean')

    return getattr(ds.resample(time='1M'), method)().std(dim='time')
