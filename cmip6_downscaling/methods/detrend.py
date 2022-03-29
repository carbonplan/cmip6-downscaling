from __future__ import annotations

import pandas as pd


def add_circular_temporal_pad(data, offset, timeunit='D'):
    """
    Pad the beginning of data with the last values of data, and pad the end of data with the first values of data.

    Parameters
    ----------
    data: xr.Dataset
        data to be padded, must have time as a dimension
    offset: int
        number of time points to be padded
    timeunit: str
        unit of offset. Default is 'D' = days

    Returns
    -------
    padded: xr.Dataset
        The padded dataset
    """

    padded = data.pad(time=offset, mode='wrap')

    time_coord = padded.time.values
    time_coord[:offset] = (data.time[:offset] - pd.Timedelta(offset, timeunit)).values
    time_coord[-offset:] = (data.time[-offset:] + pd.Timedelta(offset, timeunit)).values

    padded = padded.assign_coords({'time': time_coord})

    return padded


def days_in_year(year, use_leap_year=False):
    """
    Returns the number of days in the year input.

    Parameters
    ----------
    year: int
        year in question
    use_leap_year: bool
        whether to return 366 for leap years

    Returns
    -------
    n_days_in_year: int
        Number of days in the year of question
    """
    if (year % 4 == 0) and use_leap_year:
        return 366
    return 365
    # there might be calendars with 360 days


def pad_with_edge_year(data):
    """
    Pad data with the first and last year available. similar to the behavior of np.pad(mode='edge')
    but uses the 365 edge values instead of repeating only 1 edge value (366 if leap year)

    Parameters
    ----------
    data: xr.Dataset
        data to be padded, must have time as a dimension

    Returns
    -------
    padded: xr.Dataset
        The padded dataset
    """

    def pad_with(vector, pad_width, iaxis, kwargs):
        pstart = pad_width[0]
        pend = pad_width[1]
        if pstart > 0:
            start = vector[pstart : pstart * 2]
            vector[:pstart] = start
        if pend > 0:
            end = vector[-pend * 2 : -pend]
            vector[-pend:] = end

    # TODO: change use_leap_year to True once we unify the GCMs into Gregorian calendars
    prev_year = days_in_year(data.time[0].dt.year.values - 1, use_leap_year=False)
    next_year = days_in_year(data.time[-1].dt.year.values + 1, use_leap_year=False)
    padded = data.pad({'time': (prev_year, next_year)}, mode=pad_with)
    time_coord = padded.time.values
    time_coord[:prev_year] = (data.time[:prev_year] - pd.Timedelta(prev_year, 'D')).values
    time_coord[-next_year:] = (data.time[-next_year:] + pd.Timedelta(next_year, 'D')).values

    padded = padded.assign_coords({'time': time_coord})

    return padded


def calc_epoch_trend(data, historical_period, day_rolling_window=21, year_rolling_window=31):
    """
    Calculate the epoch trend as a multi-day, multi-year rolling average. The trend is calculated as the anomaly
    against the historical mean (also a multi-day rolling average).

    Parameters
    ----------
    data : xr.Dataset
        Data to calculate epoch trend on
    historical_period: slice
        Slice indicating the historical period to be used when calculating historical mean
    day_rolling_window: int
        Number of days to include when calculating the rolling average
    year_rolling_window: int
        Number of years to include when calculating the rolling average

    Returns
    -------
    trend: xr.Dataset
        The long term average trend
    """
    # the rolling windows need to be odd numbers since the rolling average is centered
    assert day_rolling_window % 2 == 1
    d_offset = int((day_rolling_window - 1) / 2)

    assert year_rolling_window % 2 == 1
    y_offset = int((year_rolling_window - 1) / 2)

    # get historical mean as a rolling average -- the result has one number for each day of year
    # which is the average of the neighboring day of years over multiple years
    padded = add_circular_temporal_pad(data=data.sel(time=historical_period), offset=d_offset)
    hist_mean = (
        padded.rolling(time=day_rolling_window, center=True)
        .mean()
        .dropna('time')
        .groupby("time.dayofyear")
        .mean()
    )

    # get rolling average for the entire data -- the result has one number for each day in the entire time series
    # which is the average of the neighboring day of years in neighboring years
    padded = add_circular_temporal_pad(data=data, offset=d_offset)
    func = lambda x: x.rolling(time=year_rolling_window, center=True).mean()
    rolling_doy_mean = (
        padded.rolling(time=day_rolling_window, center=True)
        .mean()
        .dropna('time')
        .groupby('time.dayofyear')
        .apply(func)
        .dropna('time')
    ).compute()  # TODO: this .compute is needed otherwise the pad_with_edge_year function below returns all nulls for unknown reasons, root cause?

    # repeat the first/last year to cover the time periods without enough data for the rolling average
    for i in range(y_offset):
        rolling_doy_mean = pad_with_edge_year(rolling_doy_mean)

    # remove historical mean from rolling average, leaving trend as the anamoly
    trend = rolling_doy_mean.groupby('time.dayofyear') - hist_mean

    assert trend.isnull().sum().values == 0

    return trend


def remove_epoch_trend(data, trend, **kwargs):
    """
    Subtract the trend from data
    """
    ea_data = data - trend

    return ea_data
