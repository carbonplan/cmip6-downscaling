from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def add_circular_temporal_pad(data: xr.Dataset, offset: int, timeunit: str = 'D') -> xr.Dataset:
    """
    Pad the beginning of data with the last values of data, and pad the end of data with the first values of data.

    Parameters
    ----------
    data : xr.Dataset
        data to be padded, must have time as a dimension
    offset : int
        number of time points to be padded
    timeunit : str
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


def days_in_year(year: int, use_leap_year: bool = False) -> int:
    """
    Returns the number of days in the year input.

    Parameters
    ----------
    year : int
        year in question
    use_leap_year : bool
        whether to return 366 for leap years

    Returns
    -------
    n_days_in_year : int
        Number of days in the year of question
    """
    if (year % 4 == 0) and use_leap_year:
        return 366
    return 365


def pad_with_edge_year(data: xr.Dataset) -> xr.Dataset:
    """
    Pad data with the first and last year available. similar to the behavior of np.pad(mode='edge')
    but uses the 365 edge values instead of repeating only 1 edge value (366 if leap year)

    Parameters
    ----------
    data : xr.Dataset
        data to be padded, must have time as a dimension

    Returns
    -------
    padded : xr.Dataset
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


def generate_batches(
    n: int, batch_size: int, buffer_size: int, one_indexed: bool = False
) -> tuple[list, list]:
    """
    Given the max value n, batch_size, and buffer_size, returns batches (include the buffer) and
    cores (exclude the buffer). For the smallest numbers, the largest values would be included in the buffer, and
    vice versa. For example, with n=10, batch_size=5, buffer_size=3, one_indexed=False. The `cores` output will contain
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], and `batches` output will contain [[7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]].

    Parameters
    ----------
    n : int
        The max value to be included.
    batch_size : int
        The number of core values to include in each batch.
    buffer_size : int
        The number of buffer values to include in each batch in both directions.
    one_indexed : bool
        Whether we should consider n to be one indexed or not. With n = 2, one_indexed=False would generate cores containing [0, 1].
        ``one_indexed=True`` would generate cores containing [1, 2].

    Returns
    -------
    batches : List
        List of batches including buffer values.
    cores : List
        List of core values in each batch excluding buffer values.
    """
    cores = []
    batches = []
    if one_indexed:
        xmin = 1
        xmax = n + 1
    else:
        xmin = 0
        xmax = n
    for start in range(xmin, xmax, batch_size):
        end = min(start + batch_size, xmax)
        cores.append(np.arange(start, end))

        # add buffer
        end = end + buffer_size
        start = start - buffer_size
        batch = np.arange(start, end)
        batch[batch < xmin] += n
        batch[batch > xmax - 1] -= n
        batches.append(batch)

    return batches, cores
