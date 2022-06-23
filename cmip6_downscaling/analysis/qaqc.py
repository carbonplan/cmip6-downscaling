from __future__ import annotations

import xarray as xr

from ..constants import (
    APHYSICAL_PRECIP_HIGH,
    APHYSICAL_PRECIP_LOW,
    APHYSICAL_TEMP_HIGH,
    APHYSICAL_TEMP_LOW,
)


def check_is_bad_data(ds: xr.Dataset, type: str) -> xr.Dataset:
    """Basic qaqc checks

    Parameters
    ----------
    ds : xr.Dataset
        Any dataset
    type : str
        kind of qaqc you're doing

    Returns
    -------
    xr.Dataset
        boolean mask of whether it is bad data
    """
    if type == 'nulls':
        ds = ds.isnull()
    elif type == 'aphysical_high_temp':
        ds = ds > APHYSICAL_TEMP_HIGH
    elif type == 'aphysical_low_temp':
        ds = ds < APHYSICAL_TEMP_LOW
    elif type == 'aphysical_low_precip':
        ds = ds < APHYSICAL_PRECIP_LOW
    elif type == 'aphysical_high_precip':
        ds = ds > APHYSICAL_PRECIP_HIGH
    else:
        raise TypeError('metric unavailable')
    return ds


def make_qaqc_ds(
    ds: xr.Dataset, checks: list = ['nulls', 'aphysical_high_temp', 'aphysical_low_temp']
) -> xr.Dataset:
    """Compile qaqc checks into one dataset

    Parameters
    ----------
    ds : xr.Dataset
        any dataset
    checks : list, optional
        which bad data checks you want to do, by default ['nulls', 'aphysical_high_temp', 'aphysical_low_temp']

    Returns
    -------
    xr.Dataset
        dataset with all dimensions of ds but expanded along dimension of `qaqc_check`
    """
    qaqc_ds = xr.Dataset()
    ds_list = []

    for check in checks:
        ds_list.append(check_is_bad_data(ds, check))
    qaqc_ds = xr.concat(ds_list, dim='qaqc_check')
    qaqc_ds = qaqc_ds.assign_coords({'qaqc_check': checks})
    return qaqc_ds
