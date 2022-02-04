import xarray as xr

from cmip6_downscaling.constants import APHYSICAL_TEMP_HIGH, APHYSICAL_TEMP_LOW


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
    return ds


def make_qaqc_ds(
    ds: xr.Dataset, qaqc_checks: list = ['nulls', 'aphysical_high_temp', 'aphysical_low_temp']
) -> xr.Dataset:
    """Compile qaqc checks into one dataset

    Parameters
    ----------
    ds : xr.Dataset
        any dataset
    qaqc_checks : list, optional
        which bad data checks you want to do, by default ['nulls', 'aphysical_high_temp', 'aphysical_low_temp']

    Returns
    -------
    xr.Dataset
        dataset with all dimensions of ds but expanded along dimension of `qaqc_check`
    """
    qaqc_ds = xr.Dataset()
    ds_list = []

    for qaqc_check in qaqc_checks:
        ds_list.append(check_is_bad_data(ds, qaqc_check))
    qaqc_ds = xr.concat(ds_list, dim='qaqc_check')
    qaqc_ds = qaqc_ds.assign_coords({'qaqc_check': qaqc_checks})
    return qaqc_ds
