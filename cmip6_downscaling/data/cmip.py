from __future__ import annotations

import dask
import pandas as pd
import xarray as xr

from . import cat
from .utils import lon_to_180, to_standard_calendar as convert_to_standard_calendar


def postprocess(ds: xr.Dataset, to_standard_calendar: bool = True) -> xr.Dataset:
    """Post process input experiment

    - Drops band variables (if present)
    - Drops height variable (if present)
    - Squeezes length 1 dimensions (if present)
    - Standardizes longitude convention to [-180, 180]
    - Reorders latitudes to [-90, 90]
    - Shifts time from Noon (12:00) start to Midnight (00:00)

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    to_standard_calendar : bool, optional
        Whether to convert time to standard calendar, by default True

    Returns
    -------
    ds : xr.Dataset
        Post processed dataset
    """

    with xr.set_options(keep_attrs=True):
        # drop band variables
        if ('lat_bnds' in ds) or ('lat_bnds' in ds.coords):
            ds = ds.drop('lat_bnds')
        if ('lon_bnds' in ds) or ('lon_bnds' in ds.coords):
            ds = ds.drop('lon_bnds')
        if ('time_bnds' in ds) or ('time_bnds' in ds.coords):
            ds = ds.drop('time_bnds')

        # drop height variable
        if 'height' in ds:
            ds = ds.drop('height')

        # squeeze length 1 dimensions
        ds = ds.squeeze(drop=True)

        # standardize longitude convention
        ds = lon_to_180(ds)

        # Reorders latitudes to [-90, 90]
        if ds.lat[0] > ds.lat[-1]:
            ds = ds.reindex({"lat": ds.lat[::-1]})

        if to_standard_calendar:

            # checks calendar
            ds = convert_to_standard_calendar(ds)

            # Shifts time from Noon (12:00) start to Midnight (00:00) start to match with Obs
            # ds.coords['time'] = ds['time'].resample(time='1D').first()

            ds['time'] = pd.date_range(
                start=ds['time'].data[0],
                end=ds['time'].data[-1],
                normalize=True,
                freq="1D",
            )
        return ds


def load_cmip(
    activity_ids: str = None,
    experiment_ids: str = None,
    member_ids: str = None,
    source_ids: str = None,
    table_ids: str = None,
    grid_labels: str = None,
    variable_ids: list[str] = None,
) -> xr.Dataset:
    """Loads CMIP6 GCM dataset based on input criteria.

    Parameters
    ----------
    activity_ids : list, optional
        activity_ids in CMIP6 catalog, by default ["CMIP", "ScenarioMIP"],
    experiment_ids : list, optional
        experiment_ids in CMIP6 catalog, by default ["historical", "ssp370"],  ex:#  "ssp126", "ssp245",  "ssp585"
    member_ids : list, optional
        member_ids in CMIP6 catalog, by default ["r1i1p1f1"]
    source_ids : list, optional
        source_ids in CMIP6 catalog, by default ["MIROC6"]
    table_ids : list, optional
        table_ids in CMIP6 catalog, by default ["day"]
    grid_labels : list, optional
        grid_labels in CMIP6 catalog, by default ["gn"]
    variable_ids : list, optional
        variable_ids in CMIP6 catalog, by default ['tasmax']

    Returns
    -------
    ds : xr.Dataset
        Dataset or zarr group with CMIP data
    """
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):

        col = cat.cmip6()
        col_subset = col.search(
            activity_id=activity_ids,
            experiment_id=experiment_ids,
            member_id=member_ids,
            source_id=source_ids,
            table_id=table_ids,
            grid_label=grid_labels,
            variable_id=variable_ids,
        )

        keys = list(col_subset.keys())
        if len(keys) != 1:
            raise ValueError(f'intake-esm search returned {len(keys)}, expected exactly 1.')

        ds = col_subset[keys[0]]().to_dask().pipe(postprocess)

        # convert to mm/day - helpful to prevent rounding errors from very tiny numbers
        if 'pr' in ds:
            # with xr.set_options(keep_attrs=True):
            ds['pr'] *= 86400
            ds['pr'] = ds['pr'].astype('float32')

        return ds


def get_gcm(
    scenario: str,
    member_id: str,
    table_id: str,
    grid_label: str,
    source_id: str,
    variable: str,
    time_slice: slice,
) -> xr.Dataset:
    """
    Load historical or future GCM into one dataset.

    Parameters
    ----------
    scenario : str
        Name of scenario
    member_id : str
        Name of member ID
    table_id : str
        Name of table ID
    grid_label : str
        Name of grid_label
    source_id : str
        Name of source_id
    variable : str
        Name of variable to load
    time_slice : slice
        Time period that you're hoping to load

    Returns
    -------
    ds_gcm : xr.Dataset
        A dataset containing either historical or future GCM data
    """

    kws = dict(
        member_ids=member_id,
        table_ids=table_id,
        grid_labels=grid_label,
        source_ids=source_id,
        variable_ids=variable,
    )
    if float(time_slice.stop) < 2015:
        # you're working with historical data
        ds_gcm = load_cmip(activity_ids='CMIP', experiment_ids='historical', **kws)
    elif float(time_slice.start) > 2014:
        # you're working with future data
        ds_gcm = load_cmip(activity_ids='ScenarioMIP', experiment_ids=scenario, **kws)
    else:
        raise ValueError(f'time slice {time_slice} not supported')
    ds_gcm = ds_gcm.reindex(time=sorted(ds_gcm.time.values))

    return ds_gcm
