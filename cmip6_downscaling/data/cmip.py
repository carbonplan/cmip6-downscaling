from typing import List, Union

import dask
import numpy as np
import xarray as xr
from xarray.core.types import T_Xarray

from cmip6_downscaling import config
from cmip6_downscaling.methods.common.containers import BBox
from cmip6_downscaling.methods.common.utils import subset_dataset

from . import cat
from .utils import lon_to_180


def postprocess(ds: xr.Dataset) -> xr.Dataset:
    """Post process input experiment

    - Drops band variables (if present)
    - Drops height variable (if present)
    - Squeezes length 1 dimensions (if present)
    - Standardizes longitude convention to [-180, 180]
    - Reorders latitudes to [-90, 90]

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset

    Returns
    -------
    ds : xr.Dataset
        Post processed dataset
    """

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
    if ds.lat[0] < ds.lat[-1]:
        ds = ds.reindex({"lat": ds.lat[::-1]})

    return ds


def load_cmip(
    activity_ids: str = "CMIP",
    experiment_ids: str = "historical",
    member_ids: str = "r1i1p1f1",
    source_ids: str = "MIROC6",
    table_ids: str = "day",
    grid_labels: str = "gn",
    variable_ids: List[str] = ["tasmax"],
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

        if isinstance(variable_ids, str):
            variable_ids = [variable_ids]

        col = cat.cmip6()

        for i, var in enumerate(variable_ids):
            stores = (
                col.search(
                    activity_id=activity_ids,
                    experiment_id=experiment_ids,
                    member_id=member_ids,
                    source_id=source_ids,
                    table_id=table_ids,
                    grid_label=grid_labels,
                    variable_id=[var],
                )
                .df['zstore']
                .to_list()
            )

            # Q: why are we using the era5 storage_options
            storage_options = config.get('data_catalog.era5.storage_options')
            if len(stores) > 1:
                raise ValueError('can only get 1 store at a time')

            ds = xr.open_zarr(stores[0], storage_options=storage_options).pipe(postprocess)

            # convert to mm/day - helpful to prevent rounding errors from very tiny numbers
            if var == 'pr':
                ds['pr'] *= 86400

            if i == 0:
                ds_out = ds
            else:
                ds_out[var] = ds[var]

        return ds_out


def get_gcm(
    gcm: str,
    scenario: str,
    variables: Union[str, List[str]],
    train_period: slice,
    predict_period: slice,
    bbox: BBox,
) -> xr.Dataset:
    """
    Load and combine historical and future GCM into one dataset.

    Parameters
    ----------
    gcm : str
        Name of GCM
    scenario : str
        Name of scenario
    variables : str or list
        Name of variable(s) to load
    train_period_start : str
        Start year of train/historical period
    train_period_end : str
        End year of train/historical period
    predict_period_start : str
        Start year of predict/future period
    predict_period_end : str
        End year of predict/future period
    bbox : BBox
        Bounding box for subset

    Returns
    -------
    ds_gcm : xr.Dataset
        A dataset containing both historical and future period of GCM data
    """
    historical_gcm = load_cmip(
        activity_ids='CMIP',
        experiment_ids='historical',
        source_ids=gcm,
        variable_ids=variables,
    )

    future_gcm = load_cmip(
        activity_ids='ScenarioMIP',
        experiment_ids=scenario,
        source_ids=gcm,
        variable_ids=variables,
    )

    ds_gcm = xr.combine_by_coords([historical_gcm, future_gcm], combine_attrs='drop_conflicts')

    ds_gcm_train = subset_dataset(
        ds=ds_gcm,
        variable=variables[0],
        time_period=train_period,
        bbox=bbox,
    )
    ds_gcm_predict = subset_dataset(
        ds=ds_gcm,
        variable=variables[0],
        time_period=predict_period,
        bbox=bbox,
    )

    ds_gcm = xr.combine_by_coords([ds_gcm_train, ds_gcm_predict], combine_attrs='drop_conflicts')
    ds_gcm = ds_gcm.reindex(time=sorted(ds_gcm.time.values))

    return ds_gcm


def get_gcm_grid_spec(gcm_name: str = None, gcm_ds: T_Xarray = None) -> str:
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):

        if gcm_ds is None:
            # Silences the /srv/conda/envs/notebook/lib/python3.9/site-packages/xarray/core/indexing.py:1228: PerformanceWarning: Slicing is producing a large chunk. To accept the large
            # chunk and silence this warning, set the option >>> with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            if gcm_name is not None:
                raise ValueError('one of gcm_ds or gcm_name has to be not empty')
            gcm_grid = load_cmip(
                source_ids=gcm_name,
            ).isel(time=0)
        else:
            gcm_grid = gcm_ds.isel(time=0)

    nlat = len(gcm_grid.lat)
    nlon = len(gcm_grid.lon)
    lat_spacing = int(np.round(abs(gcm_grid.lat[0] - gcm_grid.lat[1]), 1) * 10)
    lon_spacing = int(np.round(abs(gcm_grid.lon[0] - gcm_grid.lon[1]), 1) * 10)
    min_lat = int(np.round(gcm_grid.lat.min(), 1))
    min_lon = int(np.round(gcm_grid.lon.min(), 1))

    return f'{nlat:d}x{nlon:d}_gridsize_{lat_spacing:d}_{lon_spacing:d}_llcorner_{min_lat:d}_{min_lon:d}'
