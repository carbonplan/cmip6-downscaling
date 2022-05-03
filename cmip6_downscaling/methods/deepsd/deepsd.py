from __future__ import annotations

import fsspec
import numpy as np
import xarray as xr
import xesmf as xe

from cmip6_downscaling.workflows.paths import make_coarse_obs_path

intermediate_cache_path = 'az://flow-outputs/intermediate'

EPSILON = 1e-6  # small value to add to the denominator when normalizing to avoid division by 0
INPUT_SIZE = 51  # number of pixels in a patch example used for training deepsd model (in both lat/lon (or x/y) directions)
PATCH_STRIDE = 20  # number of pixels to skip when generating patches for deepsd training
starting_resolutions = {
    'GISS-E2-1-G': 2.0,
    'BCC-CSM2-MR': 1.0,
    'AWI-CM-1-1-MR': 1.0,
    'BCC-ESM1': 2.0,
    'SAM0-UNICON': 1.0,
    'CanESM5': 2.0,
    'MRI-ESM2-0': 1.0,
    'MPI-ESM-1-2-HAM': 2.0,
    'MPI-ESM1-2-HR': 1.0,
    'MPI-ESM1-2-LR': 2.0,
    'NESM3': 2.0,
    'NorESM2-LM': 2.0,
    'FGOALS-g3': 2.0,
    'MIROC6': 1.0,
    'ACCESS-CM2': 1.0,
    'NorESM2-MM': 1.0,
    'ACCESS-ESM1-5': 1.0,
    'AWI-ESM-1-1-LR': 2.0,
    'TaiESM1': 1.0,
    'NorCPM1': 2.0,
    'CMCC-ESM2': 1.0,
}
stacked_model_path = 'az://cmip6downscaling/training/deepsd/deepsd_models/{var}_{starting_resolution}d_to_0_25d/frozen_graph.pb'
output_node_name = '{var}_0_25/prediction:0'


def res_to_str(r):
    return str(np.round(r, 2)).replace('.', '_')


def bilinear_interpolate(output_degree, ds):
    target_grid_ds = xe.util.grid_global(output_degree, output_degree, cf=True)
    regridder = xe.Regridder(ds, target_grid_ds, "bilinear", extrap_method="nearest_s2d")
    ds_regridded = regridder(ds)

    return ds_regridded


def conservative_interpolate(output_degree, ds):
    target_grid_ds = xe.util.grid_global(output_degree, output_degree, cf=True)
    # conservative area regridding needs lat_bands and lon_bands
    regridder = xe.Regridder(ds, target_grid_ds, "conservative")
    ds_regridded = regridder(ds)
    return ds_regridded


def build_grid_spec(
    output_degree,
):
    output_degree = np.round(output_degree, 2)
    gcm_grid = xe.util.grid_global(output_degree, output_degree, cf=True)

    nlat = len(gcm_grid.lat)
    nlon = len(gcm_grid.lon)
    lat_spacing = int(np.round(abs(gcm_grid.lat[0] - gcm_grid.lat[1]), 1) * 10)
    lon_spacing = int(np.round(abs(gcm_grid.lon[0] - gcm_grid.lon[1]), 1) * 10)
    min_lat = int(np.round(gcm_grid.lat.min(), 1))
    min_lon = int(np.round(gcm_grid.lon.min(), 1))

    grid_spec = f'{nlat:d}x{nlon:d}_gridsize_{lat_spacing:d}_{lon_spacing:d}_llcorner_{min_lat:d}_{min_lon:d}'
    return grid_spec


def make_coarse_elev_path(
    output_degree,
):
    grid_spec = build_grid_spec(output_degree)
    return intermediate_cache_path + f'/elev/ERA5_full_space_{grid_spec}.zarr'


def get_elevation_data(output_degree):
    elev_path = make_coarse_elev_path(output_degree)
    elev_store = fsspec.get_mapper(elev_path)
    return xr.open_zarr(elev_store)


def normalize(ds, dims=['lat', 'lon'], epsilon=EPSILON):
    mean = ds.mean(dim=dims).compute()
    std = ds.std(dim=dims).compute()
    normed = (ds - mean) / (std + epsilon)

    return normed


def get_obs_mean(obs, train_period_start, train_period_end, variables, gcm_grid_spec, ds=None):
    # if mean is not already saved, ds must be a valid dataset
    path = make_coarse_obs_path(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variables=variables,
        gcm_grid_spec=gcm_grid_spec,
        chunking_approach='mean',
    )

    store = fsspec.get_mapper(intermediate_cache_path + '/' + path)

    if '.zmetadata' not in store:
        mean = ds.mean(dim='time')
        mean.to_zarr(store, mode="w", consolidated=True)
    else:
        mean = xr.open_zarr(store)
    return mean


def get_obs_std(obs, train_period_start, train_period_end, variables, gcm_grid_spec, ds=None):
    # if std is not already saved, ds must be a valid dataset
    path = make_coarse_obs_path(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variables=variables,
        gcm_grid_spec=gcm_grid_spec,
        chunking_approach='std',
    )
    store = fsspec.get_mapper(intermediate_cache_path + '/' + path)

    if '.zmetadata' not in store:
        std = ds.std(dim='time')
        std.to_zarr(store, mode="w", consolidated=True)
    else:
        std = xr.open_zarr(store).load()
    return std
