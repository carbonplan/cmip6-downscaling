"""Script for resampling and reformatting existing ERA5 zarr stores into a single, daily zarr store with matching variable names to CMIP6 GCM data."""

import os

import dask
import fsspec
import intake
import xarray as xr
from prefect import Flow, task
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
image = "carbonplan/cmip6-downscaling-prefect:latest"

SEC_PER_DAY = 86400
MM_PER_M = 1000

CHUNKS = {'time': -1, 'lon': 150, 'lat': 150}

var_name_dict = {
    "eastward_wind_at_10_metres": "uas",
    "northward_wind_at_10_metres": "vas",
    "eastward_wind_at_100_metres": "ua100m",
    "northward_wind_at_100_metres": "va100m",
    "dew_point_temperature_at_2_metres": "tdps",
    "air_temperature_at_2_metres": "tas",
    "air_temperature_at_2_metres_1hour_Maximum": "tasmax",
    "air_temperature_at_2_metres_1hour_Minimum": "tasmin",
    "air_pressure_at_mean_sea_level": "psl",
    "sea_surface_temperature": "tos",
    "surface_air_pressure": "ps",
    "integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation": "rsds",
    "precipitation_amount_1hour_Accumulation": "pr",
}

storage = Azure("prefect")

run_config = KubernetesRun(cpu_request=7, memory_request="16Gi", image=image, labels=["az-eu-west"])


def get_ERA5_zstore_list(year: str = None) -> list:
    col = intake.open_esm_datastore(
        "https://cmip6downscaling.blob.core.windows.net/training/ERA5_catalog.json"
    )
    store_list = list(col.df.zstore)
    if year is not None:
        store_list = [s for s in store_list if year in s]
    return store_list


def _resample(da: xr.DataArray, mode='mean') -> xr.DataArray:
    resampler = da.resample(time='1D')
    method = getattr(resampler, mode)
    return method(keep_attrs=True)


@task(log_stdout=True)
def downsample_and_combine(year: str):
    # grab zstore list and variable rename dictionary
    store_list = get_ERA5_zstore_list(year=year)

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        # Open all zarr stores for ERA5
        ds_orig = xr.open_mfdataset(
            store_list,
            engine="zarr",
            concat_dim='time',
            coords='minimal',
            compat='override',
            consolidated=True,
        )
    # drop time1 bounds
    ds_orig = ds_orig.drop("time1_bounds")

    # rename vars to match CMIP6 conventions
    ds_orig = ds_orig.rename(var_name_dict)

    ds = xr.Dataset(attrs=ds_orig.attrs)

    # mean vars
    template = ds_orig["uas"].resample(time='1D').mean(keep_attrs=True).chunk(CHUNKS)
    for v in ["uas", "vas", "ua100m", "va100m", "tdps", "tas", "psl", "tos", "ps"]:
        ds[v] = (
            ds_orig[v]
            .chunk(CHUNKS)
            .map_blocks(_resample, kwargs={'mode': 'mean'}, template=template)
        )

    # summed vars
    ds['rsds'] = (
        ds_orig["rsds"]
        .chunk(CHUNKS)
        .map_blocks(_resample, kwargs={'mode': 'sum'}, template=template)
        / SEC_PER_DAY
    )
    ds['pr'] = (
        ds_orig['pr'].chunk(CHUNKS).map_blocks(_resample, kwargs={'mode': 'sum'}, template=template)
        / SEC_PER_DAY
        * MM_PER_M
    )

    # min/max vars
    ds["tasmax"] = (
        ds_orig["tasmax"]
        .chunk(CHUNKS)
        .map_blocks(_resample, kwargs={'mode': 'max'}, template=template)
    )
    ds["tasmin"] = (
        ds_orig["tasmin"]
        .chunk(CHUNKS)
        .map_blocks(_resample, kwargs={'mode': 'min'}, template=template)
    )

    # write data as consolidated zarr store
    mapper = fsspec.get_mapper(
        f'az://training/ERA5_daily/{year}', connection_string=connection_string
    )
    ds.to_zarr(mapper, mode='w', consolidated=True)


with Flow(
    name="Resample_ERA5",
    storage=storage,
    run_config=run_config,
) as flow:
    years = list(map(str, range(1979, 2021)))
    downsample_and_combine.map(years)
