"""Script for resampling and reformatting existing ERA5 zarr stores into a single, daily zarr store with matching variable names to CMIP6 GCM data."""

import os

import fsspec
import intake
import xarray as xr
import zarr
from prefect import Flow, task
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure

# from cmip6_downscaling.workflows.share import get_store

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
account_key = os.environ.get("account_key")


def get_store(bucket, prefix, account_key=None):
    """helper function to create a zarr store"""
    if account_key is None:
        account_key = os.environ.get("AccountKey", None)

    store = zarr.storage.ABSStore(
        bucket, prefix=prefix, account_name="cmip6downscaling", account_key=account_key
    )
    return store


def get_ERA5_zstore_list():
    account_key = os.environ.get("AccountKey", None)
    col = intake.open_esm_datastore(
        "https://cmip6downscaling.blob.core.windows.net/cmip6/ERA5_catalog.json"
    )
    zstores = [zstore.split("az://cmip6/")[1] for zstore in col.df.zstore.to_list()]
    store_list = [
        get_store(bucket="cmip6", prefix=prefix, account_key=account_key) for prefix in zstores
    ]
    return store_list


def get_var_name_dict():
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
    return var_name_dict


def map_tgt(tgt, connection_string):
    """Uses fsspec to creating mapped object from target connection string"""
    tgt_map = fsspec.get_mapper(tgt, connection_string=connection_string)
    return tgt_map


def get_storage_chunks(chunk_direction):

    if chunk_direction.lower() == "time":
        # mdf.isel(time=slice(0,2)).nbytes/1e6 = ~102 MB. So every two days of full slices is 100MB
        target_chunks = {"lat": 721, "lon": 1440, "time": 2}
    elif chunk_direction.lower() == "space":
        # time_mult = 368184 (time_mult is # of time entries) -- (ds.isel(time=0,lat=slice(0,2),lon=slice(0,2)).nbytes * time_mult)/1e6 = ~88MB
        target_chunks = {"lat": 2, "lon": 2, "time": -1}
    return target_chunks


def write_zarr_store(ds, storage_chunks):
    mapped_tgt = map_tgt("az://cmip6/ERA5/ERA5_resample_chunked_time/", connection_string)
    ds.to_zarr(mapped_tgt, mode="w", consolidated=True, chunk_store=storage_chunks)


@task()
def get_zarr_store_list():
    # grab zstore list
    store_list = get_ERA5_zstore_list()
    return store_list


@task()
def downsample_and_combine(chunking_method, store_list):
    # grab variable rename dictionary
    var_name_dict = get_var_name_dict()

    # Open all zarr stores for ERA5
    ds_orig = xr.open_mfdataset(store_list, engine="zarr", concat_dim="time")
    print(ds_orig)
    ds = xr.Dataset()

    # drop time1 bounds
    ds_orig = ds_orig.drop("time1_bounds")

    # change lat convention to go from +90 to -90  to -90 to +90. This will match GCM conventions
    ds_orig = ds_orig.reindex({"lat": ds_orig.lat[::-1]})

    # rename vars to match CMIP6 conventions
    ds_orig = ds_orig.rename(var_name_dict)

    # mean vars
    ds["uas"] = ds_orig["uas"].resample(time="D").mean(keep_attrs=True)
    ds["vas"] = ds_orig["vas"].resample(time="D").mean(keep_attrs=True)
    ds["ua100m"] = ds_orig["ua100m"].resample(time="D").mean(keep_attrs=True)
    ds["va100m"] = ds_orig["va100m"].resample(time="D").mean(keep_attrs=True)
    ds["tdps"] = ds_orig["tdps"].resample(time="D").mean(keep_attrs=True)
    ds["tas"] = ds_orig["tas"].resample(time="D").mean(keep_attrs=True)
    ds["psl"] = ds_orig["psl"].resample(time="D").mean(keep_attrs=True)
    ds["tos"] = ds_orig["tos"].resample(time="D").mean(keep_attrs=True)
    ds["ps"] = ds_orig["ps"].resample(time="D").mean(keep_attrs=True)

    # summed vars
    ds["rsds"] = ds_orig["rsds"].resample(time="D").sum(keep_attrs=True) / 86400.0
    ds["pr"] = ds_orig["pr"].resample(time="D").sum(keep_attrs=True) / 86400.0 * 1000.0

    # min/max vars
    ds["tasmax"] = ds_orig["tasmax"].resample(time="D").max(keep_attrs=True)
    ds["tasmin"] = ds_orig["tasmin"].resample(time="D").min(keep_attrs=True)

    # write CRS
    ds = ds.rio.write_crs("EPSG:4326")
    print(ds)
    # write data as consolidated zarr store
    storage_chunks = get_storage_chunks(chunking_method)
    ds = ds.chunk(storage_chunks)
    write_zarr_store(ds)


run_config = KubernetesRun(
    cpu_request=2,
    memory_request="6Gi",
    image="gcr.io/carbonplan/hub-notebook:b2ea31c",
    labels=["az-eu-west"],
    env={"EXTRA_PIP_PACKAGES": "git+git://github.com/carbonplan/cmip6-downscaling"},
)
storage = Azure("prefect")

chunking_method = "space"
with Flow(
    name=f"Resample_ERA5_chunked_{chunking_method}",
    storage=storage,
    run_config=run_config,
) as flow:
    store_list = get_zarr_store_list()
    downsample_and_combine(chunking_method, store_list)