import os

import fsspec
import intake
import xarray as xr
import zarr
from prefect import Flow, task
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure

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


def get_zstore_list():
    account_key = os.environ.get("AccountKey", None)
    col = intake.open_esm_datastore(
        "https://cmip6downscaling.blob.core.windows.net/cmip6/ERA5_catalog.json"
    )
    zstores = [zstore.split("az://cmip6/")[1] for zstore in col.df.zstore.to_list()]
    store_list = [
        get_store(bucket="cmip6", prefix=prefix, account_key=account_key)
        for prefix in zstores
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
        # time_mult = 368184 # num of time slices in dataset -- (ds.isel(time=0,lat=slice(0,2),lon=slice(0,2)).nbytes * time_mult)/1e6 = ~88MB
        target_chunks = {"lat": 2, "lon": 2}
    return target_chunks


def write_zarr_store(ds):
    mapped_tgt = map_tgt(
        "az://cmip6/ERA5/ERA5_resample_chunked_time/", connection_string
    )
    ds.to_zarr(mapped_tgt, mode="w", consolidated=True)


@task()
def downsample_and_combine():
    # grab zstore list and variable rename dictionary
    var_name_dict = get_var_name_dict()
    store_list = get_zstore_list()

    # Open all zarr stores for ERA5
    ds = xr.open_mfdataset(store_list, engine="zarr", concat_dim="time")

    # drop time1 bounds
    ds = ds.drop("time1_bounds")

    # rename vars to match CMIP6 conventions
    ds = ds.rename(var_name_dict)

    # mean vars
    ds["uas"] = ds["uas"].resample(time="D").mean(keep_attrs=True)
    ds["vas"] = ds["vas"].resample(time="D").mean(keep_attrs=True)
    ds["ua100m"] = ds["ua100m"].resample(time="D").mean(keep_attrs=True)
    ds["va100m"] = ds["va100m"].resample(time="D").mean(keep_attrs=True)
    ds["tdps"] = ds["tdps"].resample(time="D").mean(keep_attrs=True)
    ds["tas"] = ds["tas"].resample(time="D").mean(keep_attrs=True)
    ds["psl"] = ds["psl"].resample(time="D").mean(keep_attrs=True)
    ds["tos"] = ds["tos"].resample(time="D").mean(keep_attrs=True)
    ds["ps"] = ds["ps"].resample(time="D").mean(keep_attrs=True)

    # summed vars
    ds["rsds"] = ds["rsds"].resample(time="D").sum(keep_attrs=True) / 86400.0
    ds["pr"] = ds["pr"].resample(time="D").sum(keep_attrs=True)

    # min/max vars
    ds["tasmax"] = ds["tasmax"].resample(time="D").max(keep_attrs=True)
    ds["tasmin"] = ds["tasmin"].resample(time="D").min(keep_attrs=True)

    # write CRS
    ds = ds.rio.write_crs("EPSG:4326")

    # write data as consolidated zarr store
    storage_chunks = get_storage_chunks("space")
    write_zarr_store(ds, storage_chunks=storage_chunks)


run_config = KubernetesRun(
    cpu_request=3,
    memory_request="3Gi",
    image="gcr.io/carbonplan/hub-notebook:7252fc3",
    labels=["az-eu-west"],
)
storage = Azure("prefect")


with Flow(
    name="Resample_ERA5_chunked_space", storage=storage, run_config=run_config
) as flow:
    downsample_and_combine()
