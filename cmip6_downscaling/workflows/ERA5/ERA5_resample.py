import os

import dask
import fsspec
import intake
import xarray as xr
from dask_kubernetes import KubeCluster, make_pod_spec
from prefect import Flow, task
from prefect.executors import DaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure

dask.config.set({"logging.distributed": "debug"})
# dask.config.set({"logging.distributed.client": "debug"})


connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
image = "carbonplan/cmip6-downscaling-prefect:latest"


def get_ERA5_zstore_list():
    col = intake.open_esm_datastore(
        "https://cmip6downscaling.blob.core.windows.net/cmip6/ERA5_catalog.json"
    )
    store_list = list(col.df.zstore)
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
    # ALONG TIME
    if chunk_direction.lower() == "time":
        # mdf.isel(time=slice(0,2)).nbytes/1e6 = ~102 MB. So every two days of full slices is 100MB
        target_chunks = {"lat": 721, "lon": 1440, "time": 30}
        # keep spatial chunks as (150,150,*n time chunks)
    elif chunk_direction.lower() == "space":
        # time_mult = 368184 (time_mult is # of time entries) -- (ds.isel(time=0,lat=slice(0,2),lon=slice(0,2)).nbytes * time_mult)/1e6 = ~88MB
        target_chunks = {"lat": 2, "lon": 2, "time": -1}
    return target_chunks


def write_zarr_store(ds, chunking_method):
    mapped_tgt = map_tgt(
        f"az://cmip6/ERA5/ERA5_resample_chunked_{chunking_method}/", connection_string
    )
    ds.to_zarr(mapped_tgt, mode="w", consolidated=True)


def chunk_dataset(ds, storage_chunks):
    return ds.chunk(chunks=storage_chunks)


@task(log_stdout=True)
def downsample_and_combine(chunking_method):
    # print('s1', dask.base.get_scheduler())
    # grab zstore list and variable rename dictionary
    var_name_dict = get_var_name_dict()
    store_list = get_ERA5_zstore_list()
    print('91, store list retrieved')
    store_list = store_list[0:13]
    # Open all zarr stores for ERA5
    ds_orig = xr.open_mfdataset(
        store_list, engine="zarr", combine="by_coords", consolidated=True, parallel=True
    )
    print('95, open_mfdataset completed')
    # drop time1 bounds
    ds_orig = ds_orig.drop("time1_bounds")
    print('98, time bounds dropped')
    # change lat convention to go from +90 to -90  to -90 to +90. This will match GCM conventions
    ds_orig = ds_orig.reindex({"lat": ds_orig.lat[::-1]})
    print('101, reindex done')
    # rename vars to match CMIP6 conventions
    ds_orig = ds_orig.rename(var_name_dict)
    print('104, rename done')
    # mean vars
    mean_vars = ["uas", "vas", "ua100m", "va100m", "tdps", "tas", "psl", "tos", "ps"]
    ds = ds_orig[mean_vars].resample(time="D").mean(keep_attrs=True)
    print('108, vars averaged')
    # summed vars
    ds["rsds"] = ds_orig["rsds"].resample(time="D").sum(keep_attrs=True) / 86400.0
    ds["pr"] = ds_orig["pr"].resample(time="D").sum(keep_attrs=True) / 86400.0 * 1000.0
    print('112, vars summed')
    # min/max vars
    ds["tasmax"] = ds_orig["tasmax"].resample(time="D").max(keep_attrs=True)
    ds["tasmin"] = ds_orig["tasmin"].resample(time="D").min(keep_attrs=True)
    print('116, vars min/max')
    # write CRS
    # ds = ds.rio.write_crs("EPSG:4326")
    print('119, crs assigned')

    # write data as consolidated zarr store
    storage_chunks = get_storage_chunks(chunking_method)
    ds = ds.chunk(storage_chunks)
    print('124, dataset chunked')
    print('125. Write to zarr store starting...')
    # print('s2', dask.base.get_scheduler())
    ds.load()
    print(ds)
    # write_zarr_store(ds,chunking_method)
    print('127, write to zarr store complete')
    # print('s3', dask.base.get_scheduler())


chunking_method = "time"

storage = Azure("prefect")

run_config = KubernetesRun(cpu_request=2, memory_request="2Gi", image=image, labels=["az-eu-west"])


# client = Client(n_workers=4, threads_per_worker=1)

# # point Prefect's DaskExecutor to our Dask cluster
# executor = DaskExecutor(address=client.scheduler.address)


# with Flow(
#     name=f"Resample_ERA5_chunked_{chunking_method}",
# ) as flow:
#     downsample_and_combine(chunking_method)

# with Flow(
#     name=f"Resample_ERA5_chunked_{chunking_method}",
#     executor=executor
# ) as flow:
#     downsample_and_combine(chunking_method)

# flow.run(executor=executor)
executor = DaskExecutor(
    cluster_class=lambda: KubeCluster(
        make_pod_spec(
            image=image,
            memory_limit="2Gi",
            memory_request="2Gi",
            cpu_limit=2,
            cpu_request=2,
            env={
                "AZURE_STORAGE_CONNECTION_STRING": os.environ["AZURE_STORAGE_CONNECTION_STRING"],
            },
        )
    ),
    adapt_kwargs={"minimum": 2, "maximum": 20},
)

with Flow(
    name=f"Resample_ERA5_chunked_{chunking_method}",
    storage=storage,
    run_config=run_config,
    executor=executor,
) as flow:
    downsample_and_combine(chunking_method)
