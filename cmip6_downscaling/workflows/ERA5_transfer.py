# Imports -----------------------------------------------------------
import os

import fsspec
import xarray as xr
from prefect import Flow, task
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

# Helper Functions -----------------------------------------------------------


def zarr_is_complete(store, check=".zmetadata"):
    """Return true if Zarr store is complete"""
    return check in store


def map_tgt(tgt, connection_string):
    """Uses fsspec to creating mapped object from target connection string"""
    tgt_map = fsspec.get_mapper(tgt, connection_string=connection_string)
    return tgt_map


def copy_cleaned_data(xdf, tgt_map, overwrite=True):
    """Copies xarray dataset to zarr store for given target"""
    if overwrite is True:
        mode = "w"
    else:
        mode = "r"
    xdf.to_zarr(tgt_map, mode=mode, consolidated=True)


def extract_name_path(file_path):
    """String formatting to update the prefix of the ERA5 store location to Azure"""
    # tgt = "az://cmip6/ERA5/" + file_path_tuple[0].split('/zarr/')[1].split('data/')[0].replace('/','_') + 'consolidated_vars/'
    tgt = "az://cmip6/ERA5/" + file_path.split('/zarr/')[1].replace(
        '/data', ''
    )  # .replace('/','_')[:-1]

    return tgt


def map_and_open_zarr_link(file_loc_str):
    mapped_key = fsspec.get_mapper(file_loc_str, anon=True)
    ds = xr.open_zarr(mapped_key, consolidated=True)
    return ds


def create_formatted_links():
    """Create list of tuples representing all year/month/variable combinations"""
    vars_list = [
        'eastward_wind_at_10_metres',
        'northward_wind_at_10_metres',
        'eastward_wind_at_100_metres',
        'northward_wind_at_100_metres',
        'dew_point_temperature_at_2_metres',
        'air_temperature_at_2_metres',
        'air_temperature_at_2_metres_1hour_Maximum',
        'air_temperature_at_2_metres_1hour_Minimum',
        'air_pressure_at_mean_sea_level',
        'sea_surface_temperature',
        'surface_air_pressure',
        'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation',
        'precipitation_amount_1hour_Accumulation',
    ]

    month_list = [str(x).zfill(2) for x in range(1, 13, 1)]
    year_list = [str(x) for x in range(1979, 2021, 1)]
    file_pattern_list = []
    for year in year_list:
        for month in month_list:
            for var in vars_list:
                file_pattern = f's3://era5-pds/zarr/{year}/{month}/data/{var}.zarr/'
                file_pattern_list.append(file_pattern)
    return file_pattern_list


def clean_ds(ds):
    if 'time0' in list(ds.coords):
        cleaned_ds = ds.rename({'time0': 'time'})
    elif 'time1' in list(ds.coords):
        cleaned_ds = ds.rename({'time1': 'time'})

    return cleaned_ds


@task()
def copy_to_azure(file_path):
    tgt = extract_name_path(file_path)
    tgt_map = map_tgt(tgt, connection_string)
    if not zarr_is_complete(tgt_map):
        ds = map_and_open_zarr_link(file_path)
        cleaned_ds = clean_ds(ds)
        copy_cleaned_data(cleaned_ds, tgt_map, overwrite=True)


run_config = KubernetesRun(
    cpu_request=3,
    memory_request="3Gi",
    image="gcr.io/carbonplan/hub-notebook:7252fc3",
    labels=["az-eu-west"],
)
storage = Azure("prefect")


with Flow(name="Transfer_ERA5", storage=storage, run_config=run_config) as flow:
    file_pattern_list = create_formatted_links()
    copy_to_azure.map(file_pattern_list)
