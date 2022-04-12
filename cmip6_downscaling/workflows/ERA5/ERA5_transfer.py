# Imports -----------------------------------------------------------
import json
import os
from typing import Dict, List

import fsspec  # type: ignore
import pandas as pd  # type: ignore
import xarray as xr
from prefect import Flow, task
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
csv_catalog_path = "az://training/ERA5_catalog.csv"
json_catalog_path = "az://training/ERA5_catalog.json"

# Helper Functions -----------------------------------------------------------


def zarr_is_complete(store: str, check: str = ".zmetadata") -> bool:
    """Return true if Zarr store is complete

    Parameters
    ----------
    store : str
        store name
    check : str, optional
        check name, by default ".zmetadata"

    Returns
    -------
    bool
        True if zarr store is complete
    """
    return check in store


def map_tgt(tgt: str) -> fsspec.FSMap:
    """Uses fsspec to creating mapped object from target connection string

    Parameters
    ----------
    tgt : str
        path store

    Returns
    -------
    fsspec.FSMap
        fsspec mapped object
    """
    tgt_map = fsspec.get_mapper(tgt, connection_string)
    return tgt_map


def copy_cleaned_data(xdf: xr.Dataset, tgt_map: str, overwrite: bool = True) -> None:
    """Copies xarray dataset to zarr store for given target

    Parameters
    ----------
    xdf : xr.Dataset
        Input xarray dataset
    tgt_map : str
        zarr store mapped target
    overwrite : bool, optional
        overwrite flag, by default True
    """
    if overwrite is True:
        mode = "w"
    else:
        mode = "r"
    xdf.to_zarr(tgt_map, mode=mode, consolidated=True)


def extract_name_path(file_path: str) -> str:
    """String formatting to update the prefix of the ERA5 store location to Azure

    Parameters
    ----------
    file_path : str
        input file path

    Returns
    -------
    str
        azure specific formatted file path
    """
    tgt = "az://cmip6/ERA5/" + file_path.split("/zarr/")[1].replace("/data", "")
    return tgt


def map_and_open_zarr_link(file_loc_str: str) -> xr.Dataset:
    """Takes zarr store, opens with fsspec and returns xarray dataset

    Parameters
    ----------
    file_loc_str : str
        zarr store target path

    Returns
    -------
    xr.Dataset
        output xarray dataset
    """
    mapped_key = fsspec.get_mapper(file_loc_str, anon=True)
    ds = xr.open_zarr(mapped_key, consolidated=True)
    return ds


def create_formatted_links() -> List:
    """Create list of tuples representing all year/month/variable combinations

    Returns
    -------
    List
        list of potential file patterns
    """
    vars_list = [
        "eastward_wind_at_10_metres",
        "northward_wind_at_10_metres",
        "eastward_wind_at_100_metres",
        "northward_wind_at_100_metres",
        "dew_point_temperature_at_2_metres",
        "air_temperature_at_2_metres",
        "air_temperature_at_2_metres_1hour_Maximum",
        "air_temperature_at_2_metres_1hour_Minimum",
        "air_pressure_at_mean_sea_level",
        "sea_surface_temperature",
        "surface_air_pressure",
        "integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation",
        "precipitation_amount_1hour_Accumulation",
    ]

    month_list = [str(x).zfill(2) for x in range(1, 13, 1)]
    year_list = [str(x) for x in range(1979, 2021, 1)]
    file_pattern_list = []
    for year in year_list:
        for month in month_list:
            for var in vars_list:
                file_pattern = f"s3://era5-pds/zarr/{year}/{month}/data/{var}.zarr/"
                file_pattern_list.append(file_pattern)
    return file_pattern_list


def open_json_catalog() -> Dict:
    """Loads local CMIP6 JSON intake catalog

    Returns
    -------
    Dict
        dictionary as json
    """
    data = json.load(open("ERA5_catalog.json"))
    return data


def write_json_catalog_to_azure():
    """Writes json catalog to Azure store"""
    data = open_json_catalog()
    with fsspec.open(json_catalog_path, "w", connection_string=connection_string) as f:
        json.dump(data, f)


def create_csv_catalog():
    """Creates csv catalog of zarr stores"""
    file_pattern_list = create_formatted_links()
    az_list = []
    for fil in file_pattern_list:
        az_fil = extract_name_path(fil)
        az_list.append(az_fil)

    df = pd.DataFrame({"zstore": az_list})
    zstore_split_str = df["zstore"].str.split(".zarr", expand=True)[0].str.split("/")
    df["variable"] = zstore_split_str.str[-1]
    df["month"] = zstore_split_str.str[-2]
    df["year"] = zstore_split_str.str[-3]
    df.to_csv(
        csv_catalog_path,
        storage_options={"connection_string": connection_string},
        index=False,
    )


def clean_ds(ds: xr.Dataset) -> xr.Dataset:
    """Minor function to rename dataset coords

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray dataset

    Returns
    -------
    xr.Dataset
        xarray dataset with renamed coordinates
    """
    if "time0" in list(ds.coords):
        cleaned_ds = ds.rename({"time0": "time"})
    elif "time1" in list(ds.coords):
        cleaned_ds = ds.rename({"time1": "time"})

    return cleaned_ds


@task()
def copy_to_azure(file_path: str):
    tgt = extract_name_path(file_path)
    tgt_map = map_tgt(tgt)
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
