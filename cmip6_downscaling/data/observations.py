import os

os.environ["PREFECT__FLOWS__CHECKPOINTING"] = "True"
import intake
import numpy as np
import xarray as xr
import zarr
from xarray_schema import DataArraySchema

from cmip6_downscaling.workflows.utils import load_paths, regrid_dataset

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

schema_map_chunks = DataArraySchema(chunks={'lat': -1, 'lon': -1})
schema_timeseries_chunks = DataArraySchema(chunks={'time': -1})
variable_name_dict = {
    "tasmax": "air_temperature_at_2_metres_1hour_Maximum",
    "tasmin": "air_temperature_at_2_metres_1hour_Minimum",
    "pr": "precipitation_amount_1hour_Accumulation",
}


def get_store(bucket, prefix, account_key=None):
    """helper function to create a zarr store"""

    if account_key is None:
        account_key = os.environ.get("AccountKey", None)

    store = zarr.storage.ABSStore(
        bucket, prefix=prefix, account_name="cmip6downscaling", account_key=account_key
    )
    return store


def open_era5(var: str, start_year: str, end_year: str) -> xr.Dataset:
    """Open ERA5 daily data for a single variable for period 1979-2021

    Parameters
    ----------
    var : str
        The variable you want to grab from the ERA5 dataset.

    start_year : str
        The first year of the time period you want to grab from ERA5 dataset.

    end_year : str
        The last year of the time period you want to grab from ERA5 dataset.

    Returns
    -------
    xarray.Dataset
        A daily dataset for one variable.
    """
    stores = [
        f'https://cmip6downscaling.blob.core.windows.net/cmip6/ERA5_daily/{y}'
        for y in range(int(start_year), int(end_year) + 1)
    ]
    ds = xr.open_mfdataset(stores, engine='zarr', consolidated=True)
    return ds[[var]]
