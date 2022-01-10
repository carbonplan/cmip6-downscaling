import os

import xarray as xr
import zarr

from cmip6_downscaling.workflows.utils import get_store


def load_cmip(model, scenario, member, bias_corrected=False):

    if bias_corrected:
        prefix = f'cmip6/bias-corrected/conus/4000m/monthly/{model}.{scenario}.{member}.zarr'
    else:
        prefix = f'cmip6/regridded/conus/monthly/4000m/{model}.{scenario}.{member}.zarr'

    store = get_store(prefix)
    ds = xr.open_zarr(store, consolidated=True)
    return ds


def load_obs():
    mapper = zarr.storage.ABSStore(
        'carbonplan-downscaling',
        prefix='obs/conus/4000m/monthly/terraclimate_plus.zarr',
        account_name="carbonplan",
        account_key=os.environ["BLOB_ACCOUNT_KEY"],
    )
    ds = xr.open_zarr(mapper, consolidated=True)

    return ds

def load_downscaled_ds(variable: str = 'tasmax',
                        gcm: str = "MIROC6",
                        obs_id:str = 'ERA5',
                        predict_period_start:str ='2079',
                        predict_period_end:str ='2079',
                        downscale_method:str ='bcsd',
                        scenario:str ='ssp370') -> xr.Dataset:
    path = f"az://cmip6/results/{downscale_method}_{scenario}_{gcm}_{predict_period_start}_{predict_period_end}_{variable}.zarr"
    return xr.open_zarr(path)
