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

def load_downscaled_ds(VARIABLE: str = 'tasmax',
                        GCM: str = "MIROC6",
                        TRAIN_PERIOD_START: str = "1990",
                        TRAIN_PERIOD_END: str = "1990",
                        PREDICT_PERIOD_START:str ='2079',
                        PREDICT_PERIOD_END:str ='2079',
                        DOWNSCALE_METHOD:str ='bcsd',
                        SCENARIO:str ='ssp370') -> xr.Dataset:
    path = f"az://cmip6/results/{DOWNSCALE_METHOD}_{SCENARIO}_{GCM}_{PREDICT_PERIOD_START}_{PREDICT_PERIOD_END}_{VARIABLE}.zarr"
    return xr.open_zarr(path)
