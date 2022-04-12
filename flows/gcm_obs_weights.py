from __future__ import annotations

import itertools

import dask
import xesmf as xe
from prefect import Flow, task, unmapped
from prefect.tasks.control_flow import merge
from upath import UPath

from cmip6_downscaling import config
from cmip6_downscaling.data.observations import open_era5
from cmip6_downscaling.runtimes import PangeoRuntime

# config.set(
#     {
#         'runtime.cloud.extra_pip_packages': 'git+https://github.com/carbonplan/cmip6-downscaling.git@main git+https://github.com/intake/intake-esm.git git+https://github.com/carbonplan/ndpyramid@weights-pyramid tabulate'
#     }
# )


folder = 'xesmf_weights/gcm_obs'

static_dir = UPath(config.get('storage.static.uri')) / folder

runtime = PangeoRuntime()


@task(log_stdout=True)
def get_stores() -> list[dict]:
    import intake

    cat = intake.open_esm_datastore(
        'https://cmip6downscaling.blob.core.windows.net/cmip6/pangeo-cmip6.json'
    )
    stores = (
        cat.df.groupby(['source_id', 'table_id', 'grid_label'])
        .first()
        .reset_index()
        .drop(columns=['member_id', 'dcpp_init_year', 'version', 'activity_id', 'institution_id'])
    ).to_dict(orient='records')

    return stores


@task(log_stdout=True)
def generate_weights(store: dict, method: str = 'bilinear') -> dict:
    import xarray as xr
    from ndpyramid.regrid import xesmf_weights_to_xarray

    target_prefix = (
        static_dir / store['source_id'] / store['table_id'] / store['grid_label'] / method
    )

    target_forwards = target_prefix / f"{store['source_id']}_to_era5.zarr"
    target_reverse = target_prefix / f"era5_to_{store['source_id']}.zarr"

    try:
        with dask.config.set({'scheduler': 'sync'}):
            ds_in = xr.open_dataset(store['zstore'], engine='zarr', chunks={}).isel(time=0)
            ds_out = open_era5(store['variable_id'], time_period=slice('2000', '2001'))
            regridder = xe.Regridder(ds_in, ds_out, method=method, extrap_method="nearest_s2d")
            weights = xesmf_weights_to_xarray(regridder)
            weights.to_zarr(target_forwards, mode='w')

            regridder_reversed = xe.Regridder(
                ds_out, ds_in, method=method, extrap_method="nearest_s2d"
            )
            weights_reversed = xesmf_weights_to_xarray(regridder_reversed)
            weights_reversed.to_zarr(target_reverse, mode='w')
    except Exception as e:
        print(f'Failed to load {store["zstore"]}')
        print(e)
    attrs_forward = {
        'source_id': store['source_id'],
        'table_id': store['table_id'],
        'grid_label': store['grid_label'],
        'regrid_method': method,
        'path': str(target_forwards),
        'direction': 'gcm_to_obs',
    }
    attrs_reverse = {
        'source_id': store['source_id'],
        'table_id': store['table_id'],
        'grid_label': store['grid_label'],
        'regrid_method': method,
        'path': str(target_reverse),
        'direction': 'obs_to_gcm',
    }
    return [attrs_forward, attrs_reverse]


@task(log_stdout=True)
def catalog(vals):
    import pandas as pd

    target = static_dir / 'weights.csv'
    flat_vals = itertools.chain(*vals)
    print(flat_vals)
    df = pd.DataFrame(flat_vals)
    df.to_csv(target, mode='w', index=False)
    print(target)


with Flow(
    name='xesmf-weights',
    storage=runtime.storage,
    run_config=runtime.run_config,
    executor=runtime.executor,
) as flow:
    stores = get_stores()
    attrs = generate_weights.map(stores, method=unmapped('bilinear'))
    vals = merge(attrs)
    _ = catalog(vals)
