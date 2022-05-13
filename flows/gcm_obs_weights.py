from __future__ import annotations

import traceback
from functools import partial

import dask
from prefect import Flow, Parameter, task
from upath import UPath

from cmip6_downscaling import config, runtimes
from cmip6_downscaling.data.cmip import postprocess
from cmip6_downscaling.data.observations import open_era5
from cmip6_downscaling.utils import write

config.set(
    {
        'runtime.cloud.extra_pip_packages': 'git+https://github.com/carbonplan/cmip6-downscaling.git@main git+https://github.com/intake/intake-esm.git'
    }
)


folder = 'xesmf_weights/gcm_obs'

static_dir = UPath(config.get('storage.static.uri')) / folder

use_cache = False  # config.get('run_options.use_cache')

runtime = runtimes.PangeoRuntime()


@task(log_stdout=True)
def get_stores(cat_url: str) -> list[dict]:
    import intake

    cat = intake.open_esm_datastore(cat_url)
    return (
        cat.df.groupby(['source_id', 'table_id', 'grid_label'])
        .first()
        .reset_index()
        .drop(columns=['member_id', 'dcpp_init_year', 'version', 'activity_id', 'institution_id'])
    ).to_dict(orient='records')


@task(log_stdout=True)
def generate_weights(stores: list[dict[str, str]], method: str = 'bilinear') -> dict:
    import xarray as xr
    import xesmf as xe
    from ndpyramid.regrid import xesmf_weights_to_xarray

    failures = []
    successes = []

    results = []

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        # use tasmax to retrieve ERA5 grid
        ds_out = open_era5('tasmax', time_period=slice('2000', '2000')).isel(time=0)
        print(ds_out)

        for store in stores:

            target_prefix = (
                static_dir / store['source_id'] / store['table_id'] / store['grid_label'] / method
            )

            target_forwards = target_prefix / f"{store['source_id']}_to_era5.zarr"
            target_reverse = target_prefix / f"era5_to_{store['source_id']}.zarr"

            try:
                with dask.config.set({'scheduler': 'sync'}):
                    ds_in = (
                        xr.open_zarr(store['zstore'])
                        .pipe(partial(postprocess, to_standard_calendar=False))
                        .isel(time=0)
                    )

                    regridder = xe.Regridder(
                        ds_in, ds_out, method=method, extrap_method="nearest_s2d"
                    )
                    weights = xesmf_weights_to_xarray(regridder)
                    write(weights, target_forwards, use_cache=use_cache)

                    regridder_reversed = xe.Regridder(
                        ds_out, ds_in, method=method, extrap_method="nearest_s2d"
                    )
                    weights_reversed = xesmf_weights_to_xarray(regridder_reversed)
                    write(weights_reversed, target_reverse, use_cache=use_cache)

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
                successes.append(store)
                results += [attrs_forward, attrs_reverse]
            except Exception:
                print(f'Failed to process {store["zstore"]}\nError: {traceback.format_exc()}')
                failures.append(store)

        return {'successes': successes, 'failures': failures, 'results': results}


@task(log_stdout=True)
def catalog(results):
    import pandas as pd

    target = static_dir / 'weights.csv'
    df = pd.DataFrame(results['results'])
    print(df.head())
    print(target)
    df.to_csv(target, mode='w', index=False)


with Flow(
    name='xesmf-weights',
    storage=runtime.storage,
    run_config=runtime.run_config,
    executor=runtime.executor,
) as flow:

    cat_url = Parameter(
        'cat_url', default='https://cmip6downscaling.blob.core.windows.net/cmip6/pangeo-cmip6.json'
    )
    method = Parameter('method', default='bilinear')
    stores = get_stores(cat_url)
    vals = generate_weights(stores, method=method)
    catalog(vals)
