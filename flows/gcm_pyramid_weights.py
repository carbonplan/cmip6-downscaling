from __future__ import annotations

import traceback
from functools import partial

import dask
from prefect import Flow, Parameter, task, unmapped
from prefect.engine.signals import SKIP
from prefect.tasks.control_flow import merge
from prefect.tasks.control_flow.filter import FilterTask
from upath import UPath

from cmip6_downscaling import config
from cmip6_downscaling.data.cmip import postprocess
from cmip6_downscaling.runtimes import CloudRuntime

config.set(
    {
        'runtime.cloud.extra_pip_packages': 'git+https://github.com/carbonplan/cmip6-downscaling.git@main git+https://github.com/intake/intake-esm.git'
    }
)

folder = 'xesmf_weights/cmip6_pyramids'

scratch_dir = UPath(config.get('storage.static.uri')) / folder

runtime = CloudRuntime()

filter_results = FilterTask(
    filter_func=lambda x: not isinstance(x, (BaseException, SKIP, type(None)))
)


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
def generate_weights(store: dict, levels: int, method: str = 'bilinear') -> dict:
    import xarray as xr
    from ndpyramid.regrid import generate_weights_pyramid

    target = (
        scratch_dir
        / store['source_id']
        / store['table_id']
        / store['grid_label']
        / f'{method}_{levels}.zarr'
    )

    print(f'weights pyramid path: {target}')
    print(f'store: {store}')

    try:
        with dask.config.set({'scheduler': 'sync'}):
            ds_in = (
                xr.open_dataset(store['zstore'], engine='zarr', chunks={})
                .pipe(partial(postprocess, to_standard_calendar=False))
                .isel(time=0)
            )
            weights_pyramid = generate_weights_pyramid(ds_in, levels, method=method)
            print(weights_pyramid)
            weights_pyramid.to_zarr(target, mode='w')
        return {
            'source_id': store['source_id'],
            'table_id': store['table_id'],
            'grid_label': store['grid_label'],
            'regrid_method': method,
            'levels': levels,
            'path': str(target),
        }

    except Exception as e:
        raise SKIP(f"Failed to process {store['zstore']}\nError: {traceback.format_exc()}") from e


@task(log_stdout=True)
def catalog(vals):
    import pandas as pd

    target = scratch_dir / 'weights.csv'
    df = pd.DataFrame(vals)
    df.to_csv(target, mode='w', index=False)
    print(target)


with Flow(
    name='xesmf-weights',
    storage=runtime.storage,
    run_config=runtime.run_config,
    executor=runtime.executor,
) as flow:
    levels = Parameter('levels', default=2)
    method = Parameter('method', default='bilinear')
    stores = get_stores()
    attrs = filter_results(
        generate_weights.map(stores, levels=unmapped(levels), method=unmapped(method))
    )
    vals = merge(attrs)
    _ = catalog(vals)
