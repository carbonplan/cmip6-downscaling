from __future__ import annotations

import dask
from prefect import Flow, task, unmapped
from prefect.engine.signals import SKIP
from prefect.tasks.control_flow import merge
from prefect.tasks.control_flow.filter import FilterTask
from upath import UPath

from cmip6_downscaling import config
from cmip6_downscaling.runtimes import PangeoRuntime

folder = 'xesmf_weights/cmip6_pyramids'

scratch_dir = UPath(config.get('storage.static.uri')) / folder

runtime = PangeoRuntime()

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
            ds_in = xr.open_dataset(store['zstore'], engine='zarr', chunks={}).isel(time=0)
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
        raise SKIP(f"Failed to load {store['zstore']}") from e


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
    stores = get_stores()
    attrs = filter_results(
        generate_weights.map(stores, levels=unmapped(4), method=unmapped('bilinear'))
    )
    vals = merge(attrs)
    _ = catalog(vals)
