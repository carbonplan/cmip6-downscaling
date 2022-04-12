from __future__ import annotations

import dask
from prefect import Flow, task
from upath import UPath

from cmip6_downscaling import config
from cmip6_downscaling.data.observations import open_era5
from cmip6_downscaling.runtimes import PangeoRuntime

folder = 'xesmf_weights/downscaled_pyramid'

scratch_dir = UPath(config.get('storage.static.uri')) / folder

runtime = PangeoRuntime()


@task(log_stdout=True)
def generate_weights(store: dict, levels: int, method: str = 'bilinear') -> dict:
    from ndpyramid.regrid import generate_weights_pyramid

    target = scratch_dir / f'{method}_{levels}.zarr'
    print(f'weights pyramid path: {target}')
    print(f'store: {store}')

    try:
        with dask.config.set({'scheduler': 'sync'}):
            ds_in = open_era5(store['variable_id'], time_period=slice('2000', '2001'))
            print(ds_in)
            weights_pyramid = generate_weights_pyramid(ds_in, levels, method=method)
            print(weights_pyramid)
            weights_pyramid.to_zarr(target, mode='w')

    except Exception as e:
        print(e)
    attrs = {
        'regrid_method': method,
        'levels': levels,
        'path': str(target),
    }
    return attrs


@task(log_stdout=True)
def catalog(vals):
    import pandas as pd

    target = scratch_dir / 'weights.csv'
    df = pd.DataFrame([vals])
    df.to_csv(target, mode='w', index=False)
    print(target)


with Flow(
    name='regrid-pyramid-weights',
    storage=runtime.storage,
    run_config=runtime.run_config,
    executor=runtime.executor,
) as flow:
    attrs = generate_weights({'variable_id': 'tasmax'}, levels=4, method='bilinear')
    _ = catalog(attrs)
