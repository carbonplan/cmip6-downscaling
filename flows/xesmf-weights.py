from __future__ import annotations

import dask
from prefect import Flow, task, unmapped
from prefect.tasks.control_flow import merge
from upath import UPath

from cmip6_downscaling import config
from cmip6_downscaling.runtimes import get_runtime

dask.config.set(
    {
        'distributed.dashboard.link': 'https://prod.azure.carbonplan.2i2c.cloud/{JUPYTERHUB_SERVICE_PREFIX}/proxy/{port}/status'
    }
)

config.set(
    {
        'runtime.cloud.extra_pip_packages': 'git+https://github.com/carbonplan/cmip6-downscaling.git@main git+https://github.com/intake/intake-esm.git git+https://github.com/carbonplan/ndpyramid tabulate'
    }
)

folder = 'xesmf-weights'

scratch_dir = UPath(config.get('storage.scratch.uri')) / folder

runtime = get_runtime()


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

    print(len(stores))
    return stores


def generate_weights_pyramid(ds_in, levels):
    import datatree
    import xarray as xr
    import xesmf as xe
    from ndpyramid.regrid import make_grid_ds

    weights_pyramid = datatree.DataTree()
    with dask.config.set(scheduler='sync'):
        for level in range(levels):
            ds_out = make_grid_ds(level=level)
            regridder = xe.Regridder(ds_in, ds_out, method='bilinear')
            w = regridder.weights.data
            dim = 'n_s'
            ds = xr.Dataset(
                {
                    'S': (dim, w.data),
                    'col': (dim, w.coords[1, :] + 1),
                    'row': (dim, w.coords[0, :] + 1),
                }
            )
            weights_pyramid[str(level)] = ds

        weights_pyramid.ds.attrs['levels'] = levels
        weights_pyramid.ds.attrs['regrid_method'] = regridder.method

    return weights_pyramid


@task(log_stdout=True)
def generate_weights(store: dict, levels: int) -> dict:

    import xarray as xr

    method = 'bilinear'

    target = (
        scratch_dir
        / store['source_id']
        / store['table_id']
        / store['grid_label']
        / f'{method}_{levels}.zarr'
    )

    print(f'weights pyramid path: {target}')

    try:
        ds_in = xr.open_dataset(store['zstore'], engine='zarr', chunks={})
        weights_pyramid = generate_weights_pyramid(ds_in, levels)

        weights_pyramid.to_zarr(target, mode='w')

    except Exception as e:
        print(f'Failed to load {store["zstore"]}')
        print(e)
    attrs = {
        'source_id': store['source_id'],
        'table_id': store['table_id'],
        'grid_label': store['grid_label'],
        'regrid_method': method,
        'levels': levels,
        'path': str(target),
    }
    return attrs


@task(log_stdout=True)
def catalog(vals):
    import pandas as pd

    target = scratch_dir / 'weights.csv'
    df = pd.DataFrame(vals)
    df.to_csv(target, index=False)
    print(target)


with Flow(name='xesmf-weights', storage=runtime.storage, run_config=runtime.run_config) as flow:
    stores = get_stores()
    attrs = generate_weights.map(stores, levels=unmapped(2))
    vals = merge(attrs)
    _ = catalog(vals)
