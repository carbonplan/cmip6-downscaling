from __future__ import annotations

from prefect import Flow, task, unmapped

from cmip6_downscaling import config
from cmip6_downscaling.runtimes import get_runtime

config.set(
    {
        'runtime.cloud.extra_pip_packages': 'git+https://github.com/carbonplan/cmip6-downscaling.git@main git+https://github.com/intake/intake-esm.git git+https://github.com/carbonplan/ndpyramid tabulate'
    }
)

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
        .sample(n=5)
    ).to_dict(orient='records')

    print(len(stores))
    return stores


@task(log_stdout=True)
def generate_weights(store: dict, levels: int) -> bool:
    import datatree
    import xarray as xr
    import xesmf as xe
    from ndpyramid.regrid import make_grid_ds

    try:
        ds_in = xr.open_dataset(store['zstore'], engine='zarr', chunks={})
        weights_pyramid = datatree.DataTree()
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
            print(ds)
            weights_pyramid[str(level)] = ds
        weights_pyramid.ds.attrs['levels'] = levels
        weights_pyramid.ds.attrs['regrid_method'] = regridder.method
    except Exception:
        print(f'Failed to load {store["zstore"]}')
    return True


with Flow(
    name='xesmf-weights',
    storage=runtime.storage,
    run_config=runtime.run_config,
    executor=runtime.executor,
) as flow:
    stores = get_stores()
    _ = generate_weights.map(stores, levels=unmapped(2))
