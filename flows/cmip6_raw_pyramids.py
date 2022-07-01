from __future__ import annotations

import traceback
from functools import partial

import dask
import xarray as xr
from prefect import Flow, Parameter, task
from prefect.backend.flow_run import FlowRunView
from prefect.client import Client
from upath import UPath

from cmip6_downscaling import __version__ as version, config, runtimes
from cmip6_downscaling.data.cmip import postprocess
from cmip6_downscaling.methods.common.tasks import _pyramid_postprocess
from cmip6_downscaling.utils import write

config.set(
    {
        'runtime.cloud.extra_pip_packages': 'git+https://github.com/carbonplan/cmip6-downscaling.git@main git+https://github.com/intake/intake-esm.git'
    }
)

config.set({'storage.results.uri': 'az://flow-outputs/results'})
scratch_dir = UPath(config.get('storage.scratch.uri')) / 'cmip6-pyramids-inputs/'
results_dir = UPath(config.get("storage.results.uri")) / 'cmip6-pyramids-raw'
use_cache = config.get('run_options.use_cache')


runtime = runtimes.CloudRuntime()


@task(log_stdout=True)
def get_assets(
    *,
    cat_url: str,
    source_id: list[str] | str,
    variable_id: list[str] | str,
    experiment_id: list[str] | str,
    table_id: list[str] | str = 'day',
    grid_label: list[str] | str = 'gn',
    member_id: list[str] | str = 'r1i1p1f1',
) -> list[tuple(str, str)]:

    import intake

    cat = intake.open_esm_datastore(cat_url).search(
        member_id=member_id,
        table_id=table_id,
        grid_label=grid_label,
        variable_id=variable_id,
        source_id=source_id,
        experiment_id=experiment_id,
        # require_all_on=['variable_id'],
    )
    # Add member_id to groupby_attrs
    if 'member_id' not in cat.esmcat.aggregation_control.groupby_attrs:
        cat.esmcat.aggregation_control.groupby_attrs += ['member_id', 'variable_id']

    cat.esmcat.aggregation_control.aggregations = []
    return [(cat[key].df.iloc[0].zstore, key) for key in cat.keys()]


def get_pyramid_weights(
    *,
    source_id: str,
    table_id: str,
    grid_label: str,
    levels: int = 2,
    regrid_method: str = "bilinear",
) -> str:
    import pandas as pd

    weights = pd.read_csv(config.get('weights.gcm_pyramid_weights.uri'))
    return (
        weights[
            (weights.regrid_method == regrid_method)
            & (weights.levels == levels)
            & (weights.source_id == source_id)
            & (weights.table_id == table_id)
        ]
        .iloc[0]
        .path
    )


def preprocess(ds) -> xr.Dataset:
    time_slice = slice('1950', '2100')
    ds = ds.sel(time=time_slice).pipe(partial(postprocess, to_standard_calendar=False))
    return ds


def _compute_summary_helper(path, freq, chunks):
    import xarray as xr

    with xr.set_options(keep_attrs=True), dask.config.set(
        **{'array.slicing.split_large_chunks': False}
    ):
        ds = xr.open_zarr(path).pipe(preprocess)
        if ds.attrs['variable_id'] in {'tasmax', 'tasmin'}:
            out = ds.resample(time=freq).mean(dim='time')
        elif ds.attrs['variable_id'] in {'pr'}:
            out = (ds * 86400).resample(time=freq).sum(dim='time')
            out['pr'].attrs['units'] = 'mm'
        else:
            raise NotImplementedError('variable not implemented')
        return out.astype('float32').chunk(chunks)


@task(log_stdout=True)
def compute_summary(assets: list[tuple(str, str)]) -> dict[str, list[str]]:
    successes = []
    failures = []
    for asset, key in assets:
        for freq, chunks in [('1MS', {'time': 12}), ('YS', {'time': 10})]:
            try:
                ds = _compute_summary_helper(asset, freq, chunks)
                target = scratch_dir / f'{freq}-summary' / f'{key}.{freq}'
                write(ds, target)
                successes.append(target)
            except Exception:
                failures.append(asset)
                print(f'***{asset}***:\n{traceback.format_exc()}')
    return {'successes': successes, 'failures': failures}


@task(log_stdout=True)
def compute_pyramids(results: dict[str, list[str]], levels: int) -> dict[str, list[str]]:
    import datatree
    import xarray as xr
    from carbonplan_data.metadata import get_cf_global_attrs
    from ndpyramid import pyramid_regrid
    from upath import UPath

    from cmip6_downscaling.methods.common.tasks import _load_coords

    stores = results['successes']
    successes = []
    failures = []
    for store in stores:
        try:
            store = UPath(store)
            name = store.name
            parts = name.split('.')
            source_id = parts[2]
            table_id = parts[4]
            grid_label = parts[5]
            target = results_dir / name
            with xr.set_options(keep_attrs=True):
                ds = xr.open_zarr(store).pipe(_load_coords)
                ds.coords['date_str'] = ds['time'].dt.strftime('%Y-%m-%d').astype('S10')
                ds.attrs.update(
                    {'title': ds.attrs['title']}, **get_cf_global_attrs(version=version)
                )

                target_pyramid = datatree.open_datatree('az://static/target-pyramid', engine='zarr')
                weights_pyramid_path = get_pyramid_weights(
                    source_id=source_id, table_id=table_id, grid_label=grid_label
                )
                weights_pyramid = datatree.open_datatree(weights_pyramid_path, engine='zarr')

                # create pyramid
                dta = pyramid_regrid(
                    ds,
                    target_pyramid=target_pyramid,
                    levels=levels,
                    weights_pyramid=weights_pyramid,
                    regridder_kws={'ignore_degenerate': True, 'extrap_method': "nearest_s2d"},
                )

                dta = _pyramid_postprocess(dta, levels=levels)
                write(dta, target, use_cache=False)
                successes.append(target)
        except Exception:
            failures.append(store)
            print(f'***{store}***:\n{traceback.format_exc()}')

    return {'successes': successes, 'failures': failures}


@task(log_stdout=True)
def report(results: dict[str, list[str]]) -> None:
    print(f"*** Failures: {len(results['failures'])} ***")
    print(results['failures'])
    print('\n\n')
    print(f"*** Successes: {len(results['successes'])} ***")
    print(results['successes'])


def run_flow(flow_id: str) -> None:
    client = Client()
    flow_run_id = client.create_flow_run(flow_id=flow_id)
    flow_run = FlowRunView.from_flow_run_id(flow_run_id)
    run_url = client.get_cloud_url("flow-run", flow_run_id)
    print(run_url)
    return flow_run_id, flow_run, run_url


with Flow(
    'cmip6-raw-pyramids',
    storage=runtime.storage,
    run_config=runtime.run_config,
    executor=runtime.executor,
) as flow:
    levels = Parameter('levels', default=2)
    cat_url = Parameter(
        "catalog", default="https://cmip6downscaling.blob.core.windows.net/cmip6/pangeo-cmip6.json"
    )

    source_ids = ["MRI-ESM2-0", "NorESM2-LM", "CanESM5", "MIROC6", "BCC-CSM2-MR"]
    source_id = Parameter("source_id", default=source_ids)
    variable_id = Parameter("variable_id", default=["tasmax", "tasmin", "pr"])
    experiment_id = Parameter("experiment_id", default=["historical", "ssp245", "ssp370", "ssp585"])
    assets = get_assets(
        cat_url=cat_url, source_id=source_id, variable_id=variable_id, experiment_id=experiment_id
    )
    results = compute_summary(assets)
    report(results)
    pyramids_results = compute_pyramids(results, levels)
    report(pyramids_results)
