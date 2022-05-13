from prefect import Flow, Parameter, task

from cmip6_downscaling import config, runtimes

config.set(
    {
        'runtime.cloud.extra_pip_packages': 'git+https://github.com/carbonplan/cmip6-downscaling.git git+https://github.com/intake/intake-esm.git git+https://github.com/ncar-xdev/ecgtools.git'
    }
)

runtime = runtimes.PangeoRuntime()


def parse_cmip6(store):
    import traceback

    import intake
    from ecgtools.builder import INVALID_ASSET, TRACEBACK
    from upath import UPath

    from cmip6_downscaling.methods.common.utils import zmetadata_exists

    try:

        path = UPath(store)
        if not zmetadata_exists(path):
            raise ValueError(f'{path} not a valid zarr store')

        aggregation = None
        name = path.name
        (
            activity_id,
            institution_id,
            source_id,
            experiment_id,
            table_id,
            grid_label,
            member_id,
            variable_id,
            timescale,
        ) = name.split('.')
        query = {
            'activity_id': activity_id,
            'institution_id': institution_id,
            'source_id': source_id,
            'experiment_id': experiment_id,
            'table_id': table_id,
            'member_id': member_id,
            'variable_id': variable_id,
        }
        if timescale == '1MS':
            timescale = 'month'
        elif timescale == 'YS':
            timescale = 'year'

        if timescale in {'year', 'month'}:
            if variable_id in {'tasmin', 'tasmax'}:
                aggregation = 'mean'
            elif variable_id in {'pr'}:
                aggregation = 'sum'

        uri = f"https://cmip6downscaling.azureedge.net/{str(path).split('//')[-1]}"
        cat_url = "https://cmip6downscaling.blob.core.windows.net/cmip6/pangeo-cmip6.json"
        cat = intake.open_esm_datastore(cat_url)
        stores = cat.search(**query).df.zstore.tolist()
        if stores:
            original_dataset_uris = [
                f"https://cmip6downscaling.azureedge.net/{dataset.split('//')[-1]}"
                for dataset in stores
            ]
        else:
            original_dataset_uris = []
        return {
            'name': name,
            **query,
            'timescale': timescale,
            'aggregation': aggregation,
            'uri': uri,
            'original_dataset_uris': original_dataset_uris,
        }

    except Exception:
        return {INVALID_ASSET: path, TRACEBACK: traceback.format_exc()}


@task(log_stdout=True)
def get_cmip6_pyramids(paths: list[str]):
    import ecgtools

    builder = ecgtools.Builder(
        paths=paths,
        depth=1,
        joblib_parallel_kwargs={'n_jobs': -1, 'verbose': 1},
        exclude_patterns=["*.json"],
    )
    builder.build(parsing_func=parse_cmip6)

    return builder.df.to_dict(orient='records')


@task(log_stdout=True)
def create_catalog(*, catalog_path: str, cmip6_pyramids=None, era5_pyramids=None):
    import json

    import fsspec

    datasets = []
    if cmip6_pyramids:
        datasets += cmip6_pyramids
    if era5_pyramids:
        datasets += era5_pyramids

    catalog = {
        "version": "1.0.0",
        "title": "CMIP6 downscaling catalog",
        "description": "",
        "history": "",
        "datasets": datasets,
    }

    fs = fsspec.filesystem('az', account_name='cmip6downscaling')
    with fs.open(catalog_path, 'w') as f:
        json.dump(catalog, f, indent=2)

    print(f'web-catalog located at: {catalog_path}')


with Flow(
    'web-catalog', executor=runtime.executor, run_config=runtime.run_config, storage=runtime.storage
) as flow:

    paths = Parameter('paths', default=['az://flow-outputs/results/cmip6-pyramids-raw'])
    web_catalog_path = Parameter(
        'web-catalog-path',
        default='az://flow-outputs/results/pyramids/combined-cmip6-era5-pyramids-catalog-web.json',
    )
    cmip6_pyramids = get_cmip6_pyramids(paths)
    create_catalog(catalog_path=web_catalog_path, cmip6_pyramids=cmip6_pyramids)
