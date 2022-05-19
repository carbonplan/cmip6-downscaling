import functools
import json

import fsspec
from prefect import Flow, Parameter, task

from cmip6_downscaling import config, runtimes

config.set(
    {
        'runtime.cloud.extra_pip_packages': 'git+https://github.com/carbonplan/cmip6-downscaling.git git+https://github.com/intake/intake-esm.git git+https://github.com/ncar-xdev/ecgtools.git'
    }
)

runtime = runtimes.CloudRuntime()


def parse_cmip6(store, cdn):
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

        uri = f"{cdn}/{str(path).split('//')[-1]}"
        cat_url = "https://cmip6downscaling.blob.core.windows.net/cmip6/pangeo-cmip6.json"
        cat = intake.open_esm_datastore(cat_url)
        stores = cat.search(**query).df.zstore.tolist()
        if stores:
            original_dataset_uris = [f"{cdn}/{dataset.split('//')[-1]}" for dataset in stores]
        else:
            original_dataset_uris = []
        return {
            'name': name,
            **query,
            'timescale': timescale,
            'aggregation': aggregation,
            'uri': uri,
            'original_dataset_uris': original_dataset_uris,
            'method': 'raw',
        }

    except Exception:
        return {INVALID_ASSET: path, TRACEBACK: traceback.format_exc()}


@task(log_stdout=True)
def get_cmip6_pyramids(paths: list[str], cdn: str):
    import ecgtools

    builder = ecgtools.Builder(
        paths=paths,
        depth=1,
        joblib_parallel_kwargs={'n_jobs': -1, 'verbose': 1},
        exclude_patterns=["*.json"],
    )
    builder.build(parsing_func=functools.partial(parse_cmip6, cdn=cdn))

    return builder.df.to_dict(orient='records')


def parse_cmip6_downscaled_pyramid(data, cdn: str):
    import intake
    from upath import UPath

    parameters = data['parameters']

    datasets = data['datasets']

    query = {
        'source_id': parameters['model'],
        'member_id': parameters['member'],
        'experiment_id': parameters['scenario'],
        'variable_id': parameters['variable'],
    }

    cat_url = "https://cmip6downscaling.blob.core.windows.net/cmip6/pangeo-cmip6.json"
    cat = intake.open_esm_datastore(cat_url).search(table_id='day', **query)
    records = cat.df.to_dict(orient='records')
    if records and len(records) == 1:
        entry = records[0]
        query['institution_id'] = entry['institution_id']
        query['activity_id'] = entry['activity_id']
        query['original_dataset_uris'] = [f"{cdn}/{str(entry['zstore']).split('//')[-1]}"]

    template = {**query, 'method': parameters['method']}
    results = []
    if 'daily_pyramid_path' in datasets:
        daily_pyramid_attrs = {
            **template,
            **{
                'timescale': 'day',
                'uri': f"{cdn}/{str(datasets['daily_pyramid_path']).split('//')[-1]}",
                'name': UPath(datasets['daily_pyramid_path']).name,
            },
        }
        results.append(daily_pyramid_attrs)
    if 'monthly_pyramid_path' in datasets:
        monthly_pyramid_attrs = {
            **template,
            **{
                'timescale': 'month',
                'uri': f"{cdn}/{str(datasets['monthly_pyramid_path']).split('//')[-1]}",
                'name': UPath(datasets['monthly_pyramid_path']).name,
            },
        }
        results.append(monthly_pyramid_attrs)
    if 'annual_pyramid_path' in datasets:
        yearly_pyramid_attrs = {
            **template,
            **{
                'timescale': 'year',
                'uri': f"{cdn}t/{str(datasets['annual_pyramid_path']).split('//')[-1]}",
                'name': UPath(datasets['annual_pyramid_path']).name,
            },
        }
        results.append(yearly_pyramid_attrs)

    for attrs in results:
        if attrs['timescale'] in {'year', 'month'}:
            if attrs['variable_id'] in {'tasmin', 'tasmax'}:
                attrs['aggregation'] = 'mean'
            elif attrs['variable_id'] in {'pr'}:
                attrs['aggregation'] = 'sum'
    return results


@task(log_stdout=True)
def get_cmip6_downscaled_pyramids(path, cdn):

    mapper = fsspec.get_mapper(path)
    fs = mapper.fs
    latest_runs = fs.glob(path)
    datasets = []
    for run in latest_runs:
        with fs.open(run) as f:
            data = json.load(f)

        if any(
            data['datasets'].get(arg) is not None
            for arg in ['daily_pyramid_path', 'monthly_pyramid_path', 'annual_pyramid_path']
        ):
            datasets += parse_cmip6_downscaled_pyramid(data, cdn)
    return datasets


@task(log_stdout=True)
def create_catalog(
    *,
    catalog_path: str,
    cmip6_raw_pyramids=None,
    cmip6_downscaled_pyramids=None,
    era5_pyramids=None,
):
    import json

    import fsspec

    datasets = []
    if cmip6_raw_pyramids:
        datasets += cmip6_raw_pyramids
    if cmip6_downscaled_pyramids:
        datasets += cmip6_downscaled_pyramids
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
    downscaled_pyramids = Parameter(
        'downscaled-pyramids-path',
        default='az://flow-outputs/results/0.1.[3,5]/runs/*/latest.json',
    )

    # https://cmip6downscaling.azureedge.net
    cdn = Parameter('cdn', default='https://cmip6downscaling.blob.core.windows.net')

    cmip6_raw_pyramids = get_cmip6_pyramids(paths, cdn)
    cmip6_downscaled_pyramids = get_cmip6_downscaled_pyramids(downscaled_pyramids, cdn)
    create_catalog(
        catalog_path=web_catalog_path,
        cmip6_raw_pyramids=cmip6_raw_pyramids,
        cmip6_downscaled_pyramids=cmip6_downscaled_pyramids,
    )
