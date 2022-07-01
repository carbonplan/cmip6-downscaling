from __future__ import annotations

import datetime
import json
import traceback

import fsspec
import numpy as np
import pandas as pd
from prefect import Flow, Parameter, task
from upath import UPath

from cmip6_downscaling import config, runtimes

runtime = runtimes.CloudRuntime()

config.set(
    {
        'runtime.cloud.extra_pip_packages': 'git+https://github.com/carbonplan/cmip6-downscaling.git git+https://github.com/intake/intake-esm.git git+https://github.com/ncar-xdev/ecgtools.git'
    }
)


def parse_store(store: str):

    from ecgtools.builder import INVALID_ASSET, TRACEBACK

    try:
        path = UPath(store)
        path_parts = path.parts

        (
            activity_id,
            institution_id,
            source_id,
            experiment_id,
            member_id,
            timescale,
            method,
            variable_id,
            _,
        ) = path_parts[-1].split('.')
        return {
            'activity_id': activity_id,
            'institution_id': institution_id,
            'source_id': source_id,
            'experiment_id': experiment_id,
            'member_id': member_id,
            'timescale': timescale,
            'variable_id': variable_id,
            'method': method,
            'downscaled_daily_data_uri': f"{store.replace('abfs://', 'https://cmip6downscaling.blob.core.windows.net/')}",
            'version': 'v1',
        }

    except Exception:

        return {INVALID_ASSET: path, TRACEBACK: traceback.format_exc()}


@task(log_stdout=True)
def generate_intake_esm_catalog(*, intake_esm_catalog_bucket: str):

    import ecgtools

    builder = ecgtools.Builder(
        paths=['az://version1/data'],
        depth=1,
        joblib_parallel_kwargs={'n_jobs': -1, 'verbose': 1},
        exclude_patterns=["*.json"],
    )
    builder.build(parsing_func=parse_store)

    builder.save(
        name="global-downscaled-cmip6",
        description="Global downscaled climate projections from CMIP6",
        path_column_name="downscaled_daily_data_uri",
        variable_column_name="variable_id",
        data_format="zarr",
        groupby_attrs=[
            "activity_id",
            "institution_id",
            "source_id",
            "experiment_id",
            "timescale",
            "method",
        ],
        aggregations=[
            {'type': 'union', 'attribute_name': 'variable_id'},
            {
                "type": "join_new",
                "attribute_name": "member_id",
                "options": {"coords": "minimal", "compat": "override"},
            },
        ],
        catalog_type='dict',
        directory=intake_esm_catalog_bucket,
    )


@task(log_stdout=True)
def generate_minified_web_catalog(
    *, parent_catalog: str, web_catalog: str, cdn: str = None
) -> None:

    with fsspec.open(parent_catalog) as f:
        data = json.load(f)

    df = pd.DataFrame.from_records(data['datasets'])
    df = (
        df.drop(
            columns=[
                'cmip6_downscaling_version',
                'source_path',
                'daily_downscaled_data_uri',
                'table_id',
            ]
        )
        .rename(columns={'destination_path': 'daily_downscaled_data_uri'})
        .replace(np.nan, None)
    )
    if cdn:
        df['uri'] = df['uri'].apply(
            lambda x: x.replace('https://cmip6downscaling.blob.core.windows.net', cdn) if x else x
        )
    print(df.head())

    catalog = {
        "version": "v1.0.0",
        "title": "CMIP6 downscaling catalog",
        "description": "Global downscaled climate projections from CMIP6",
        "history": "",
        "last_updated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": df.to_dict(orient='records'),
    }

    with fsspec.open(web_catalog, 'w') as f:
        json.dump(catalog, f, indent=2)

    print(f'Wrote minified web catalog to: {web_catalog}')


with Flow(
    'final-catalog',
    executor=runtime.executor,
    storage=runtime.storage,
    run_config=runtime.run_config,
) as flow:

    parent_catalog = Parameter(
        'parent-catalog',
        default="az://scratch/results/pyramids/combined-cmip6-era5-pyramids-catalog-web.json",
    )

    web_catalog = Parameter(
        'web-catalog',
        default="az://scratch/results/pyramids/minified-pyramids-web-catalog.json",
    )

    intake_esm_catalog_bucket = Parameter(
        'intake-esm-catalog-bucket', default='az://version1/catalogs/'
    )

    cdn = Parameter('cdn', default=None)

    generate_minified_web_catalog(parent_catalog=parent_catalog, web_catalog=web_catalog, cdn=cdn)
    generate_intake_esm_catalog(intake_esm_catalog_bucket=intake_esm_catalog_bucket)
