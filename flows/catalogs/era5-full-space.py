from prefect import Flow, Parameter, task

from cmip6_downscaling import config
from cmip6_downscaling.runtimes import CloudRuntime

config.set(
    {
        'runtime.cloud.extra_pip_packages': 'git+https://github.com/carbonplan/cmip6-downscaling.git@main git+https://github.com/ncar-xdev/ecgtools.git git+https://github.com/intake/intake-esm.git tabulate'
    }
)

runtime = CloudRuntime()


def parse_era5(path):
    from upath import UPath

    _upath = UPath(path)
    parts = _upath.parts

    return {
        'year': parts[-2],
        'product_type': parts[-3],
        'timescale': 'daily',
        'zstore': path,
        'cf_variable_name': _upath.stem,
    }


@task(log_stdout=True)
def build_catalog(*, name: str, bucket: str) -> None:
    import ecgtools

    print(ecgtools.__version__)

    storage_options = {'account_name': 'cmip6downscaling'}
    builder = ecgtools.Builder(
        paths=[bucket],
        depth=2,
        storage_options=storage_options,
        joblib_parallel_kwargs={'n_jobs': -1, 'verbose': 2},
        exclude_patterns=["*.json"],
    )

    builder.build(parsing_func=parse_era5)
    print(builder.df.head())

    builder.save(
        name=name,
        path_column_name='zstore',
        variable_column_name='cf_variable_name',
        data_format='zarr',
        aggregations=[
            {
                'type': 'join_existing',
                'attribute_name': 'year',
                "options": {"dim": "time", "coords": "minimal"},
            }
        ],
        groupby_attrs=['product_type', 'timescale', 'cf_variable_name'],
        directory=bucket,
    )


with Flow('ERA5-full-space', storage=runtime.storage, run_config=runtime.run_config) as flow:
    name = Parameter('name', default='ERA5-full-space')
    bucket = Parameter('path', default='az://training/ERA5_daily_full_space')
    build_catalog(name=name, bucket=bucket)
