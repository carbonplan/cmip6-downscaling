from prefect import Flow, Parameter, task
from prefect.backend.artifacts import create_markdown_artifact

from cmip6_downscaling import config
from cmip6_downscaling.runtimes import CloudRuntime

config.set(
    {
        'runtime.cloud.extra_pip_packages': 'git+https://github.com/carbonplan/cmip6-downscaling.git@main git+https://github.com/ncar-xdev/ecgtools.git git+https://github.com/intake/intake-esm.git tabulate'
    }
)

runtime = CloudRuntime()


var_name_dict = {
    "eastward_wind_at_10_metres": {'cmip6_name': "uas", 'aggregation_method': "mean"},
    "northward_wind_at_10_metres": {'cmip6_name': "vas", 'aggregation_method': "mean"},
    "eastward_wind_at_100_metres": {'cmip6_name': "ua100m", 'aggregation_method': "mean"},
    "northward_wind_at_100_metres": {'cmip6_name': "va100m", 'aggregation_method': "mean"},
    "dew_point_temperature_at_2_metres": {'cmip6_name': "tdps", 'aggregation_method': "mean"},
    "air_temperature_at_2_metres": {'cmip6_name': "tas", 'aggregation_method': "mean"},
    "air_temperature_at_2_metres_1hour_Maximum": {
        'cmip6_name': "tasmax",
        'aggregation_method': "max",
    },
    "air_temperature_at_2_metres_1hour_Minimum": {
        'cmip6_name': "tasmin",
        'aggregation_method': "min",
    },
    "air_pressure_at_mean_sea_level": {'cmip6_name': "psl", 'aggregation_method': "mean"},
    "sea_surface_temperature": {'cmip6_name': "tos", 'aggregation_method': "mean"},
    "surface_air_pressure": {'cmip6_name': "ps", 'aggregation_method': "mean"},
    "integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation": {
        'cmip6_name': "rsds",
        'aggregation_method': "sum",
    },
    "precipitation_amount_1hour_Accumulation": {'cmip6_name': "pr", 'aggregation_method': "sum"},
}


def parse_era5(path):
    import xarray as xr
    from upath import UPath

    _upath = UPath(path)
    parts = _upath.parts
    attrs = {'year': parts[-3], 'month': parts[-2], 'standard_name': _upath.stem}
    ds = xr.open_zarr(path)
    var_attrs = ds[attrs['standard_name']].attrs
    attrs['product_type'] = var_attrs['product_type']
    attrs['short_name'] = var_attrs['shortNameECMWF']
    attrs['cf_variable_name'] = var_name_dict[attrs['standard_name']]['cmip6_name']
    attrs['aggregation_method'] = var_name_dict[attrs['standard_name']]['aggregation_method']
    attrs['units'] = var_attrs['units']
    attrs['zstore'] = path
    attrs['timescale'] = 'hourly'
    attrs['time'] = f'{attrs["year"]}-{attrs["month"]}'
    return attrs


@task(log_stdout=True)
def build_catalog(*, name: str, bucket: str) -> None:

    import ecgtools

    print(ecgtools.__version__)

    storage_options = {'account_name': 'cmip6downscaling'}
    builder = ecgtools.Builder(
        paths=['az://training/ERA5'],
        depth=2,
        storage_options=storage_options,
        joblib_parallel_kwargs={'n_jobs': -1, 'verbose': 2},
        exclude_patterns=["*.json"],
    )

    builder.build(
        parsing_func=parse_era5,
    )

    print(builder.df.head())

    builder.save(
        name=name,
        path_column_name='zstore',
        variable_column_name='standard_name',
        data_format='zarr',
        aggregations=[
            {
                'type': 'join_existing',
                'attribute_name': 'time',
                "options": {"dim": "time", "coords": "minimal"},
            }
        ],
        groupby_attrs=['product_type', 'short_name', 'timescale', 'year'],
        directory=bucket,
    )


@task(log_stdout=True)
def create_report(*, name: str, bucket: str) -> None:
    import intake
    import intake_esm

    print(intake_esm.__version__)
    cat = intake.open_esm_datastore(f'{bucket}/{name}.json')
    print(cat)
    print(cat.keys())

    report = f'''
# ERA5 Catalog Report

{cat.nunique().to_markdown(tablefmt="github")}

'''

    create_markdown_artifact(report)


with Flow('ERA5-catalog', storage=runtime.storage, run_config=runtime.run_config) as flow:
    name = Parameter('name', default='ERA5-azure')
    bucket = Parameter('path', default='az://training')
    _ = create_report(
        name=name, bucket=bucket, upstream_tasks=[build_catalog(name=name, bucket=bucket)]
    )
