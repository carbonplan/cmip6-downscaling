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
    "eastward_wind_at_10_metres": "uas",
    "northward_wind_at_10_metres": "vas",
    "eastward_wind_at_100_metres": "ua100m",
    "northward_wind_at_100_metres": "va100m",
    "dew_point_temperature_at_2_metres": "tdps",
    "air_temperature_at_2_metres": "tas",
    "air_temperature_at_2_metres_1hour_Maximum": "tasmax",
    "air_temperature_at_2_metres_1hour_Minimum": "tasmin",
    "air_pressure_at_mean_sea_level": "psl",
    "sea_surface_temperature": "tos",
    "surface_air_pressure": "ps",
    "integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation": "rsds",
    "precipitation_amount_1hour_Accumulation": "pr",
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
    attrs['cf_variable_name'] = var_name_dict.get(attrs['standard_name'], None)
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
        groupby_attrs=['product_type', 'timescale', 'short_name'],
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
