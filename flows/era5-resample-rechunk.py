from prefect import Flow, Parameter, task, unmapped

from cmip6_downscaling import config
from cmip6_downscaling.runtimes import PangeoRuntime

config.set({"runtime.pangeo.n_workers": 10, "runtime.pangeo.threads_per_worker": 1})

# config.set(
#     {
#         'runtime.cloud.extra_pip_packages': 'git+https://github.com/carbonplan/cmip6-downscaling.git@main git+https://github.com/intake/intake-esm.git'
#     }
# )

runtime = PangeoRuntime()


@task(log_stdout=True)
def get_datasets_keys(*, catalog_path: str):
    import intake
    import intake_esm

    print(intake_esm.__version__)

    cat = intake.open_esm_datastore(catalog_path).search(standard_name='integral*')
    print(cat)
    return list(cat.keys())


@task(log_stdout=True)
def resample_to_daily(*, catalog_path: str, key: str):

    import dask
    import intake
    import xarray as xr

    cat = intake.open_esm_datastore(catalog_path)
    ds = cat[key](xarray_open_kwargs={'chunks': {}}).to_dask()

    with xr.set_options(keep_attrs=True):
        old_variable_name = ds.attrs['intake_esm_attrs/standard_name']
        new_variable_name = ds.attrs['intake_esm_attrs/cf_variable_name']
        ds = ds.rename({old_variable_name: new_variable_name})
        mode = ds.attrs['intake_esm_attrs/aggregation_method']
        resampler = ds.resample(time='1D')
        method = getattr(resampler, mode)
        ds = method(keep_attrs=True)
        ds = ds.chunk({'time': 30, 'lat': -1, 'lon': -1})

        if new_variable_name == 'pr':
            ds[new_variable_name] *= 1000
            ds[new_variable_name].attrs['original_units'] = ds[new_variable_name].attrs['units']
            ds[new_variable_name].attrs['units'] = 'mm'

        product_type = ds.attrs['intake_esm_attrs/product_type']
        year = ds.attrs['intake_esm_attrs/year']

        target = (
            f'az://training/ERA5_daily_full_space/{product_type}/{year}/{new_variable_name}.zarr'
        )

        for attr in list(ds.attrs):
            if attr.startswith('intake_esm'):
                del ds.attrs[attr]

        ds = dask.optimize(ds)[0]
        to_zarr = ds.to_zarr(target, mode='w', consolidated=True, compute=False)
        to_zarr.compute(retries=5)
        print(f'finished writing: {target}')


with Flow(
    'ERA5-resample-rechunk',
    storage=runtime.storage,
    run_config=runtime.run_config,
    executor=runtime.executor,
) as flow:
    catalog_path = Parameter('catalog_path', default='az://training/ERA5-azure.json')
    keys = get_datasets_keys(catalog_path=catalog_path)
    _ = resample_to_daily.map(catalog_path=unmapped(catalog_path), key=keys)
