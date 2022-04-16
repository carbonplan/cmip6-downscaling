from prefect import Flow, Parameter, task, unmapped
from tqdm import tqdm

from cmip6_downscaling import config
from cmip6_downscaling.methods.common.utils import zmetadata_exists
from cmip6_downscaling.runtimes import PangeoRuntime

config.set({"runtime.pangeo.n_workers": 10, "runtime.pangeo.threads_per_worker": 1})

runtime = PangeoRuntime()


@task(log_stdout=True)
def get_datasets_keys(*, catalog_path: str, start: int, stop: int):
    import intake
    import intake_esm

    print(intake_esm.__version__)

    cat = intake.open_esm_datastore(catalog_path)
    print(cat)
    keys = sorted(cat.keys())[start:stop]
    print(keys)
    return keys


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

        product_type = ds.attrs['intake_esm_attrs/product_type']
        year = ds.attrs['intake_esm_attrs/year']

        target = (
            f'az://training/ERA5_daily_full_space/{product_type}/{year}/{new_variable_name}.zarr'
        )

        if zmetadata_exists(target):
            print(f"found existing target: {target}")
            return target
        mode = ds.attrs['intake_esm_attrs/aggregation_method']
        resampler = ds.resample(time='1D')
        method = getattr(resampler, mode)
        ds = method(keep_attrs=True)
        ds = ds.chunk({'time': 30, 'lat': -1, 'lon': -1})

        if new_variable_name == 'pr':
            ds[new_variable_name] *= 1000
            ds[new_variable_name].attrs['original_units'] = ds[new_variable_name].attrs['units']
            ds[new_variable_name].attrs['units'] = 'mm'

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
    start = Parameter('start', default=0)
    stop = Parameter('stop', default=-1)
    keys = get_datasets_keys(catalog_path=catalog_path, start=start, stop=stop)
    _ = resample_to_daily.map(catalog_path=unmapped(catalog_path), key=keys)


if __name__ == '__main__':
    step = 10
    values = [(start, start + step) for start in range(30, 550, step)]
    for start, stop in tqdm(values):
        print(start, stop)
        flow.run(start=start, stop=stop)
