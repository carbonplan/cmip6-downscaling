import xarray as xr
from prefect import Flow, Parameter, task

from cmip6_downscaling import config
from cmip6_downscaling.runtimes import CloudRuntime

config.set(
    {
        'runtime.cloud.extra_pip_packages': 'git+https://github.com/carbonplan/cmip6-downscaling.git@main git+https://github.com/intake/intake-esm.git tabulate'
    }
)

runtime = CloudRuntime()


@task(log_stdout=True)
def get_datasets(*, catalog_path: str):
    import intake
    import intake_esm

    print(intake_esm.__version__)

    cat = intake.open_esm_datastore(catalog_path).search(short_name='mx2t')
    dsets = cat.to_dataset_dict(xarray_open_kwargs=dict(chunks={}))
    print(dsets.keys())
    return list(dsets.values())


@task(log_stdout=True)
def resample_to_daily(*, ds: xr.Dataset):

    old_variable_name = ds.attrs['intake_esm_attrs/standard_name']
    new_variable_name = ds.attrs['intake_esm_attrs/cf_variable_name']
    ds = ds.rename({old_variable_name: new_variable_name})

    mode = ds.attrs['intake_esm_attrs/aggregation_method']
    resampler = ds.resample(time='1D')
    method = getattr(resampler, mode)
    ds = method(keep_attrs=True).chunk({'time': 30})

    for attr in list(ds.attrs):
        if attr.startswith('intake_esm'):
            del ds.attrs[attr]
    print(ds)


with Flow(
    'ERA5-resample-rechunk',
    storage=runtime.storage,
    run_config=runtime.run_config,
    executor=runtime.executor,
) as flow:
    catalog_path = Parameter('catalog_path', default='az://training/ERA5-azure.json')
    dsets = get_datasets(catalog_path=catalog_path)
    _ = resample_to_daily.map(ds=dsets)
