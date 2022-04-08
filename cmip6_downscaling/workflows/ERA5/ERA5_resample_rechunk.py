from prefect import Flow, task

from cmip6_downscaling import config
from cmip6_downscaling.runtimes import CloudRuntime

print(config)

runtime = CloudRuntime()


@task
def foo():
    import xarray as xr

    ds = xr.tutorial.open_dataset('rasm')
    print(ds)
    print("Hello World!")


with Flow(
    'ERA5-resample-rechunk',
    storage=runtime.storage,
    run_config=runtime.run_config,
) as flow:
    foo()
