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
def load_data(*, catalog_path: str):
    import intake

    cat = intake.open_esm_datastore(catalog_path)
    dsets = cat.to_dataset_dict(chunks={})
    print(dsets.keys())
    print(dsets)


with Flow('ERA5-resample', storage=runtime.storage, run_config=runtime.run_config) as flow:
    catalog_path = Parameter('catalog_path', default='az://training/ERA5-azure.json')
    load_data(catalog_path=catalog_path)
