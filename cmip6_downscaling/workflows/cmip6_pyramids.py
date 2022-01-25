import os

import intake
from intake_esm.merge_util import AggregationError
from prefect import Flow, unmapped
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure

from cmip6_downscaling.tasks.pyramid import regrid

LEVELS = 2
TIME_SLICE = slice('1950', '2100')


image = "carbonplan/cmip6-downscaling-prefect:2021.12.06"
storage = Azure("prefect")
extra_pip_packages = 'git+https://github.com/carbonplan/ndpyramid@a326e5b97257147e05733e068801ecb6b0d17888 git+https://github.com/TomNicholas/datatree@54edbf77fdc756b74bfca00986f4a68e04643b29'
env = {
    'EXTRA_PIP_PACKAGES': extra_pip_packages,
    'AZURE_STORAGE_CONNECTION_STRING': os.environ['AZURE_STORAGE_CONNECTION_STRING'],
    'OMP_NUM_THREADS': '1',
    'MPI_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'DASK_DISTRIBUTED__WORKER__RESOURCES__TASKSLOTS': '1',
}

run_config = KubernetesRun(
    cpu_request=4,
    memory_request="8Gi",
    image=image,
    labels=["az-eu-west"],
    env=env,
)


def get_cat():
    col_url = "https://cmip6downscaling.blob.core.windows.net/cmip6/pangeo-cmip6.json"
    cat = intake.open_esm_datastore(col_url).search(
        member_id=['r1i1p1f1'],
        table_id=["day"],
        grid_label=['gn'],
        variable_id=['tasmin', 'tasmax', 'pr'],
        require_all_on=['variable_id'],
    )
    return cat


with Flow(name="cmip6_pyramids", storage=storage, run_config=run_config) as flow:

    cat = get_cat()

    datasets = []
    uris = []

    for key in cat.keys():
        try:
            ds = (
                cat[key](storage_options={'account_name': 'cmip6downscaling'})
                .to_dask()
                .sel(time=TIME_SLICE)
            )
        except AggregationError:
            print(f'skipping {key}')

        for member_id in ds.member_id.data:
            datasets.append(ds.sel(member_id=member_id))
            uris.append(f'scratch/cmip6-web-test-8/{key}/{member_id}')

    regrid.map(ds=datasets, uri=uris, levels=unmapped(LEVELS), other_chunks=unmapped({'time': 31}))
