import os

import dask
import datatree as dt
import intake
from adlfs import AzureBlobFileSystem
from carbonplan_data.metadata import get_cf_global_attrs
from carbonplan_data.utils import set_zarr_encoding
from intake_esm.merge_util import AggregationError
from ndpyramid import pyramid_regrid
from prefect import Flow, task
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure

# import datetime
# import prefect
# import xarray as xr
# from prefect.executors import DaskExecutor
# from dask_kubernetes import KubeCluster, make_pod_spec

LEVELS = 2
TIME_SLICE = slice('1950', '2100')
# TIME_SLICE = slice('2010', '2020')
PIXELS_PER_TILE = 128


image = "carbonplan/cmip6-downscaling-prefect:2021.12.06"
storage = Azure("prefect")
extra_pip_packages = (
    'git+https://github.com/carbonplan/ndpyramid@e85b8365224f50b69783129000eb39eccdf6c711'
)
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
# pod_spec = make_pod_spec(
#     image=image,
#     memory_limit="8Gi",
#     memory_request="8Gi",
#     threads_per_worker=4,
#     cpu_limit=4,
#     cpu_request=4,
#     env=env,
# )
# pod_spec.spec.containers[0].args.extend(['--resources', 'TASKSLOTS=1'])

# executor = DaskExecutor(
#     cluster_class=lambda: KubeCluster(pod_spec, deploy_mode='remote'),
#     adapt_kwargs={"minimum": 1, "maximum": 2},
# )


def get_fs() -> AzureBlobFileSystem:
    connection_string = os.environ['AZURE_STORAGE_CONNECTION_STRING']
    fs = AzureBlobFileSystem(connection_string=connection_string)
    return fs


def get_target_pyramid(fs) -> dt.DataTree:
    mapper = fs.get_mapper('scratch/epsg_3857_grid_pyramid.zarr')
    return dt.open_datatree(mapper, engine='zarr')


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


def postprocess(dt: dt.DataTree) -> dt.DataTree:

    for level in range(len(dt.children)):
        dt.ds.attrs['multiscales'][0]['datasets'][level]['pixels_per_tile'] = PIXELS_PER_TILE

    for child in dt.children:

        child.ds = child.ds.chunk({"x": PIXELS_PER_TILE, "y": PIXELS_PER_TILE, "time": 31})
        child.ds['date_str'] = child.ds['date_str'].chunk(-1)

        child.ds = set_zarr_encoding(
            child.ds, codec_config={"id": "zlib", "level": 1}, float_dtype="float32"
        )
        child.ds.time.encoding['dtype'] = 'int32'
        child.ds.time_bnds.encoding['dtype'] = 'int32'
    dt.ds.attrs.update(**get_cf_global_attrs())
    return dt


@task(log_stdout=True)
#  tags=['dask-resource:TASKSLOTS=1'], retry_delay=datetime.timedelta(seconds=30)
def regrid(key: str, levels: int = LEVELS) -> None:

    print(key, levels)

    fs = get_fs()

    with dask.config.set(scheduler='threads'):

        cat = get_cat()

        try:
            ds = (
                cat[key](storage_options={'account_name': 'cmip6downscaling'})
                .to_dask()
                .sel(time=TIME_SLICE)
            )
        except AggregationError:
            return

        ds.coords['date_str'] = ds['time'].dt.strftime('%Y-%m-%d').astype('S10')
        print('dataset:')
        print(ds)

        # explicitly load all coordinates
        for var, da in ds.coords.items():
            ds[var] = da.load()

        # print('getting target')
        # target = get_target_pyramid(fs)

        for member_id in ds.member_id.data:

            uri = f'scratch/cmip6-web-test-7/{key}/{member_id}'
            mapper = fs.get_mapper(uri)
            if '.zmetadata' in mapper:
                return

            # with dask.config.set(scheduler='single-threaded'):

            print('regridding')
            dt = pyramid_regrid(
                ds.sel(member_id=member_id),
                target_pyramid=None,
                levels=levels,
            )

            print('postprocess')
            dt = postprocess(dt)
            print(str(dt))

            # write
            print(f'starting to write {uri}')
            dt.to_zarr(mapper, mode='w')


with Flow(name="cmip6_pyramids", storage=storage, run_config=run_config) as flow:

    cat = get_cat()
    keys = list(cat.keys())
    print(keys)

    regrid.map(keys)
