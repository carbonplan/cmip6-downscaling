import os
import time
import uuid

import ESMF
import numpy as np
from dask.distributed import get_worker
from dask_kubernetes import KubeCluster, make_pod_spec
from prefect import Flow, task
from prefect.executors import DaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure

from cmip6_downscaling import config
from cmip6_downscaling.runtimes import get_runtime

runtime = get_runtime()


@task(log_stdout=True)
def configprinter():
    print(config.config)


@task(log_stdout=True, tags=['dask-resource:TASKSLOTS=1'])
def my_task(num: int) -> None:
    time.sleep(30)
    print(num, get_worker().id, uuid.uuid4().hex)


@task(log_stdout=True, tags=['dask-resource:TASKSLOTS=1'])
def make_grid(shape):
    print(executor.cluster_kwargs())
    print(shape, get_worker().id, uuid.uuid4().hex)

    time.sleep(1)
    _ = ESMF.Grid(
        np.array(shape),
        staggerloc=ESMF.StaggerLoc.CENTER,
        coord_sys=ESMF.CoordSys.SPH_DEG,
        num_peri_dims=None,  # with out this, ESMF seems to seg fault (clue?)
    )
    return shape


image = "carbonplan/cmip6-downscaling-prefect:2022.01.05"
storage = Azure("prefect")

env = {
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
pod_spec = make_pod_spec(
    image=image,
    memory_limit="4Gi",
    memory_request="4Gi",
    threads_per_worker=2,
    cpu_limit=2,
    cpu_request=2,
    env=env,
)
pod_spec.spec.containers[0].args.extend(['--resources', 'TASKSLOTS=1'])

executor = DaskExecutor(
    cluster_class=lambda: KubeCluster(pod_spec, deploy_mode='remote'),
    adapt_kwargs={"minimum": 2, "maximum": 2},
)


# with Flow(
#     name="esmf_stresstest",
#     storage=runtime.storage,
#     run_config=runtime.run_config,
#     executor=runtime.executor,
with Flow(
    name="esmf_stresstest",
    storage=storage,
    run_config=run_config,
    executor=executor,
) as flow:
    # nums = range(4)
    # my_task.map(nums)

    tasks = [
        make_grid((590, 870)),
        make_grid((600, 880)),
        make_grid((610, 890)),
        make_grid((620, 900)),
    ]
