import os
import time
import uuid

from dask.distributed import get_worker
from dask_kubernetes import KubeCluster, make_pod_spec
from prefect import Flow, task
from prefect.executors import DaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure

image = "carbonplan/cmip6-downscaling-prefect:2021.12.06"
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
    adapt_kwargs={"minimum": 1, "maximum": 2},
)


@task(log_stdout=True, tags=['dask-resource:TASKSLOTS=1'])
def my_task(num: int) -> None:
    time.sleep(30)
    print(num, get_worker().id, uuid.uuid4().hex)


with Flow(
    name="test_cloud_flow", storage=storage, run_config=run_config, executor=executor
) as flow:

    nums = range(4)
    my_task.map(nums)
