import time
import uuid

import ESMF
import numpy as np
from dask.distributed import get_worker
from prefect import Flow, task

from cmip6_downscaling import runtimes


@task(log_stdout=True, tags=['dask-resource:TASKSLOTS=1'])
def my_task(num: int) -> None:
    time.sleep(1)
    print(num, get_worker().id, uuid.uuid4().hex)


@task(log_stdout=True, tags=['dask-resource:TASKSLOTS=1'])
def make_grid(shape):
    print(shape, get_worker().id, uuid.uuid4().hex)
    time.sleep(1)
    _ = ESMF.Grid(
        np.array(shape),
        staggerloc=ESMF.StaggerLoc.CENTER,
        coord_sys=ESMF.CoordSys.SPH_DEG,
        num_peri_dims=None,  # with out this, ESMF seems to seg fault (clue?)
    )
    return shape


runtime = runtimes.get_runtime()

with Flow(
    name="test_task_multi",
    storage=runtime.storage,
    run_config=runtime.run_config,
    executor=runtime.executor,
) as flow:

    nums = range(4)
    my_task.map(nums)

    tasks = [make_grid((59, 87)), make_grid((60, 88)), make_grid((61, 89)), make_grid((62, 90))]
