import time
import uuid

import ESMF
import numpy as np
from dask.distributed import get_worker
from prefect import Flow, task

from cmip6_downscaling.runtimes import get_runtime

runtime = get_runtime()


@task(log_stdout=True, tags=['dask-resource:TASKSLOTS=1'])
def my_task(num: int) -> None:
    time.sleep(30)
    print(num, get_worker().id, uuid.uuid4().hex)


@task(log_stdout=True, tags=['dask-resource:TASKSLOTS=1'])
def make_grid(shape):
    print(shape, get_worker().id, uuid.uuid4().hex)
    time.sleep(30)
    _ = ESMF.Grid(
        np.array(shape),
        staggerloc=ESMF.StaggerLoc.CENTER,
        coord_sys=ESMF.CoordSys.SPH_DEG,
        num_peri_dims=None,  # with out this, ESMF seems to seg fault (clue?)
    )
    return shape


with Flow(
    name="test_cloud_flow",
    storage=get_runtime.storage,
    run_config=get_runtime.run_config,
    executor=get_runtime.executor,
) as flow:

    # nums = range(4)
    # my_task.map(nums)

    tasks = [
        make_grid((590, 870)),
        make_grid((600, 880)),
        make_grid((610, 890)),
        make_grid((620, 900)),
    ]
