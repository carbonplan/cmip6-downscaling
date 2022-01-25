from cmip6_downscaling.runtimes import (  # CloudRuntime,
    BaseRuntime,
    LocalRuntime,
    PangeoRuntime,
    TestRuntime,
    get_runtime,
)


def test_runtimes():

    # cloud_runtime = CloudRuntime()
    # assert isinstance(cloud_runtime, BaseRuntime)
    local_runtime = LocalRuntime()
    assert isinstance(local_runtime, BaseRuntime)
    test_runtime = TestRuntime()
    assert isinstance(test_runtime, BaseRuntime)
    pangeo_runtime = PangeoRuntime()
    assert isinstance(pangeo_runtime, BaseRuntime)


def test_get_runtime():
    runtime = get_runtime()
    assert isinstance(runtime, BaseRuntime)
