import pytest

from cmip6_downscaling.runtimes import (
    BaseRuntime,
    CIRuntime,
    LocalRuntime,
    PangeoRuntime,
    get_runtime,
)


@pytest.mark.parametrize('runtime', [LocalRuntime, CIRuntime, PangeoRuntime])
def test_runtimes(runtime):

    _runtime = runtime()
    assert isinstance(_runtime, BaseRuntime)


def test_get_runtime():
    runtime = get_runtime()
    assert isinstance(runtime, BaseRuntime)
