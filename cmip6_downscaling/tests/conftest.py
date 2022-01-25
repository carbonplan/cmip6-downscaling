import pytest

from cmip6_downscaling import config


@pytest.fixture(scope="session", autouse=True)
def set_test_config():
    config.set(
        {
            'storage.intermediate.uri': '/tmp/intermediate',
            'storage.results.uri': '/tmp/results',
            'storage.temporary.uri': '/tmp/temporary',
        }
    )
