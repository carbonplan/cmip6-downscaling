from cmip6_downscaling import config


def test_config():
    # check that required config keys are always there
    assert config.get('storage.intermediate.uri')
    assert config.get('storage.results.uri')
    assert config.get('storage.temporary.uri')
    assert config.get('runtime.cloud.storage.options')
    assert config.get('runtime.local.storage_options')
    assert config.get('runtime.test.storage_options')
    assert config.get('runtime.pangeo.storage_options')
