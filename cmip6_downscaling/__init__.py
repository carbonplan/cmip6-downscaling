from dask.utils import SerializableLock
from donfig import Config

from cmip6_downscaling.config import _defaults

config = Config("cmip6_downscaling", defaults=[_defaults])
config.config_lock = SerializableLock()

CLIMATE_NORMAL_PERIOD = (1970, 2000)
