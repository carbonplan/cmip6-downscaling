from __future__ import annotations

from dask.utils import SerializableLock
from donfig import Config
from pkg_resources import DistributionNotFound, get_distribution

from .config import _defaults

try:
    version = get_distribution(__name__).version
except DistributionNotFound:  # pragma: no cover
    version = '0.0.0'  # pragma: no cover
__version__ = version

config = Config("cmip6_downscaling", defaults=[_defaults])
config.config_lock = SerializableLock()
config.expand_environment_variables()
CLIMATE_NORMAL_PERIOD = (1970, 2000)
