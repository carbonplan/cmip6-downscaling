# flake8: noqa
from __future__ import annotations

from dask.utils import SerializableLock
from donfig import Config

from ._version import __version__
from .config import _defaults

config = Config("cmip6_downscaling", defaults=[_defaults])
config.config_lock = SerializableLock()
config.expand_environment_variables()
CLIMATE_NORMAL_PERIOD = (1970, 2000)
