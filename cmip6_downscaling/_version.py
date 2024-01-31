from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version('cmip6_downscaling')
except _PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
