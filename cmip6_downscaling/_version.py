from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cmip6-downscaling")
except PackageNotFoundError:
    # package is not installed
    __version__ = 'unknown'
