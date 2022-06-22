from importlib.metadata import PackageNotFoundError  # , version

try:
    __version__ = "0.1.9"
    # __version__ = version("cmip6-downscaling")
except PackageNotFoundError:
    # package is not installed
    __version__ = 'unknown'
