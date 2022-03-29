from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("package-name")
except PackageNotFoundError:
    # package is not installed
    __version__ = 'unknown'
