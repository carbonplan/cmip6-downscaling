import xarray as xr
import xesmf as xe
from carbonplan_data.metadata import get_cf_global_attrs
from prefect import task
from upath import UPath

from ... import config
from ..._version import __version__
from ..common.utils import zmetadata_exists
from .containers import RunParameters

version = __version__
scratch_dir = UPath(config.get("storage.scratch.uri"))
intermediate_dir = UPath(config.get("storage.intermediate.uri")) / __version__
results_dir = UPath(config.get("storage.results.uri")) / __version__
use_cache = config.get('run_options.use_cache')


@task(tags=['dask-resource:TASKSLOTS=1'], log_stdout=True)
def coarsen_and_interpolate(fine_path: UPath, coarse_path: UPath) -> UPath:
    """
    Coarsen up obs and then interpolate it back to the original finescale grid.
    Parameters
    ----------
    fine_path : UPath
        Path to finescale (likely observational) dataset
    coarse_path : UPath
        Path to coarse scale that will be the template for the coarsening.

    Returns
    -------
    UPath
        Path to interpolated dataset.
    """
    ds_name = f'source_path_{fine_path.name}/target_path_{coarse_path.name}'
    target = intermediate_dir / 'coarsen_and_interpolate' / ds_name

    if use_cache and zmetadata_exists(target):
        print(f'found existing target: {target}')
        return target

    fine_ds = xr.open_zarr(fine_path)
    target_ds = xr.open_zarr(coarse_path)

    # coarsen
    regridder = xe.Regridder(fine_ds, target_ds, "bilinear", extrap_method="nearest_s2d")
    coarse_ds = regridder(fine_ds)

    # interpolate back to the fine grid
    regridder = xe.Regridder(coarse_ds, fine_ds, "bilinear", extrap_method="nearest_s2d")
    interpolated_ds = regridder(coarse_ds)

    interpolated_ds.attrs.update({'title': ds_name}, **get_cf_global_attrs(version=version))
    interpolated_ds.to_zarr(target, mode='w')

    return target


@task(tags=['dask-resource:TASKSLOTS=1'], log_stdout=True)
def fit_and_predict(
    xtrain_path: UPath, ytrain_path: UPath, xpred_path: UPath, run_parameters: RunParameters
) -> UPath:
    """Prepare inputs (e.g. normalize), use them to fit a GARD model based upon
    specified parameters and then use that fitted model to make a prediction.

    Parameters
    ----------
    xtrain_path : UPath
        Path to training dataset (interpolated obs)
    ytrain_path : UPath
        Path to target dataset (interpolated GCM)
    xpred_path : UPath
        Path to future prediction dataset (interpolated GCM)
    run_parameters : RunParameters
        Parameters for run set-up and model specs

    Returns
    -------
    UPath
        Path to output dataset
    """
