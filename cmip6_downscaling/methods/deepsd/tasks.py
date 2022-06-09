import xarray as xr
from carbonplan_data.metadata import get_cf_global_attrs
from prefect import task
from upath import UPath

from ... import __version__ as version, config
from ..common.containers import RunParameters, str_to_hash
from ..common.utils import set_zarr_encoding, zmetadata_exists
from .utils import bilinear_interpolate, conservative_interpolate

scratch_dir = UPath(config.get("storage.scratch.uri"))
intermediate_dir = UPath(config.get("storage.intermediate.uri")) / version
results_dir = UPath(config.get("storage.results.uri")) / version
use_cache = config.get('run_options.use_cache')


@task(log_stdout=True)
def shift_coarsen_interpolate(orig_path: UPath, run_parameters: RunParameters) -> UPath:
    """Shift and coarsen grid then interpolate back to twice the coarsened grid resolution.
    Parameters
    ----------
    orig_path : UPath
        Path to original (likely observational) dataset
    run_parameters : RunParameters
        Prefect run parameters

    Returns
    -------
    UPath
        Path to interpolated dataset.
    """
    # Similar to coarsen_and_interpolate in GARD tasks (maybe could be combined?)
    ds_hash = str_to_hash(str(orig_path))
    target = intermediate_dir / 'shift_coarsen_interpolate' / ds_hash

    if use_cache and zmetadata_exists(target):
        print(f'found existing target: {target}')
        interpolated_ds = xr.open_zarr(target)
        return target

    orig_ds = xr.open_zarr(orig_path)

    # Interpolate obs to 720x1440
    # TODO: Jupyter Notebook included precipitation unit conversion during the shift step
    shift_ds = bilinear_interpolate(ds=orig_ds, output_degree=0.25)
    # Coarsen obs
    coarse_ds = conservative_interpolate(ds=shift_ds, output_degree=run_parameters.output_degree)
    # Interpolate back to 2x higher resolution
    interpolated_ds = bilinear_interpolate(
        ds=coarse_ds, output_degree=run_parameters.output_degree * 2
    )
    interpolated_ds.attrs.update(
        {'title': 'shift_coarsen_interpolate'}, **get_cf_global_attrs(version=version)
    )
    interpolated_ds.pipe(set_zarr_encoding).to_zarr(target, mode='w')
    return target
