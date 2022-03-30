import xarray as xr
import xesmf as xe
from carbonplan_data.metadata import get_cf_global_attrs
from prefect import task
from scipy.stats import norm as norm
from skdownscale.pointwise_models import (  # AnalogRegression,; PureAnalog,; PureRegression,
    PointWiseDownscaler,
)
from skdownscale.pointwise_models.utils import default_none_kwargs
from upath import UPath

from ... import config
from ..._version import __version__
from ..common.bias_correction import bias_correct_gcm_by_method, bias_correct_obs_by_method
from ..common.utils import zmetadata_exists
from .containers import RunParameters
from .utils import get_gard_model, read_scrf

code_version = __version__
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

    interpolated_ds.attrs.update({'title': ds_name}, **get_cf_global_attrs(version=code_version))
    interpolated_ds.to_zarr(target, mode='w')

    return target


@task(tags=['dask-resource:TASKSLOTS=1'], log_stdout=True)
def fit_and_predict(
    xtrain_path: UPath,
    ytrain_path: UPath,
    xpred_path: UPath,
    run_parameters: RunParameters,
    dim: str = 'time',
) -> UPath:
    """Prepare inputs (e.g. normalize), use them to fit a GARD model based upon
    specified parameters and then use that fitted model to make a prediction.

    Parameters
    ----------
    xtrain_path : UPath
        Path to training dataset (interpolated GCM) chunked full_time
    ytrain_path : UPath
        Path to target dataset (interpolated obs) chunked full_time
    xpred_path : UPath
        Path to future prediction dataset (interpolated GCM) chunked full_time
    run_parameters : RunParameters
        Parameters for run set-up and model specs
    dim : str, optional
        Dimension to apply the model along. Default is ``time``.

    Returns
    -------
    UPath
        Path to output dataset chunked full_time
    """
    # TODO: turn this into a hash
    ds_name = (
        f'train_obs_{xtrain_path.name}/train_gcm_{ytrain_path.name}/prediction_{xpred_path.name}'
    )
    # TODO: swap this in once we pull hash naming PR
    # ds_hash = str_to_hash()

    target = intermediate_dir / 'gard_fit_and_predict' / ds_name

    if use_cache and zmetadata_exists(target):
        print(f'found existing target: {target}')
        return target

    kws = default_none_kwargs(run_parameters.bc_kwargs, copy=True)

    # load in datasets
    ytrain = xr.open_zarr(ytrain_path)
    xtrain = xr.open_zarr(xtrain_path)
    xpred = xr.open_zarr(xpred_path)

    # make sure you have the variables you need in obs
    for v in xpred.data_vars:
        assert v in ytrain.data_vars

    # data transformation (this wants full-time chunking)
    # transformed_obs is for the training period
    transformed_obs = bias_correct_obs_by_method(
        da_obs=ytrain, method=run_parameters.bias_correction_method, bc_kwargs=kws
    ).to_dataset(dim="variable")

    # transformed_gcm is for the prediction period
    transformed_gcm = bias_correct_gcm_by_method(
        gcm_train=xtrain,
        obs_train=ytrain,
        gcm_predict=xpred,
        method=run_parameters.bias_correction_method,
        bc_kwargs=kws,
    ).to_dataset(dim="variable")

    # model definition
    model = PointWiseDownscaler(
        model=get_gard_model(run_parameters.model_type, run_parameters.model_params), dim=dim
    )

    # model fitting
    model.fit(
        transformed_gcm.sel(time=run_parameters.train_period.time_slice),
        transformed_obs[run_parameters.variable],
    )

    # model prediction
    out = model.predict(transformed_gcm).to_dataset(dim='variable')
    out.to_zarr(target)
    return out


@task(tags=['dask-resource:TASKSLOTS=1'], log_stdout=True)
def postprocess(
    model_output_path: xr.Dataset,
    run_parameters: RunParameters,
    **kwargs,
) -> xr.Dataset:
    """
    Add perturbation to the mean prediction of GARD to more accurately represent extreme events. The perturbation is
    generated with the prediction error during model fit scaled with a spatio-temporally correlated random field.

    Parameters
    ----------
    model_output : xr.Dataset
        GARD model prediction output. Should contain three variables: pred (predicted mean), prediction_error
        (prediction error in fit), and exceedance_prob (probability of exceedance for threshold)
    scrf : xr.DataArray
        Spatio-temporally correlated random fields (SCRF)
    model_params : Dict
        Model parameter dictionary

    Returns
    -------
    downscaled : xr.Dataset
        Final downscaled output
    """
    # TODO: turn this into a hash
    ds_name = 'gard_daily_output'
    # TODO: swap this in once we pull hash naming PR
    # ds_hash = str_to_hash()

    target = results_dir / 'daily' / ds_name

    if use_cache and zmetadata_exists(target):
        print(f'found existing target: {target}')
        return target

    model_output = xr.open_zarr(model_output_path)

    if run_parameters.model_params is not None:
        thresh = run_parameters.model_params.get('thresh')
    else:
        thresh = None

    # read scrf
    scrf = read_scrf(run_parameters)

    ## CURRENTLY needs calendar to be gregorian
    ## TODO: merge in the calendar conversion for GCMs and this should work great!
    assert len(scrf.time) == len(model_output.time)
    assert len(scrf.lat) == len(model_output.lat)
    assert len(scrf.lon) == len(model_output.lon)

    scrf = scrf.assign_coords(
        {'lat': model_output.lat, 'lon': model_output.lon, 'time': model_output.time}
    )

    if thresh is not None:
        # convert scrf from a normal distribution to a uniform distribution
        scrf_uniform = xr.apply_ufunc(
            norm.cdf, scrf, dask='parallelized', output_dtypes=[scrf.dtype]
        )

        # find where exceedance prob is exceeded
        mask = scrf_uniform > (1 - model_output['exceedance_prob'])

        # Rescale the uniform distribution
        new_uniform = (scrf_uniform - (1 - model_output['exceedance_prob'])) / model_output[
            'exceedance_prob'
        ]

        # Get the normal distribution equivalent of new_uniform
        r_normal = xr.apply_ufunc(
            norm.ppf, new_uniform, dask='parallelized', output_dtypes=[new_uniform.dtype]
        )

        downscaled = model_output['pred'] + r_normal * model_output['prediction_error']

        # what do we do for thresholds like heat wave?
        valids = xr.ufuncs.logical_or(mask, downscaled >= 0)
        downscaled = downscaled.where(valids, 0)
    else:
        downscaled = model_output['pred'] + scrf * model_output['prediction_error']
    # downscaled = downscaled.chunk({'time': 365, 'lat': 150, 'lon': 150})
    downscaled.to_dataset(name=run_parameters.variable).to_zarr(target)

    return target
