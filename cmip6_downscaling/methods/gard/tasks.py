from dataclasses import asdict

import dask
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import zarr
from carbonplan_data.metadata import get_cf_global_attrs
from prefect import task
from skdownscale.pointwise_models import PointWiseDownscaler
from skdownscale.pointwise_models.utils import default_none_kwargs
from upath import UPath

from cmip6_downscaling import __version__ as version, config

from ..common.bias_correction import bias_correct_gcm_by_method
from ..common.containers import RunParameters, str_to_hash
from ..common.utils import apply_land_mask, zmetadata_exists
from .utils import add_random_effects, get_gard_model

scratch_dir = UPath(config.get("storage.scratch.uri"))
intermediate_dir = UPath(config.get("storage.intermediate.uri")) / version
results_dir = UPath(config.get("storage.results.uri")) / version
use_cache = config.get('run_options.use_cache')


@task(log_stdout=True)
def coarsen_and_interpolate(fine_path: UPath, coarse_path: UPath) -> UPath:
    """Coarsen up obs and then interpolate it back to the original finescale grid.

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
    ds_hash = str_to_hash(str(fine_path) + str(coarse_path))
    target = intermediate_dir / 'coarsen_and_interpolate' / ds_hash

    if use_cache and zmetadata_exists(target):
        print(f'found existing target: {target}')
        interpolated_ds = xr.open_zarr(target)
        return target

    fine_ds = xr.open_zarr(fine_path)
    target_ds = xr.open_zarr(coarse_path)

    # coarsen
    regridder = xe.Regridder(fine_ds, target_ds, "bilinear", extrap_method="nearest_s2d")
    coarse_ds = regridder(fine_ds)

    # interpolate back to the fine grid
    regridder = xe.Regridder(coarse_ds, fine_ds, "bilinear", extrap_method="nearest_s2d")
    interpolated_ds = regridder(coarse_ds, keep_attrs=True)

    interpolated_ds.attrs.update(
        {'title': 'coarsen_and_interpolate'}, **get_cf_global_attrs(version=version)
    )
    interpolated_ds.to_zarr(target, mode='w')
    return target


def _fit_and_predict_wrapper(xtrain, ytrain, xpred, scrf, run_parameters, dim='time'):

    xpred = xpred.rename({'t2': 'time'})
    scrf = scrf.rename({'t2': 'time'})
    kws = default_none_kwargs(run_parameters.bias_correction_kwargs, copy=True)

    # transformed gcm is the interpolated GCM for the prediction period transformed
    # w.r.t. the interpolated obs used in the training (because that transformation
    # is essentially part of the model)
    bias_corrected_gcm_pred = (
        bias_correct_gcm_by_method(
            gcm_pred=xpred[run_parameters.variable],
            method=run_parameters.bias_correction_method,
            bc_kwargs=kws,
            obs=xtrain[run_parameters.variable],
        )
        .to_dataset(dim='variable')
        .rename({'variable_0': run_parameters.variable})
    )

    # model definition
    model = PointWiseDownscaler(
        model=get_gard_model(run_parameters.model_type, run_parameters.model_params), dim=dim
    )

    # model fitting
    model.fit(xtrain[run_parameters.variable], ytrain[run_parameters.variable])

    # model prediction
    out = model.predict(bias_corrected_gcm_pred[run_parameters.variable]).to_dataset(dim='variable')

    downscaled = add_random_effects(out, scrf, run_parameters)

    return downscaled


@task(log_stdout=True)
def fit_and_predict(
    xtrain_path: UPath,
    ytrain_path: UPath,
    xpred_path: UPath,
    scrf_path: UPath,
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
    xhist_path: UPath
        Path to historical prediction dataset (interpolated GCM)
    xpred_path : UPath
        Path to future prediction dataset (interpolated GCM) chunked full_time
    scrf_path : UPath
        Path to scrf chunked in full_time
    run_parameters : RunParameters
        Parameters for run set-up and model specs
    dim : str, optional
        Dimension to apply the model along. Default is ``time``.

    Returns
    -------
    path : UPath
        Path to output dataset chunked full_time
    """
    ds_hash = str_to_hash(
        str(xtrain_path)
        + str(ytrain_path)
        + str(xpred_path)
        + str(scrf_path)
        + run_parameters.run_id_hash
        + str(dim)
    )
    target = intermediate_dir / 'gard_fit_and_predict' / ds_hash

    if use_cache and zmetadata_exists(target):
        print(f'found existing target: {target}')
        return target

    # load in datasets
    xtrain = xr.open_zarr(xtrain_path).pipe(apply_land_mask)
    ytrain = xr.open_zarr(ytrain_path).pipe(apply_land_mask)
    xpred = xr.open_zarr(xpred_path).pipe(apply_land_mask)
    scrf = xr.open_zarr(scrf_path).pipe(apply_land_mask)
    # make sure you have the variables you need in obs
    for v in xpred.data_vars:
        assert v in ytrain.data_vars
    # data transformation (this wants full-time chunking)
    # transformed_obs is for the training period

    # we need only the prediction GCM (xpred), but we'll transform it into the space of the
    # transformed interpolated obs (xtrain)
    # Create a template dataset for map blocks
    # This feals a bit fragile.
    template_var = list(xpred.data_vars.keys())[0]
    template_da = xpred[template_var]
    template = xr.Dataset()
    for var in [run_parameters.variable]:
        template[var] = template_da

    # rename time variable to play nice with mapblocks - can't have same dimension name on later arguments
    out = xr.map_blocks(
        _fit_and_predict_wrapper,
        xtrain,
        args=(ytrain, xpred.rename({'time': 't2'}), scrf.rename({'time': 't2'}), run_parameters),
        kwargs={'dim': dim},
        template=template,
    )
    out.attrs.update({'title': 'gard_fit_and_predict'}, **get_cf_global_attrs(version=version))
    out = dask.optimize(out)[0]
    # remove apply_land_mask after scikit-downscale#110 is merged
    t = out.pipe(apply_land_mask).to_zarr(target, compute=False, mode='w', consolidated=False)
    t.compute(retries=5)

    zarr.consolidate_metadata(target)
    return target


@task(log_stdout=True)
def read_scrf(prediction_path: UPath, run_parameters: RunParameters):
    """
    Read spatial-temporally correlated random fields on file and subset into the correct spatial/temporal domain according to model_output.
    The random fields are stored in decade (10 year) long time series for the global domain and pre-generated using `scrf.ipynb`.

    Parameters
    ----------
    prediction_path : UPath
        Path to prediction dataset
    run_parameters : RunParameters
        Parameters for run set-up and model specs


    Returns
    -------
    scrf : xr.DataArray
        Spatio-temporally correlated random fields (SCRF)
    """
    # TODO: this is a temporary creation of random fields. ultimately we probably want to have
    # ~150 years of random fields, but this is fine.

    ds_hash = str_to_hash(
        "{obs}_{variable}_{latmin}_{latmax}_{lonmin}_{lonmax}_{predict_dates[0]}_{predict_dates[1]}".format(
            **asdict(run_parameters)
        )
    )

    target = intermediate_dir / 'scrf' / ds_hash

    if use_cache and zmetadata_exists(target):
        print(f'found existing target: {target}')
        return target
    prediction_ds = xr.open_zarr(prediction_path)
    scrf_ten_years = xr.open_zarr(f'az://static/scrf/ERA5_{run_parameters.variable}_1981_1990.zarr')

    scrf_list = []
    for year in np.arange(
        int(run_parameters.predict_period.start), int(run_parameters.predict_period.stop) + 10, 10
    ):
        scrf_list.append(scrf_ten_years.drop('time'))
    scrf = xr.concat(scrf_list, dim='time')
    scrf['time'] = pd.date_range(
        start=f'{run_parameters.predict_period.start}-01-01', periods=scrf.dims['time']
    )

    scrf = scrf.sel(time=run_parameters.predict_period.time_slice)

    scrf = scrf.drop('spatial_ref').astype('float32')

    scrf = scrf.sel(
        lat=prediction_ds.lat.values, lon=prediction_ds.lon.values, time=prediction_ds.time.values
    )
    assert len(scrf.time) == len(prediction_ds.time)
    assert len(scrf.lat) == len(prediction_ds.lat)
    assert len(scrf.lon) == len(prediction_ds.lon)

    scrf = scrf.assign_coords(
        {'lat': prediction_ds.lat, 'lon': prediction_ds.lon, 'time': prediction_ds.time}
    )

    scrf = dask.optimize(scrf)[0]
    t = scrf.to_zarr(target, compute=False, mode='w', consolidated=False)
    t.compute(retries=2)
    zarr.consolidate_metadata(target)

    return target
