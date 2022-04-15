import dask
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

from cmip6_downscaling import __version__ as version, config

from ..common.bias_correction import bias_correct_gcm_by_method, bias_correct_obs_by_method
from ..common.containers import RunParameters, str_to_hash
from ..common.utils import zmetadata_exists
from .utils import get_gard_model, read_scrf

scratch_dir = UPath(config.get("storage.scratch.uri"))
intermediate_dir = UPath(config.get("storage.intermediate.uri")) / version
results_dir = UPath(config.get("storage.results.uri")) / version
use_cache = config.get('run_options.use_cache')

good_fit_predict_chunks = {'lat': 24, 'lon': 24, 'time': 10957}


@task(tags=['dask-resource:taskslots=1'], log_stdout=True)
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
    ds_hash = str_to_hash(str(fine_path) + str(coarse_path))
    target = intermediate_dir / 'coarsen_and_interpolate' / ds_hash

    if use_cache and zmetadata_exists(target):
        print(f'found existing target: {target}')
        interpolated_ds = xr.open_zarr(target)
        print('the interpolated chunks look like this')
        print(interpolated_ds.chunks)
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
    print('the interpolated chunks look like this')
    print(interpolated_ds.chunks)
    return target


def _fit_and_predict_wrapper(xtrain, ytrain, xpred, run_parameters, dim='time'):

    xpred = xpred.rename({'t2': 'time'})

    kws = default_none_kwargs(run_parameters['bias_correction_kwargs'], copy=True)

    # data transformation (this wants full-time chunking)
    # transformed_obs is for the training period
    transformed_obs = bias_correct_obs_by_method(
        da_obs=ytrain, method=run_parameters['bias_correction_method'], bc_kwargs=kws
    ).to_dataset(dim="variable")

    # we need two transformed gcms - one for training and one for prediction
    # for transformed gcm_train we pass the same thing as the training and the
    # prediction since we're just transforming it
    transformed_gcm_train = bias_correct_gcm_by_method(
        gcm_train=xtrain,
        obs_train=ytrain,
        gcm_predict=xtrain,
        method=run_parameters['bias_correction_method'],
        bc_kwargs=kws,
    ).to_dataset(dim="variable")

    # for transformed_gcm_pred we pass the gcm train and then also the gcm_pred
    # to transform the gcm_pred
    transformed_gcm_pred = bias_correct_gcm_by_method(
        gcm_train=xtrain,
        obs_train=ytrain,
        gcm_predict=xpred,
        method=run_parameters['bias_correction_method'],
        bc_kwargs=kws,
    ).to_dataset(dim="variable")

    # model definition
    model = PointWiseDownscaler(
        model=get_gard_model(run_parameters['model_type'], run_parameters['model_params']), dim=dim
    )

    # model fitting
    model.fit(
        transformed_gcm_train.assign_coords({"time": transformed_obs.time.values})[
            run_parameters['variable']
        ],
        transformed_obs[run_parameters['variable']],
    )

    # model prediction
    out = model.predict(transformed_gcm_pred[run_parameters['variable']]).to_dataset(dim='variable')

    # t = out.to_zarr(
    #                 data_mapper,
    #                 mode='a',
    #                 region=region, #{'lat': index_locations_based_on_xpred
    #                                 # 'lon' index_locations_based_on_xpred
    #                                 #  'time': slice(0,len(time))}
    #                 compute=True,
    # t.compute()
    return out


@task(log_stdout=True)
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
    ds_hash = str_to_hash(
        str(xtrain_path)
        + str(ytrain_path)
        + str(xpred_path)
        + run_parameters.run_id_hash
        + str(dim)
    )
    target = intermediate_dir / 'gard_fit_and_predict' / ds_hash

    if use_cache and zmetadata_exists(target):
        print(f'found existing target: {target}')
        return target

    print(f'xtrain dataset is located here {xtrain_path}')
    print(f'ytrain dataset is located here {ytrain_path}')
    print(f'xpred dataset is located here {xpred_path}')
    # load in datasets
    xtrain = xr.open_zarr(xtrain_path)  # .isel(lon=slice(0, 48*6), lat=slice(0, 48*6))
    ytrain = xr.open_zarr(ytrain_path)  # .isel(lon=slice(0, 48*6), lat=slice(0, 48*6))
    xpred = xr.open_zarr(xpred_path)  # .isel(lon=slice(0, 48*6), lat=slice(0, 48*6))

    # make sure you have the variables you need in obs
    for v in xpred.data_vars:
        assert v in ytrain.data_vars

    # _fit_wrapper(xtrain, ytrain)
    # data transformation (this wants full-time chunking)
    # transformed_obs is for the training period

    # we need two transformed gcms - one for training and one for prediction
    # for transformed gcm_train we pass the same thing as the training and the
    # prediction since we're just transforming it
    # Create a template dataset for map blocks
    # This feals a bit fragile.
    template_var = list(xpred.data_vars.keys())[0]
    # .to_dataarray(dim='variable')
    template_da = xpred[template_var]
    template = xr.Dataset()
    for var in ['pred', 'exceedance_prob', 'prediction_error']:
        template[var] = template_da

    out = xr.map_blocks(
        _fit_and_predict_wrapper,
        xtrain,
        args=(ytrain, xpred.rename({'time': 't2'}), run_parameters),
        kwargs={'dim': dim},
        template=template,
    )

    out.attrs.update({'title': 'gard_fit_and_predict'}, **get_cf_global_attrs(version=version))
    out = dask.optimize(out)[0]
    # out = wait(out.persist()) # this was great
    t = out.to_zarr(target, compute=False, mode='w')
    t.compute(retries=5)
    return target


@task(log_stdout=True)
def postprocess(
    model_output_path: UPath,
    run_parameters: RunParameters,
) -> UPath:
    """
    Add perturbation to the mean prediction of GARD to more accurately represent extreme events. The perturbation is
    generated with the prediction error during model fit scaled with a spatio-temporally correlated random field.

    Parameters
    ----------
    model_output_path : UPath
        GARD model prediction output. Should contain three variables: pred (predicted mean), prediction_error
        (prediction error in fit), and exceedance_prob (probability of exceedance for threshold)
    run_parameters : RunParameters
        Model parameter dictionary

    Returns
    -------
    downscaled : UPath
        Final downscaled output
    """
    ds_hash = str_to_hash(str(model_output_path) + run_parameters.run_id_hash)
    target = results_dir / 'daily' / ds_hash
    print(f'searching for dataset here: {model_output_path}')
    if use_cache and zmetadata_exists(target):
        print(f'found existing target: {target}')
        return target

    print(model_output_path)
    model_output = xr.open_zarr(model_output_path)

    if run_parameters.model_params is not None:
        thresh = run_parameters.model_params.get('thresh')
    else:
        thresh = None

    # read scrf
    scrf = read_scrf(run_parameters)  # maybe this belongs in cat?
    scrf = scrf.chunk(
        {
            'lat': model_output.chunks['lat'][0],
            'lon': model_output.chunks['lon'][0],
        }
    )
    ## CURRENTLY needs calendar to be gregorian
    ## TODO: merge in the calendar conversion for GCMs and this should work great!
    assert len(scrf.time) == len(model_output.time)
    assert len(scrf.lat) == len(model_output.lat)
    assert len(scrf.lon) == len(model_output.lon)

    scrf = scrf.assign_coords(
        {'lat': model_output.lat, 'lon': model_output.lon, 'time': model_output.time}
    )
    print(scrf)
    print(scrf.chunks)
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
        downscaled = model_output['pred'] + scrf['scrf'] * model_output['prediction_error']
    (dask.optimize(downscaled.to_dataset(name=run_parameters.variable))[0]).to_zarr(
        target, mode='w'
    )

    return target
