import functools
from dataclasses import asdict

import fsspec
import numpy as np
import tensorflow as tf
import xarray as xr
import zarr
from carbonplan_data.metadata import get_cf_global_attrs
from prefect import task
from upath import UPath

from ... import __version__ as version, config
from ...data.observations import open_era5
from ...data.utils import lon_to_180
from ..common.bias_correction import bias_correct_gcm_by_method
from ..common.containers import RunParameters, str_to_hash
from ..common.utils import (
    apply_land_mask,
    blocking_to_zarr,
    set_zarr_encoding,
    subset_dataset,
    validate_zarr_store,
)
from .utils import (
    EPSILON,
    INFERENCE_BATCH_SIZE,
    bilinear_interpolate,
    conservative_interpolate,
    get_elevation_data,
    initialize_empty_dataset,
    normalize,
    output_node_name,
    res_to_str,
    stacked_model_path,
    starting_resolutions,
)

scratch_dir = UPath(config.get("storage.scratch.uri"))
intermediate_dir = UPath(config.get("storage.intermediate.uri")) / version
results_dir = UPath(config.get("storage.results.uri")) / version
use_cache = config.get('run_options.use_cache')

xr.set_options(keep_attrs=True)

is_cached = functools.partial(validate_zarr_store, raise_on_error=False)


@task(log_stdout=True)
def shift(path: UPath, path_type: str, run_parameters: RunParameters) -> UPath:
    """Interpolate obs data to grid specs in ``xe.util.grid_global``.

    Parameters
    ----------
    path : UPath
        Path to original dataset.
    path_type : str
        Specify whether gcm or obs to set output resolution
    run_parameters : RunParameters
        Parameters for run set-up and model specs

    Returns
    -------
    UPath
        Path to shifted dataset.
    """
    if path_type == 'obs':
        output_degree = 0.25
    elif path_type == 'gcm':
        output_degree = starting_resolutions[run_parameters.model]
    else:
        raise ValueError('path_type must be gcm or obs')

    ds_hash = str_to_hash(str(path) + str(output_degree))
    target = intermediate_dir / 'shift' / ds_hash

    if use_cache and is_cached(target):
        print(f'found existing target: {target}')
        shifted_ds = xr.open_zarr(target)
        return target

    orig_ds = xr.open_zarr(path)

    # Note: Notebook included unit conversion at this step for precipitation, but this is done earlier in `load_cmip`
    shifted_ds = bilinear_interpolate(ds=orig_ds, output_degree=output_degree)
    shifted_ds.attrs.update({'title': 'shift'}, **get_cf_global_attrs(version=version))
    print(f'writing shifted dataset to {target}')
    shifted_ds = set_zarr_encoding(shifted_ds)
    blocking_to_zarr(ds=shifted_ds, target=target, validate=True, write_empty_chunks=True)
    return target


@task(log_stdout=True)
def coarsen_obs(path: UPath, output_degree: float) -> UPath:
    """Coarsen grid using conservative interpolation.

    Parameters
    ----------
    path : UPath
        Path to original (likely observational) dataset.
    output_degree : float
        Resolution to coarsen the dataset to.

    Returns
    -------
    UPath
        Path to coarsened dataset.
    """
    # Similar to coarsen_and_interpolate in GARD tasks (maybe could be combined?)
    ds_hash = str_to_hash(str(path) + str(output_degree))
    target = intermediate_dir / 'coarsen' / ds_hash

    if use_cache and is_cached(target):
        print(f'found existing target: {target}')
        coarse_ds = xr.open_zarr(target)
        return target

    orig_ds = xr.open_zarr(path)

    # Coarsen obs
    coarse_ds = conservative_interpolate(ds=orig_ds, output_degree=output_degree)
    coarse_ds.attrs.update({'title': 'coarsen_interpolate'}, **get_cf_global_attrs(version=version))
    print(f'writing coarsened dataset to {target}')
    coarse_ds = set_zarr_encoding(coarse_ds)
    blocking_to_zarr(ds=coarse_ds, target=target, validate=True, write_empty_chunks=True)
    return target


@task(log_stdout=True)
def coarsen_and_interpolate(path: UPath, output_degree: float) -> UPath:
    """Coarsen grid and interpolate back to twice the coarsened grid resolution.

    Parameters
    ----------
    path : UPath
        Path to original (likely observational) dataset.
    output_degree : float
        Resolution to coarsen the dataset to.

    Returns
    -------
    UPath
        Path to interpolated dataset with a resolution twice that specified by
        ``output_degre``.
    """
    # Similar to coarsen_and_interpolate in GARD tasks (maybe could be combined?)
    ds_hash = str_to_hash(str(path) + str(output_degree))
    target = intermediate_dir / 'coarsen_interpolate' / ds_hash

    if use_cache and is_cached(target):
        print(f'found existing target: {target}')
        interpolated_ds = xr.open_zarr(target)
        return target

    orig_ds = xr.open_zarr(path)

    # Coarsen obs
    coarse_ds = conservative_interpolate(ds=orig_ds, output_degree=output_degree)
    # Interpolate back to 2x higher resolution
    interpolated_ds = bilinear_interpolate(ds=coarse_ds, output_degree=output_degree / 2)
    interpolated_ds.attrs.update(
        {'title': 'coarsen_interpolate'}, **get_cf_global_attrs(version=version)
    )
    print(f'writing interpolated dataset to {target}')
    interpolated_ds = set_zarr_encoding(interpolated_ds)
    blocking_to_zarr(ds=interpolated_ds, target=target, validate=True, write_empty_chunks=True)
    return target


@task(log_stdout=True)
def rescale(source_path: UPath, obs_path: UPath, run_parameters: RunParameters) -> UPath:
    """Rescale GCM data that has been normalized based on data in obs_path.

    Parameters
    ----------
    source_path : UPath
        Path to normalized model output
    obs_path : UPath
        Path to original (likely observational) dataset to back transform based on
    run_parameters : RunParameters
        Parameters for run set-up and model specs
    Returns
    -------
    UPath
        Path to rescaled dataset
    """
    ds_hash = str_to_hash(str(source_path) + str(obs_path))
    target = results_dir / 'deepsd_rescale' / ds_hash

    if use_cache and is_cached(target):
        print(f'found existing target: {target}')
        rescaled_ds = xr.open_zarr(target)
        return target

    orig_ds = xr.open_zarr(source_path)
    obs_ds = xr.open_zarr(obs_path).sel(
        time=slice(run_parameters.train_dates[0], run_parameters.train_dates[1])
    )
    obs_mean = obs_ds.mean(dim='time')
    obs_std = obs_ds.std(dim='time')

    rescaled_ds = (orig_ds * (obs_std[run_parameters.variable] + EPSILON)) + obs_mean[
        run_parameters.variable
    ]
    # Clip negative precipitation values
    if run_parameters.variable == "pr":
        rescaled_ds = rescaled_ds.clip(min=0)
    rescaled_ds.attrs.update({'title': 'deepsd_output'}, **get_cf_global_attrs(version=version))
    print(f'writing rescaled dataset to {target}')
    rescaled_ds = rescaled_ds.pipe(apply_land_mask).pipe(set_zarr_encoding)
    blocking_to_zarr(ds=rescaled_ds, target=target, validate=True, write_empty_chunks=True)
    return target


@task(log_stdout=True)
def normalize_gcm(predict_path: UPath, historical_path: UPath) -> UPath:
    """Normalize gcm data based on historical data.

    Parameters
    ----------
    predict_path : UPath
        Path to dataset that will be normalized.
    historical_path : UPath
        Path to dataset to normalized based on.
    Returns
    -------
    UPath
        Path to normalized dataset.
    """
    # Create path for output file
    ds_hash = str_to_hash(str(predict_path) + str(historical_path))
    target = intermediate_dir / 'normalize' / ds_hash

    # Skip step if output file already exists when using cache
    if use_cache and is_cached(target):
        print(f'found existing target: {target}')
        norm_ds = xr.open_zarr(target)
        return target

    predict_ds = xr.open_zarr(predict_path)
    historical_ds = xr.open_zarr(historical_path)

    historical_ds_mean = historical_ds.mean(dim="time").compute()
    historical_ds_std = historical_ds.std(dim="time").compute()

    norm_ds = (predict_ds - historical_ds_mean) / (historical_ds_std + EPSILON)

    norm_ds = lon_to_180(norm_ds)
    norm_ds = norm_ds.chunk({'time': INFERENCE_BATCH_SIZE, 'lat': -1, 'lon': -1})

    norm_ds.attrs.update({'title': 'normalize'}, **get_cf_global_attrs(version=version))
    print(f'writing normalized predict dataset to {target}')
    norm_ds = set_zarr_encoding(norm_ds)
    blocking_to_zarr(ds=norm_ds, target=target, validate=True, write_empty_chunks=True)

    return target


@task(log_stdout=True)
def inference(gcm_path: UPath, run_parameters: RunParameters) -> UPath:
    """Run inference on normalized gcm data.

    Parameters
    ----------
    gcm_path : UPath
        Path to normalized dataset.
    run_parameters : RunParameters
        Parameters for run set-up and model specs.
    Returns
    -------
    UPath
        Path to dataset containing model predictions.
    """
    import tensorflow_io  # noqa

    # # Check that GPU is available
    # print(tf.config.list_physical_devices('GPU'))

    tf.compat.v1.disable_eager_execution()

    # Create path for output file
    ds_hash = str_to_hash(str(gcm_path))
    target = intermediate_dir / 'inference' / ds_hash

    # Skip step if output file already exists when using cache
    if use_cache and is_cached(target):
        print(f'found existing target: {target}')
        downscaled_batch = xr.open_zarr(target)
        return target

    # find all the output resolution for each SRCNN in the stacked model according to the starting resolution of the GCM of interest
    if starting_resolutions[run_parameters.model] == 2.0:
        output_resolutions = [0.25, 0.5, 1.0]
    elif starting_resolutions[run_parameters.model] == 1.0:
        output_resolutions = [0.25, 0.5]
    else:
        raise ValueError("needs to be either 2.0 or 1.0")

    # make sure this is from low res to high res
    output_resolutions = sorted(output_resolutions, reverse=True)

    # get elevations at all relevant resolutions
    elevs = []
    for output_res in output_resolutions:
        elev = get_elevation_data(output_res)
        elev_norm = normalize(ds=elev, dims=["lat", "lon"], epsilon=EPSILON).elevation.values
        elevs.append(tf.constant(elev_norm[np.newaxis, :, :, np.newaxis].astype(np.float32)))

    input_map = {"elev_%i" % i: elevs[i] for i in range(len(output_resolutions))}

    # now read in the frozen graph of the stacked model, set placeholder for x, constant for elevs
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None, 1))
    input_map["lr_x"] = x

    model_path = stacked_model_path.format(
        var=run_parameters.variable,
        starting_resolution=res_to_str(starting_resolutions[run_parameters.model]),
    )
    output_node = output_node_name.format(var=run_parameters.variable)

    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

        (y,) = tf.import_graph_def(
            graph_def,
            input_map=input_map,
            return_elements=[output_node],
            name="deepsd",
            op_dict=None,
            producer_op_list=None,
        )

    # Read the gcm model data

    gcm_norm = xr.open_zarr(gcm_path)
    attrs = gcm_norm[run_parameters.variable].attrs

    batch_size = INFERENCE_BATCH_SIZE
    n = len(gcm_norm.time.values)

    elev_hr = get_elevation_data(0.25)

    print("initializing")
    initialize_empty_dataset(
        lats=elev_hr.lat.values,
        lons=elev_hr.lon.values,
        times=gcm_norm.time.values,
        output_path=target,
        var=run_parameters.variable,
        chunks={'time': batch_size, 'lat': 48, 'lon': 48},
        attrs=attrs,
    )

    print("batching")
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        print(start, stop)

        X = gcm_norm.isel(time=slice(start, stop))[run_parameters.variable].values

        downscaled_batch = np.empty(
            shape=(stop - start, len(elev_hr.lat.values), len(elev_hr.lon.values))
        )
        with tf.compat.v1.Session() as sess:
            for i in range(X.shape[0]):
                _x = X[i][np.newaxis, :, :, np.newaxis]
                _y = sess.run(y, feed_dict={x: _x})
                downscaled_batch[i, :, :] = _y[0, :, :, 0]

        downscaled_batch = xr.DataArray(
            downscaled_batch,
            dims=["time", "lat", "lon"],
            coords=[
                gcm_norm.isel(time=slice(start, stop)).time.values,
                elev_hr.lat.values,
                elev_hr.lon.values,
            ],
        )

        region = {
            "lat": slice(0, len(elev_hr.lat.values)),
            "lon": slice(0, len(elev_hr.lon.values)),
            "time": slice(start, stop),
        }

        print("saving to zarr store")

        task = (
            downscaled_batch.to_dataset(name=run_parameters.variable)
            .chunk({'time': -1, 'lat': 48, 'lon': 48})
            .to_zarr(
                target,
                mode="a",
                region=region,
                compute=False,
            )
        )
        task.compute(retries=10)

    return target


@task(log_stdout=True)
def update_var_attrs(
    target_path: UPath, source_path: UPath, run_parameters: RunParameters
) -> UPath:
    """Update attrs for a DataArray in a zarr store.

    Parameters
    ----------
    target_path : UPath
        Store to add DataArray attrs to.
    source_path : UPath
        Store to get DataArray attrs from.
    run_parameters : RunParameters
        Parameters for run set-up and model specs.
    Returns
    -------
    UPath
        Path to dataset containing corrected attrs.
    """

    target_attrs = f'{target_path}/{run_parameters.variable}/.zattrs'
    source_attrs = f'{source_path}/{run_parameters.variable}/.zattrs'
    print(f'copying attrs from {source_attrs} to {target_attrs}')
    fs = fsspec.filesystem('az', account_name='cmip6downscaling')
    fs.copy(source_attrs, target_attrs)
    zarr.consolidate_metadata(target_path)
    return target_path


@task(log_stdout=True)
def bias_correction(
    downscaled_path: UPath, obs_path: UPath, run_parameters: RunParameters
) -> UPath:
    """Bias correct downscaled data product.

    Parameters
    ----------
    downscaled_path : UPath
        Path to downscaled dataset.
    obs_path : UPath
        Path to obs dataset to bias correct based on.
    run_parameters : RunParameters
        Parameters for run set-up and model specs.
    Returns
    -------
    UPath
        Path to dataset containing bias corrected model predictions.
    """
    # Create path for output file
    ds_hash = str_to_hash(str(downscaled_path) + str(obs_path))
    target = results_dir / 'deepsd_bias_correction' / ds_hash

    # Skip step if output file already exists when using cache
    if use_cache and is_cached(target):
        print(f'found existing target: {target}')
        bc_output = xr.open_zarr(target)
        return target
    # TO-DO: Retain attrs during bias correction
    obs_ds = xr.open_zarr(obs_path)
    obs_ds = apply_land_mask(obs_ds)
    downscaled_ds = xr.open_zarr(downscaled_path)
    bc_output = (
        bias_correct_gcm_by_method(
            gcm_pred=downscaled_ds[run_parameters.variable],
            method=run_parameters.bias_correction_method,
            bc_kwargs=run_parameters.bias_correction_kwargs,
            obs=obs_ds[run_parameters.variable],
        )
        .to_dataset(dim='variable')
        .rename({'variable_0': run_parameters.variable})
    )
    bc_output.attrs.update(
        {'title': 'deepsd_output_bias_corrected'}, **get_cf_global_attrs(version=version)
    )
    print(f'writing bias corrected dataset to {target}')
    bc_output = set_zarr_encoding(bc_output)
    blocking_to_zarr(ds=bc_output, target=target, validate=True, write_empty_chunks=True)
    return target


@task(log_stdout=True)
def get_validation(run_parameters: RunParameters) -> UPath:
    """Task to return observation data subset from input parameters.

    Parameters
    ----------
    run_parameters : RunParameters
        RunParameter dataclass defined in common/conatiners.py. Constructed from prefect parameters.

    Returns
    -------
    UPath
        Path to subset observation dataset.
    """

    title = "validation ds: {obs}_{variable}_{latmin}_{latmax}_{lonmin}_{lonmax}_{predict_dates[0]}_{predict_dates[1]}".format(
        **asdict(run_parameters)
    )
    ds_hash = str_to_hash(
        "{obs}_{variable}_{latmin}_{latmax}_{lonmin}_{lonmax}_{predict_dates[0]}_{predict_dates[1]}".format(
            **asdict(run_parameters)
        )
    )
    target = intermediate_dir / 'get_validation' / ds_hash

    if use_cache and is_cached(target):
        print(f'found existing target: {target}')
        return target

    ds = open_era5(run_parameters.variable, run_parameters.predict_period)

    subset = subset_dataset(
        ds,
        run_parameters.variable,
        run_parameters.predict_period.time_slice,
        run_parameters.bbox,
        chunking_schema={'time': 365, 'lat': 150, 'lon': 150},
    )

    for key in subset.variables:
        subset[key].encoding = {}

    subset.attrs.update({'title': title}, **get_cf_global_attrs(version=version))
    print(f'writing validation dataset to {target}', subset)
    store = subset.pipe(set_zarr_encoding).to_zarr(target, mode='w', compute=False)
    store.compute(retries=2)
    return target
