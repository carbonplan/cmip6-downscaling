from typing import List, Union

import xarray as xr
from skdownscale.pointwise_models import PointWiseDownscaler
from skdownscale.pointwise_models.bcsd import BcsdPrecipitation, BcsdTemperature

from cmip6_downscaling import config
from cmip6_downscaling.constants import ABSOLUTE_VARS, RELATIVE_VARS
from cmip6_downscaling.data.cmip import get_gcm, load_cmip
from cmip6_downscaling.data.observations import open_era5
from cmip6_downscaling.workflows.paths import (
    make_bias_corrected_path,
    make_coarse_obs_path,
    make_gcm_predict_subset_path,
    make_gcm_train_subset_path,
    make_interpolated_obs_path,
    make_interpolated_prediction_path_full_space,
    make_interpolated_prediction_path_full_time,
    make_return_obs_path,
    make_spatial_anomalies_path,
)
from cmip6_downscaling.workflows.utils import (
    BBox,
    lon_to_180,
    rechunk_zarr_array_with_caching,
    regrid_ds,
    subset_dataset,
)


def return_obs(
    obs: str,
    variable: Union[str, List[str]],
    train_period: slice,
    bbox: BBox,
    obs_identifier: str,
    **kwargs
) -> xr.Dataset:
    """Loads ERA5 observation data for given time bounds and variable

    Parameters
    ----------
    obs : str
        Input obs
    variable: str
        The variable included in the dataset.
    train_period: slice
        Start and end year slice of training/historical period. Ex: slice('1990','1990')
    predict_period: slice
        Start and end year slice of predict period. Ex: slice('2020','2020')
    bbox : BBox
        Bounding box including latmin,latmax,lonmin,lonmax.
    **kwargs : dict, optional
            Other arguments to be used in generating the target path

    Returns
    -------
    xr.Dataset
        Loaded xarray dataset of ERA5 observation data. Chunked in time: 365
    """
    if isinstance(variable, str):
        variable = [variable]

    obs_load = open_era5(variable, train_period)
    obs_load_180 = lon_to_180(obs_load)
    obs_ds = subset_dataset(
        obs_load_180,
        variable[0],
        train_period,
        bbox,
        chunking_schema={'time': 365, 'lat': 150, 'lon': 150},
    )
    del obs_ds[variable[0]].encoding['chunks']
    return obs_ds


def get_coarse_obs(
    gcm: str,
    scenario: str,
    variable: Union[str, List[str]],
    train_period: slice,
    predict_period: slice,
    bbox: BBox,
    obs_identifier: str,
    gcm_grid_spec: str,
    chunking_approach: str,
    **kwargs
) -> xr.Dataset:
    """Regrids the observation dataset to match the GCM resolution

    Parameters
    ----------

    gcm : str
        Input GCM
    scenario: str
        Input GCM scenario
    variable: str
        The variable included in the dataset.
    train_period: slice
        Start and end year slice of training/historical period. Ex: slice('1990','1990')
    predict_period: slice
        Start and end year slice of predict period. Ex: slice('2020','2020')
    bbox : BBox
        Bounding box including latmin,latmax,lonmin,lonmax.

    **kwargs : dict, optional
            Other arguments to be used in generating the target path

    Returns
    -------
    xr.Dataset
        observation dataset at coarse resolution
    """
    # Load single slice of target cmip6 dataset for target grid dimensions
    # gcm_one_slice = load_cmip(return_type='xr', variable_ids=[variable]).isel(time=0)

    obs_ds_path = (
        config.get("storage.intermediate.uri")
        + '/'
        + make_return_obs_path(obs_identifier=obs_identifier)
    )
    if isinstance(variable, str):
        variable = [variable]

    gcm_ds = load_cmip(source_ids=gcm, return_type='xr', variable_ids=variable)

    gcm_ds_180 = lon_to_180(gcm_ds)
    gcm_subset = subset_dataset(gcm_ds_180, variable[0], train_period, bbox)

    # rechunk and regrid observation dataset to target gcm resolution
    coarse_obs_ds = regrid_ds(ds_path=obs_ds_path, target_grid_ds=gcm_subset)

    return coarse_obs_ds


def get_interpolated_obs(
    target_grid_ds: xr.Dataset, gcm_grid_spec: str, chunking_approach: str, obs_identifier: str
):

    coarse_obs_ds_path = (
        config.get("storage.intermediate.uri")
        + '/'
        + make_return_obs_path(obs_identifier=obs_identifier)
    )
    interpolated_obs = regrid_ds(ds_path=coarse_obs_ds_path, target_grid_ds=target_grid_ds)
    return interpolated_obs


def get_spatial_anomalies(
    gcm: str,
    scenario: str,
    variable: str,
    train_period: slice,
    predict_period: slice,
    bbox: BBox,
    obs_identifier: str,
    gcm_grid_spec: str,
    chunking_approach: str,
    **kwargs
) -> xr.Dataset:

    """Returns spatial anomalies
    Calculate the seasonal cycle (12 timesteps) spatial anomaly associated
    with aggregating the fine_obs to a given coarsened scale and then reinterpolating
    it back to the original spatial resolution. The outputs of this function are
    dependent on three parameters:
    * a grid (as opposed to a specific GCM since some GCMs run on the same grid)
    * the time period which fine_obs (and by construct coarse_obs) cover
    * the variable
    We will save these anomalies to use them in the post-processing. We will add them to the
    spatially-interpolated coarse predictions to add the spatial heterogeneity back in.
    Conceptually, this step figures out, for example, how much colder a finer-scale pixel
    containing Mt. Rainier is compared to the coarse pixel where it exists. By saving those anomalies,
    we can then preserve the fact that "Mt Rainier is x degrees colder than the pixels around it"
    for the prediction. It is important to note that that spatial anomaly is the same for every month of the
    year and the same for every day. So, if in January a finescale pixel was on average 4 degrees colder than
    the neighboring pixel to the west, in every day in the prediction (historic or future) that pixel
    will also be 4 degrees colder.

    Parameters
    ----------
    gcm : str
        Input GCM
    scenario: str
        Input GCM scenario
    variable: str
        The variable included in the dataset.
    train_period: slice
        Start and end year slice of training/historical period. Ex: slice('1990','1990')
    predict_period: slice
        Start and end year slice of predict period. Ex: slice('2020','2020')
    bbox : BBox
        Bounding box including latmin,latmax,lonmin,lonmax.
    **kwargs : dict, optional
            Other arguments to be used in generating the target path

    Returns
    -------
    seasonal_cycle_spatial_anomalies : xr.Dataset
        Spatial anomaly for each month (i.e. of shape (nlat, nlon, 12))
    """

    interpolated_obs_path = (
        config.get("storage.intermediate.uri")
        + '/'
        + make_interpolated_obs_path(
            gcm_grid_spec=gcm_grid_spec,
            chunking_approach=chunking_approach,
            obs_identifier=obs_identifier,
        )
    )
    # take interpolated obs and rechunk into full_time -- returns path
    coarse_obs_interpolated_rechunked_path = rechunk_zarr_array_with_caching(
        interpolated_obs_path, chunking_approach='full_time', max_mem='2GB'
    )

    obs_path = (
        config.get("storage.intermediate.uri")
        + '/'
        + make_return_obs_path(obs_identifier=obs_identifier)
    )
    # original obs rechunked into full-time
    obs_rechunked_path = rechunk_zarr_array_with_caching(
        obs_path, chunking_approach='full_time', max_mem='2GB'
    )
    coarse_obs_interpolated_rechunked_ds = xr.open_zarr(coarse_obs_interpolated_rechunked_path)
    obs_rechunked_ds = xr.open_zarr(obs_rechunked_path)

    # calculate the difference between the actual obs (with finer spatial heterogeneity)
    # and the interpolated coarse obs this will be saved and added to the
    # spatially-interpolated coarse predictions to add the spatial heterogeneity back in.

    spatial_anomalies = obs_rechunked_ds - coarse_obs_interpolated_rechunked_ds
    seasonal_cycle_spatial_anomalies = spatial_anomalies.groupby("time.month").mean()

    return seasonal_cycle_spatial_anomalies


def return_coarse_obs_full_time(
    gcm: str,
    scenario: str,
    variable: str,
    train_period: slice,
    predict_period: slice,
    bbox: BBox,
    obs_identifier: str,
    gcm_grid_spec: str,
    **kwargs
) -> xr.Dataset:

    """

    Return coarse observation dataset that has been chunked in time.

    Parameters
    ----------
    gcm : str
        Input GCM
    scenario: str
        Input GCM scenario
    variable: str
        The variable included in the dataset.
    train_period: slice
        Start and end year slice of training/historical period. Ex: slice('1990','1990')
    predict_period: slice
        Start and end year slice of predict period. Ex: slice('2020','2020')
    bbox : BBox
        Bounding box including latmin,latmax,lonmin,lonmax.
    **kwargs : dict, optional

    Returns
    -------
    xr.Dataset
        coarse_obs_full_time_ds rechunked dataset
    """
    coarse_obs_path_full_space_path = (
        config.get("storage.intermediate.uri")
        + '/'
        + make_coarse_obs_path(
            obs_identifier=obs_identifier,
            gcm_grid_spec=gcm_grid_spec,
            chunking_approach='full_space',
        )
    )
    # this is temp_path: az://flow-outputs/temporary/djnqgjlkbr.zarr. this is target_path: az://flow-outputs/test_intermediate_zarr/az://flow-outputs/test_intermediate_zarr/coarsened_obs/ERA5/tasmax/-90.0_90.0_-180.0_180.0/1990_1990/full_time_128x256_gridsize_14_14_llcorner_-88_-180.zarr
    #    az://flow-outputs/test_intermediate_zarr/coarsened_obs/ERA5/tasmax/-90.0_90.0_-180.0_180.0/1990_1990/full_time_128x256_gridsize_14_14_llcorner_-88_-180.zarr
    coarse_obs_path_full_time_path = make_coarse_obs_path(
        obs_identifier=obs_identifier, gcm_grid_spec=gcm_grid_spec, chunking_approach='full_time'
    )

    coarse_obs_full_time_path = rechunk_zarr_array_with_caching(
        coarse_obs_path_full_space_path,
        output_path=coarse_obs_path_full_time_path,
        chunking_approach='full_time',
        max_mem='2GB',
    )
    return coarse_obs_full_time_path


def return_gcm_train_full_time(
    coarse_obs_full_time_path: str,
    gcm: str,
    scenario: str,
    variable: str,
    train_period: slice,
    predict_period: slice,
    bbox: BBox,
    gcm_identifier: str,
    **kwargs
) -> xr.Dataset:
    """Returns GCM training rechunked dataset in full time.

    Parameters
    ----------
    coarse_obs_full_time_path : str
        Output coarse observation dataset path rechunked in full_time
    gcm : str
        Input GCM
    scenario: str
        Input GCM scenario
    variable: str
        The variable included in the dataset.
    train_period: slice
        Start and end year slice of training/historical period. Ex: slice('1990','1990')
    predict_period: slice
        Start and end year slice of predict period. Ex: slice('2020','2020')
    bbox : BBox
        Bounding box including latmin,latmax,lonmin,lonmax.
    **kwargs : dict, optional

    Returns
    -------
    xr.Dataset
        x_train rechunked dataset in full time.
    """
    gcm_train_ds = load_cmip(source_ids=gcm, variable_ids=[variable], return_type='xr')
    gcm_train_ds_180 = lon_to_180(gcm_train_ds)

    gcm_train_ds_subset = subset_dataset(
        gcm_train_ds_180,
        variable,
        train_period,
        bbox,
        # is the chunking schema needed?
        chunking_schema={'time': 365, 'lat': 150, 'lon': 150},
    )
    del gcm_train_ds_subset[variable].encoding['chunks']

    # this call was to force the timestamps for the cmip data to use the friendlier era5 timestamps. (i forget which dataset used which time formats). i could picture this introducing a tricky bug though (for instance if gcm timestamp didn't align for some reason) so we could use another conversion system if that is better. Perhaps datetime equivilence test.
    coarse_obs_full_time_ds = xr.open_zarr(coarse_obs_full_time_path)
    gcm_train_ds_subset['time'] = coarse_obs_full_time_ds.time.values
    gcm_train_subset_path = (
        config.get("storage.intermediate.uri")
        + '/'
        + make_gcm_train_subset_path(gcm_identifier=gcm_identifier)
    )
    gcm_train_ds_subset.to_zarr(gcm_train_subset_path, mode='w')

    gcm_train_subset_full_time_path = rechunk_zarr_array_with_caching(
        gcm_train_subset_path,
        chunking_approach='full_time',
        max_mem='2GB',
    )
    gcm_train_ds_subset = xr.open_zarr(gcm_train_subset_full_time_path)

    return gcm_train_ds_subset


def return_gcm_predict_rechunked(
    gcm_train_subset_full_time_ds: xr.Dataset,
    gcm: str,
    scenario: str,
    variable: str,
    train_period: slice,
    predict_period: slice,
    bbox: BBox,
    gcm_identifier: str,
    **kwargs
) -> xr.Dataset:
    """Returns GCM prediction rechunked dataset in full time.  Chunks are matched to chunks of gcm train. In the current use case, this means in full_time.

    Parameters
    ----------
    gcm_train_subset_full_time_ds : xr.Dataset
        Input gcm training rechunked dataset
    gcm : str
        Input GCM
    scenario: str
        Input GCM scenario
    variable: str
        The variable included in the dataset.
    train_period: slice
        Start and end year slice of training/historical period. Ex: slice('1990','1990')
    predict_period: slice
        Start and end year slice of predict period. Ex: slice('2020','2020')
    bbox : BBox
        Bounding box including latmin,latmax,lonmin,lonmax.
    **kwargs : dict, optional

    Returns
    -------
    xr.Dataset
        gcm predict rechunked dataset
    """
    gcm_predict_ds = get_gcm(
        gcm=gcm,
        scenario=scenario,
        variables=[variable],
        train_period=train_period,
        predict_period=predict_period,
        bbox=bbox,
    )

    gcm_predict_ds_subset = subset_dataset(
        gcm_predict_ds,
        variable,
        predict_period,
        bbox,
    )

    del gcm_predict_ds_subset[variable].encoding['chunks']

    gcm_predict_ds_subset_path = (
        config.get("storage.intermediate.uri")
        + '/'
        + make_gcm_predict_subset_path(gcm_identifier=gcm_identifier)
    )
    # Note: Explain why this .chunk is needed!
    gcm_predict_ds_subset.chunk({'time': 1500}).to_zarr(gcm_predict_ds_subset_path, mode='w')

    gcm_predict_rechunked_path = rechunk_zarr_array_with_caching(
        gcm_predict_ds_subset_path,
        # note: predict dataset might need to match the spatial chunking of the training dataset -- time chunking can be differant.
        template_chunk_array=gcm_train_subset_full_time_ds,
        max_mem='2GB',
    )
    gcm_predict_rechunked_ds = xr.open_zarr(gcm_predict_rechunked_path)
    return gcm_predict_rechunked_ds


def fit_and_predict(
    gcm_train_subset_full_time_ds: xr.Dataset,
    gcm_predict_rechunked_ds: xr.Dataset,
    gcm: str,
    scenario: str,
    variable: str,
    train_period: slice,
    predict_period: slice,
    bbox: BBox,
    chunking_approach: str,
    obs_identifier: str,
    gcm_grid_spec: str,
    gcm_identifier: str,
    dim: str = "time",
    **kwargs
) -> xr.Dataset:
    """Fit bcsd model on prepared CMIP data with obs at corresponding spatial scale.
    Then predict for a set of CMIP data (likely future).

    Parameters
    ----------
    gcm_train_subset_full_time_ds : xr.Dataset
        GCM training dataset chunked along space
    gcm_predict_rechunked_ds : xr.Dataset
        GCM prediction dataset chunked along space.
    gcm : str
        Input GCM
    scenario: str
        Input GCM scenario
    variable: str
        The variable included in the dataset.
    train_period: slice
        Start and end year slice of training/historical period. Ex: slice('1990','1990')
    predict_period: slice
        Start and end year slice of predict period. Ex: slice('2020','2020')
    bbox : BBox
        Bounding box including latmin,latmax,lonmin,lonmax.
    dim : str, optional
        dimension on which you want to do the modelling, by default "time"
    **kwargs : dict, optional

    Returns
    -------
    bias_corrected_ds : xr.Dataset
        Bias-corrected dataset
    """
    if variable in ABSOLUTE_VARS:
        bcsd_model = BcsdTemperature(return_anoms=False)
    elif variable in RELATIVE_VARS:
        bcsd_model = BcsdPrecipitation(return_anoms=False)

    pointwise_model = PointWiseDownscaler(model=bcsd_model, dim=dim)
    coarse_obs_full_time_path = (
        config.get("storage.intermediate.uri")
        + '/'
        + make_coarse_obs_path(
            chunking_approach=chunking_approach,
            obs_identifier=obs_identifier,
            gcm_grid_spec=gcm_grid_spec,
        )
    )
    coarse_obs_rechunked_validated_path = rechunk_zarr_array_with_caching(
        coarse_obs_full_time_path, template_chunk_array=gcm_train_subset_full_time_ds
    )
    coarse_obs_rechunked_validated_ds = xr.open_zarr(coarse_obs_rechunked_validated_path)

    pointwise_model.fit(
        gcm_train_subset_full_time_ds[variable], coarse_obs_rechunked_validated_ds[variable]
    )
    bias_corrected_da = pointwise_model.predict(gcm_predict_rechunked_ds[variable])

    bias_corrected_ds = bias_corrected_da.astype('float32').to_dataset(name=variable)

    return bias_corrected_ds


def get_interpolated_prediction(
    target_grid_obs_ds: xr.Dataset, gcm_grid_spec: str, gcm_identifier: str
) -> xr.Dataset:

    bias_corrected_path = (
        config.get("storage.intermediate.uri")
        + '/'
        + make_bias_corrected_path(gcm_identifier=gcm_identifier)
    )

    interpolated_prediction_ds = regrid_ds(
        ds_path=bias_corrected_path, target_grid_ds=target_grid_obs_ds
    )
    return interpolated_prediction_ds


def rechunked_interpolated_prediciton_task_full_time(
    gcm_identifier: str, gcm_grid_spec: str, **kwargs
) -> str:

    interpolated_prediction_path = (
        config.get("storage.intermediate.uri")
        + '/'
        + make_interpolated_prediction_path_full_space(
            gcm_identifier=gcm_identifier, gcm_grid_spec=gcm_grid_spec
        )
    )
    rechunked_interpolated_prediction_path = make_interpolated_prediction_path_full_time(
        gcm_identifier=gcm_identifier, gcm_grid_spec=gcm_grid_spec
    )

    rechunked_interpolated_prediction_path = rechunk_zarr_array_with_caching(
        interpolated_prediction_path,
        output_path=rechunked_interpolated_prediction_path,
        chunking_approach='full_time',
        max_mem='2GB',
    )

    return rechunked_interpolated_prediction_path


def postprocess_bcsd(
    gcm: str,
    scenario: str,
    variable: str,
    train_period: slice,
    predict_period: slice,
    bbox: BBox,
    obs_identifier: str,
    gcm_identifier: str,
    gcm_grid_spec: str,
    **kwargs
) -> xr.Dataset:
    """Downscale the bias-corrected data by interpolating and then
    adding the spatial anomalies back in.

    Parameters
    ----------
    gcm : str
        Input GCM
    scenario: str
        Input GCM scenario
    variable: str
        The variable included in the dataset.
    train_period : slice
        Start and end year slice of training/historical period. Ex: slice('1990', '1990')
    predict_period : slice
        Start and end year slice of predict period. Ex: slice('2020', '2020')
    bbox : BBox
        Bounding box including latmin,latmax,lonmin,lonmax.
    **kwargs : dict, optional

    Returns
    -------
    bcsd_results_ds : xr.Dataset
        Final BCSD dataset
    """

    spatial_anomalies_path = (
        config.get("storage.intermediate.uri")
        + '/'
        + make_spatial_anomalies_path(obs_identifier=obs_identifier, gcm_grid_spec=gcm_grid_spec)
    )
    rechunked_interpolated_prediction_path = (
        config.get("storage.intermediate.uri")
        + '/'
        + make_interpolated_prediction_path_full_time(
            gcm_identifier=gcm_identifier, gcm_grid_spec=gcm_grid_spec
        )
    )

    spatial_anomalies_ds = xr.open_zarr(spatial_anomalies_path)
    rechunked_interpolated_prediction_ds = xr.open_zarr(rechunked_interpolated_prediction_path)

    bcsd_results_ds = (
        rechunked_interpolated_prediction_ds.groupby("time.month") + spatial_anomalies_ds
    )
    del bcsd_results_ds['month'].encoding['chunks']

    # failes without .chunk: #ValueError: Zarr requires uniform chunk sizes except for final chunk. Variable named 'tasmax' has incompatible dask chunks: ((31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), (93, 93, 93, 93, 93, 93, 93, 70), (93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 45)). Consider rechunking using `chunk()`.
    rechunked_bcsd_results_ds = bcsd_results_ds.chunk({'time': 365, 'lat': -1, 'lon': -1})

    return rechunked_bcsd_results_ds
