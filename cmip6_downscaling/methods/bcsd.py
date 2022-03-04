from typing import List, Union

import xarray as xr
from skdownscale.pointwise_models import PointWiseDownscaler
from skdownscale.pointwise_models.bcsd import BcsdPrecipitation, BcsdTemperature

from cmip6_downscaling.constants import ABSOLUTE_VARS, RELATIVE_VARS
from cmip6_downscaling.data.cmip import get_gcm, load_cmip
from cmip6_downscaling.data.observations import open_era5
from cmip6_downscaling.workflows.utils import (
    BBox,
    lon_to_180,
    rechunk_zarr_array_with_caching,
    regrid_ds,
    subset_dataset,
)


def return_obs(
    obs: str, variable: Union[str, List[str]], train_period: slice, bbox: BBox, **kwargs
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
    print('180 to 180')
    obs_load_180 = lon_to_180(obs_load)
    print('subset dataset')
    obs_ds = subset_dataset(
        obs_load_180,
        variable[0],
        train_period,
        bbox,
        chunking_schema={'time': 365, 'lat': 150, 'lon': 150},
    )
    return obs_ds


def get_coarse_obs(
    obs_ds: xr.Dataset,
    gcm: str,
    scenario: str,
    variable: Union[str, List[str]],
    train_period: slice,
    predict_period: slice,
    bbox: BBox,
    **kwargs
) -> xr.Dataset:
    """Regrids the observation dataset to match the GCM resolution

    Parameters
    ----------
    obs_ds : xr.Dataset
        Input observation dataset.
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

    if isinstance(variable, str):
        variable = [variable]

    gcm_ds = load_cmip(return_type='xr', variable_ids=variable)

    gcm_ds_180 = lon_to_180(gcm_ds)
    gcm_subset = subset_dataset(gcm_ds_180, variable[0], train_period, bbox)

    # rechunk and regrid observation dataset to target gcm resolution
    coarse_obs_ds = regrid_ds(ds=obs_ds, target_grid_ds=gcm_subset)

    return coarse_obs_ds


def get_spatial_anomalies(
    obs_ds: xr.Dataset,
    interpolated_obs_ds: xr.Dataset,
    gcm: str,
    scenario: str,
    variable: str,
    train_period: slice,
    predict_period: slice,
    bbox: BBox,
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
    obs_ds : xr.Dataset
        Input observation dataset.
    interpolated_obs_ds : xr.Dataset
        Regridded interpolated obs dataset
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

    coarse_obs_interpolated_rechunked = rechunk_zarr_array_with_caching(
        interpolated_obs_ds, chunking_approach='full_time', max_mem='8GB'
    )

    obs_rechunked = rechunk_zarr_array_with_caching(
        obs_ds, chunking_approach='full_time', max_mem='8GB'
    )

    # calculate the difference between the actual obs (with finer spatial heterogeneity)
    # and the interpolated coarse obs this will be saved and added to the
    # spatially-interpolated coarse predictions to add the spatial heterogeneity back in.

    spatial_anomalies = obs_rechunked - coarse_obs_interpolated_rechunked
    seasonal_cycle_spatial_anomalies = spatial_anomalies.groupby("time.month").mean()

    return seasonal_cycle_spatial_anomalies


def return_coarse_obs_full_time(
    coarse_obs_ds: xr.Dataset,
    gcm: str,
    scenario: str,
    variable: str,
    train_period: slice,
    predict_period: slice,
    bbox: BBox,
    **kwargs
) -> xr.Dataset:

    """

    Return coarse observation dataset that has been chunked in time.

    Parameters
    ----------
    coarse_obs_ds : xr.Dataset
        Input coarse observation dataset
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
    coarse_obs_full_time_ds = rechunk_zarr_array_with_caching(
        coarse_obs_ds,
        chunking_approach='full_time',
        max_mem='8GB',
    )
    return coarse_obs_full_time_ds


def return_gcm_train_full_time(
    coarse_obs_full_time_ds: xr.Dataset,
    gcm: str,
    scenario: str,
    variable: str,
    train_period: slice,
    predict_period: slice,
    bbox: BBox,
    **kwargs
) -> xr.Dataset:
    """Returns GCM training rechunked dataset in full time.

    Parameters
    ----------
    coarse_obs_full_time_ds : xr.Dataset
        Output coarse observation dataset rechunked in full_time
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
        # chunking_schema={'time': 365, 'lat': 150, 'lon': 150},
    )

    # this call was to force the timestamps for the cmip data to use the friendlier era5 timestamps. (i forget which dataset used which time formats). i could picture this introducing a tricky bug though (for instance if gcm timestamp didn't align for some reason) so we could use another conversion system if that is better. Perhaps datetime equivilence test.
    gcm_train_ds_subset['time'] = coarse_obs_full_time_ds.time.values

    gcm_train_subset_full_time_ds = rechunk_zarr_array_with_caching(
        gcm_train_ds_subset,
        chunking_approach='full_time',
        max_mem='8GB',
    )
    return gcm_train_subset_full_time_ds


def return_gcm_predict_rechunked(
    gcm_train_subset_full_time_ds: xr.Dataset,
    gcm: str,
    scenario: str,
    variable: str,
    train_period: slice,
    predict_period: slice,
    bbox: BBox,
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

    gcm_predict_rechunked_ds = rechunk_zarr_array_with_caching(
        gcm_predict_ds_subset,
        template_chunk_array=gcm_train_subset_full_time_ds,
        max_mem='8GB',
    )

    return gcm_predict_rechunked_ds


def fit_and_predict(
    gcm_train_subset_full_time_ds: xr.Dataset,
    coarse_obs_full_time_ds: xr.Dataset,
    gcm_predict_rechunked_ds: xr.Dataset,
    gcm: str,
    scenario: str,
    variable: str,
    train_period: slice,
    predict_period: slice,
    bbox: BBox,
    dim: str = "time",
    **kwargs
) -> xr.Dataset:
    """Fit bcsd model on prepared CMIP data with obs at corresponding spatial scale.
    Then predict for a set of CMIP data (likely future).

    Parameters
    ----------
    gcm_train_subset_full_time_ds : xr.Dataset
        GCM training dataset chunked along space
    coarse_obs_full_time_ds : xr.Dataset
        Obs training dataset chunked along space
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

    coarse_obs_rechunked_validated_ds = rechunk_zarr_array_with_caching(
        coarse_obs_full_time_ds, template_chunk_array=gcm_train_subset_full_time_ds
    )
    pointwise_model.fit(
        gcm_train_subset_full_time_ds[variable], coarse_obs_rechunked_validated_ds[variable]
    )
    bias_corrected_da = pointwise_model.predict(gcm_predict_rechunked_ds[variable])

    bias_corrected_ds = bias_corrected_da.astype('float32').to_dataset(name=variable)

    return bias_corrected_ds


def postprocess_bcsd(
    interpolated_prediction: xr.Dataset,
    spatial_anomalies_ds: xr.Dataset,
    gcm: str,
    scenario: str,
    variable: str,
    train_period: slice,
    predict_period: slice,
    bbox: BBox,
    **kwargs
) -> xr.Dataset:
    """Downscale the bias-corrected data by interpolating and then
    adding the spatial anomalies back in.

    Parameters
    ----------
    interpolated_prediction : xr.Dataset
        interpolated prediction dataset
    spatial_anomalies_ds : xr.Dataset
        spatial anomalies dataset
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

    # interpolated prediction is chunked in time
    """Dimensions:  (lat: 721, lon: 1440, time: 43464)
    Coordinates:
    * lat      (lat) float32 90.0 89.75 89.5 89.25 ... -89.25 -89.5 -89.75 -90.0
    * lon      (lon) float32 -180.0 -179.8 -179.5 -179.2 ... 179.2 179.5 179.8
    * time     (time) datetime64[ns] 1981-01-01T12:00:00 ... 2099-12-31T12:00:00
    Data variables:
        tasmax   (time, lat, lon) float32 dask.array<chunksize=(96, 721, 1440), meta=np.ndarray>
    Attributes:
        regrid_method:  bilinear
    [2022-03-02 18:15:30+0000] INFO - prefect.TaskRunner | Frozen({'time': (96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 72), 'lat': (721,), 'lon': (1440,)})
    """

    """Dimensions:  (lat: 721, lon: 1440, month: 12)
    Coordinates:
    * lat      (lat) float32 90.0 89.75 89.5 89.25 ... -89.25 -89.5 -89.75 -90.0
    * lon      (lon) float32 -180.0 -179.8 -179.5 -179.2 ... 179.2 179.5 179.8
    * month    (month) int64 1 2 3 4 5 6 7 8 9 10 11 12
    Data variables:
        tasmax   (month, lat, lon) float32 dask.array<chunksize=(1, 17, 17), meta=np.ndarray>
    [2022-03-02 18:15:30+0000] INFO - prefect.TaskRunner | Frozen({'month': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 'lat': (17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 7), 'lon': (17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 12)})
    """
    print('spatial_anomalies_ds')
    print(spatial_anomalies_ds)

    print(spatial_anomalies_ds.chunks)
    print('interpolated_prediction:')
    print(interpolated_prediction)
    print(interpolated_prediction.chunks)
    rechunked_interpolated_prediction = rechunk_zarr_array_with_caching(
        interpolated_prediction, chunking_approach='full_time', max_mem='8GB'
    )
    print('rechunked_interpolated_prediction:')

    print(rechunked_interpolated_prediction)
    print(rechunked_interpolated_prediction.chunks)
    rechunked_spatial_anomalies = rechunk_zarr_array_with_caching(
        spatial_anomalies_ds, chunking_approach='full_time', max_mem='8GB'
    )
    print('rechunked_spatial_anomalies:')

    print(rechunked_spatial_anomalies)
    print(rechunked_spatial_anomalies.chunks)
    bcsd_results_ds = (
        rechunked_interpolated_prediction.groupby("time.month") + rechunked_spatial_anomalies
    )

    return bcsd_results_ds
