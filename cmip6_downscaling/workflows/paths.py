from typing import Optional, Tuple

from cmip6_downscaling import config


def build_obs_identifier(
    obs: str,
    train_period_start: str,
    train_period_end: str,
    latmin: str,
    latmax: str,
    lonmin: str,
    lonmax: str,
    variable: str,
    **kwargs,
) -> str:
    """
    Build the common identifier for observation related data: the same pattern is used for: 1) chunked raw obs, 2) coarsened obs, and 3) coarsened then interpolated obs

    Parameters
    ----------
    obs : str
        Name of observation dataset
    train_period_start : str
        Start year of the training period
    train_period_end : str
        End year of the training period
    latmin : str
            Minimum latitude
    latmax : str
            Maximum latitude
    lonmin : str
            Minimum longitude
    lonmax : str
            Max longitude
    variables : str or List[str]
        Name of variable

    Returns
    -------
    identifier : str
        string to be used in obs related paths as specified by the params
    """

    obs_identifier = config.get('storage.obs_identifier_template').format(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        latmin=latmin,
        latmax=latmax,
        lonmin=lonmin,
        lonmax=lonmax,
        variable=variable,
    )
    return obs_identifier


def build_gcm_identifier(
    gcm: str,
    scenario: str,
    train_period_start: str,
    train_period_end: str,
    predict_period_start: str,
    predict_period_end: str,
    latmin: str,
    latmax: str,
    lonmin: str,
    lonmax: str,
    variable: str,
    **kwargs,
) -> str:
    """
    Build the common identifier for GCM related data

    Parameters
    ----------
    gcm : str
        Name of the GCM model
    scenario : str
        Name of the future emission scenario to load
    train_period_start : str
        Start year of the training period
    train_period_end : str
        End year of the training period
    predict_period_start : str
        Start year of the prediction period
    predict_period_end : str
        End year of the prediction period
    latmin : str
            Minimum latitude
    latmax : str
            Maximum latitude
    lonmin : str
            Minimum longitude
    lonmax : str
            Max longitude
    variables : str or List[str]
        Name of variable

    Returns
    -------
    identifier : str
        string to be used in gcm related paths as specified by the params
    """

    gcm_identifier = config.get('storage.gcm_identifier_template').format(
        gcm=gcm,
        scenario=scenario,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        latmin=latmin,
        latmax=latmax,
        lonmin=lonmin,
        lonmax=lonmax,
        variable=variable,
    )
    return gcm_identifier


def make_rechunked_obs_path(
    chunking_approach: str, obs_identifier: Optional[str] = None, **kwargs
) -> str:
    """Build the path for rechunked observation

    Parameters
    ----------
    obs_identifier : str
        Output from build_obs_identifier. String to identify the observation dataset used
    chunking_approach : str
        How the data is chunked. Eg. `full_time`, `full_space`, `matched` or None

    Returns
    -------
    rechunked_obs_path: str
        Path to which rechunked observation defined by the parameters should be stored
    """
    if obs_identifier is None:
        obs_identifier = build_obs_identifier(**kwargs)

    return f"rechunked_obs/{obs_identifier}_{chunking_approach}.zarr"


def make_coarse_obs_path(
    gcm_grid_spec: str, chunking_approach: str, obs_identifier: Optional[str] = None, **kwargs
) -> str:
    """Build the path for coarsened observation

    Parameters
    ----------
    obs_identifier : str
        Output from build_obs_identifier. String to identify the observation dataset used
    chunking_approach : str
        How the data is chunked. Eg. `full_time`, `full_space`, `matched` or None
    gcm_grid_spec: str
        Output of get_gcm_grid_spec. String to identify the GCM grid

    Returns
    -------
    coarse_obs_path: str
        Path to which coarsened observation defined by the parameters should be stored
    """
    if obs_identifier is None:
        obs_identifier = build_obs_identifier(**kwargs)

    return f"coarsened_obs/{obs_identifier}_{chunking_approach}_{gcm_grid_spec}.zarr"


def make_interpolated_obs_path(
    gcm_grid_spec: str, chunking_approach: str, obs_identifier: str, **kwargs
) -> str:
    """Build the path for coarsened observation that has then been interpolated back to the observation grid

    Parameters
    ----------
    obs_identifier : str
        Output from build_obs_identifier. String to identify the observation dataset used
    chunking_approach : str
        How the data is chunked. Eg. `full_time`, `full_space`, `matched` or None
    gcm_grid_spec: str
        Output of get_gcm_grid_spec. String to identify the GCM grid

    Returns
    -------
    interpolated_obs_path: str
        Path to which interpolated observation defined by the parameters should be stored
    """
    if obs_identifier is None:
        obs_identifier = build_obs_identifier(**kwargs)
    return f"interpolated_obs/{obs_identifier}_{chunking_approach}_{gcm_grid_spec}.zarr"


def make_interpolated_gcm_path(
    obs: str, chunking_approach: str, gcm_identifier: Optional[str] = None, **kwargs
) -> str:
    """Build the path for GCM that has then been interpolated back to the observation grid

    Parameters
    ----------
    obs: str
        name of obs dataset
    obs_identifier : str
        Output from build_obs_identifier. String to identify the observation dataset used
    gcm_identifier : str
        Output from build_gcm_identifier. String to identify the GCM dataset used

    Returns
    -------
    interpolated_obs_path: str
        Path to which interpolated observation defined by the parameters should be stored
    """
    if gcm_identifier is None:
        gcm_identifier = build_gcm_identifier(**kwargs)
    return f"interpolated_gcm/{gcm_identifier}_{chunking_approach}_{obs}.zarr"


def make_bias_corrected_obs_path(
    method: str,
    gcm_grid_spec: str,
    obs_identifier: Optional[str] = None,
    chunking_approach: Optional[str] = None,
    **kwargs,
):
    """
    Path to save "bias_corrected" or pre-processed (e.g. normalized or quantile transformed) observation dataset.
    The dataset might be coarsened hence the need for gcm_grid_spec in the path.

    Parameters
    ----------
    method: str
        String to identify bias correction method used
    gcm_grid_spec: str
        Output of get_gcm_grid_spec. String to identify the GCM grid
    obs_identifier : str
        Output from build_obs_identifier. String to identify the observation dataset used
    chunking_approach : str
        How the data is chunked. Eg. `full_time`, `full_space`, `matched` or None

    Returns
    -------
    bias_corrected_obs_path: str
        Path to which bias corrected observation defined by the parameters should be stored
    """
    if obs_identifier is None:
        obs_identifier = build_obs_identifier(**kwargs)
    if chunking_approach is None:
        chunking_approach = ''
    else:
        chunking_approach = '_' + chunking_approach

    return f"bias_corrected_obs/{obs_identifier}{chunking_approach}_{gcm_grid_spec}_{method}.zarr"


def make_rechunked_gcm_path(
    chunking_approach: str, gcm_identifier: Optional[str] = None, **kwargs
) -> str:
    """Build the path for rechunked GCM

    Parameters
    ----------
    gcm_identifier : str
        Output from build_gcm_identifier. String to identify the GCM dataset used
    chunking_approach : str
        How the data is chunked. Eg. `full_time`, `full_space`, `matched` or None

    Returns
    -------
    rechunked_gcm_path: str
        Path to which rechunked GCM defined by the parameters should be stored
    """
    if gcm_identifier is None:
        gcm_identifier = build_gcm_identifier(**kwargs)

    return f"rechunked_gcm/{gcm_identifier}_{chunking_approach}.zarr"


def make_bias_corrected_gcm_path(
    method: str,
    gcm_identifier: Optional[str] = None,
    chunking_approach: Optional[str] = None,
    **kwargs,
):
    """
    Path to save bias_corrected or pre-processed (e.g. normalized or quantile transformed) GCM dataset

    Parameters
    ----------
    method: str
        String to identify bias correction method used
    gcm_identifier : str
        Output from build_gcm_identifier. String to identify the GCM dataset used
    chunking_approach : str
        How the data is chunked. Eg. `full_time`, `full_space`, `matched` or None

    Returns
    -------
    bias_corrected_obs_path: str
        Path to which bias corrected observation defined by the parameters should be stored
    """
    if gcm_identifier is None:
        gcm_identifier = build_gcm_identifier(**kwargs)
    if chunking_approach is None:
        chunking_approach = ''
    else:
        chunking_approach = '_' + chunking_approach

    return f"bias_corrected_gcm/{gcm_identifier}{chunking_approach}_{method}.zarr"


def make_gard_predict_output_path(
    gcm_identifier: str, bias_correction_method: str, model_type: str, label: str, **kwargs
):
    """
    Path to save GARD prediction output

    Parameters
    ----------
    gcm_identifier : str
        Output from build_gcm_identifier. String to identify the GCM dataset used
    bias_correction_method: str
        String to identify bias correction method used
    model_type : str
        The type of GARD model (`AnalogRegression`, `PureAnalog` or `PureRegression`)
    label: str
        The variable being predicted

    Returns
    -------
    gard_prediction_output_path: str
        Path to which GARD prediction output defined by the parameters should be stored
    """
    return f"gard_pred_output/{gcm_identifier}_{bias_correction_method}_{model_type}_{label}.zarr"


def make_scrf_path(
    obs_identifier: str,
    label: str,
    **kwargs,
):
    """
    Path where spatially-temporally correlated random fields (SCRF) are saved.

    Parameters
    ----------
    obs_identifier : str
        Output from build_obs_identifier. String to identify the observation dataset used
    label: str
        The variable being predicted

    Returns
    -------
    scrf_path : str
        Path of SCRF
    """
    return f"scrf/{obs_identifier}_{label}.zarr"


def make_gard_post_processed_output_path(
    gcm_identifier: str, bias_correction_method: str, model_type: str, label: str, **kwargs
):
    """
    Path to save GARD post processed output

    Parameters
    ----------
    gcm_identifier : str
        Output from build_gcm_identifier. String to identify the GCM dataset used
    bias_correction_method: str
        String to identify bias correction method used
    model_type : str
        The type of GARD model (`AnalogRegression`, `PureAnalog` or `PureRegression`)
    label: str
        The variable being predicted

    Returns
    -------
    gard_post_processed_output_path: str
        Path to which GARD post processed output defined by the parameters should be stored
    """
    return f"gard_post_processed_output/{gcm_identifier}_{bias_correction_method}_{model_type}_{label}.zarr"


def make_epoch_trend_path(
    gcm_identifier: str, day_rolling_window: int, year_rolling_window: int, **kwargs
):
    """
    Path to save epoch trend from MACA

    Parameters
    ----------
    gcm_identifier : str
        Output from build_gcm_identifier. String to identify the GCM dataset used
    day_rolling_window: int
        Number of days to include when calculating the rolling average
    year_rolling_window: int
        Number of years to include when calculating the rolling average

    Returns
    -------
    epoch_trend_path : str
        Path to save the epoch trend
    """
    return f'gcm_epoch_trend/{gcm_identifier}_{day_rolling_window}_day_{year_rolling_window}_year_rolling.zarr'


def make_epoch_adjusted_gcm_path(
    gcm_identifier: str, day_rolling_window: int, year_rolling_window: int, **kwargs
):
    """
    Path to save epoch adjusted GCM data from MACA

    Parameters
    ----------
    gcm_identifier : str
        Output from build_gcm_identifier. String to identify the GCM dataset used
    day_rolling_window: int
        Number of days to include when calculating the rolling average
    year_rolling_window: int
        Number of years to include when calculating the rolling average

    Returns
    -------
    epoch_adjusted_gcm_path : str
        Path to save epoch adjusted GCM data from MACA
    """
    return f'epoch_adjusted_gcm/{gcm_identifier}_{day_rolling_window}_day_{year_rolling_window}_year_rolling.zarr'


def make_epoch_adjusted_downscaled_gcm_path(
    gcm_identifier: str,
    label: str,
    subdomain_bound: Optional[Tuple[float, float, float, float]] = (0, -90, 360, 90),
    **kwargs,
):
    """
    Path to save the downscaled GCM data where the epoch trend has not been repalced. Includes subdomain
    bound because the downscaling step in MACA is performed on subdomains.

    Parameters
    ----------
    gcm_identifier : str
        Output from build_gcm_identifier. String to identify the GCM dataset used
    label: str
        The variable being predicted
    subdomain_boudn: Tuple of floats
        min_lon, min_lat, max_lon, max_lat

    Returns
    -------
    maca_epoch_adjusted_downscaled_output_path : str
        Path to save the downscaled GCM data where the epoch trend has not been repalced.
    """
    min_lon, min_lat, max_lon, max_lat = subdomain_bound
    return f'maca_epoch_adjusted_downscaled_output/{gcm_identifier}_{label}_{min_lon:0.1f}_{min_lat:0.1f}_{max_lon:0.1f}_{max_lat:0.1f}.zarr'


def make_epoch_replaced_downscaled_gcm_path(
    gcm_identifier: str, day_rolling_window: int, year_rolling_window: int, **kwargs
):
    """
    Path to save the downscaled GCM data where the epoch trend has already been repalced.

    Parameters
    ----------
    gcm_identifier : str
        Output from build_gcm_identifier. String to identify the GCM dataset used
    day_rolling_window: int
        Number of days to include when calculating the rolling average
    year_rolling_window: int
        Number of years to include when calculating the rolling average

    Returns
    -------
    epoch_replaced_gcm : str
        Path to save the downscaled GCM data where the epoch trend has already been repalced.
    """
    return f'epoch_replaced_gcm/{gcm_identifier}_{day_rolling_window}_day_{year_rolling_window}_year_rolling.zarr'


def make_maca_output_path(gcm_identifier: str, label: str, **kwargs):
    """
    Path to save final MACA output after fine scale bias correction.

    Parameters
    ----------
    gcm_identifier : str
        Output from build_gcm_identifier. String to identify the GCM dataset used
    label: str
        The variable being predicted

    Returns
    -------
    maca_output_path : str
        MACA output path
    """
    return f"maca_output/{gcm_identifier}_{label}.zarr"


def make_pyramid_path(gcm_identifier: str, **kwargs) -> str:
    """Build the path for the pyramid


    Parameters
    ----------
    gcm_identifier : str
        Output from build_gcm_identifier. String to identify the gcm dataset used

    Returns
    -------
    pyramid_path: str
        Path to gcm_predict ds file location
    """
    return f"/pyramid/{gcm_identifier}.pyr"


# ---addl bcsd paths


def make_return_obs_path(obs_identifier: str, **kwargs) -> str:
    """Build the path for return obs task

    Parameters
    ----------
    obs_identifier : str
        Output from build_obs_identifier. String to identify the observation dataset used

    Returns
    -------
    obs_ds path: str
        Path to bcsd obs_ds file location
    """
    return f"obs_ds/{obs_identifier}.zarr"


def make_spatial_anomalies_path(obs_identifier: str = None, **kwargs) -> str:
    """Build the path for spatial anomalies

    Parameters
    ----------
    obs_identifier : str
        Output from build_obs_identifier. String to identify the observation dataset used

    Returns
    -------
    spatial_anomalies_path: str
        Path to bcsd spatial anomalies file location
    """
    return f"spatial_anomalies/{obs_identifier}.zarr"


def make_gcm_predict_path(gcm_identifier: str = None, **kwargs) -> str:
    """Build the path for the gcm predict dataset

    Parameters
    ----------
    gcm_identifier : str
        Output from build_gcm_identifier. String to identify the gcm dataset used

    Returns
    -------
    gcm_predict_path: str
        Path to gcm_predict ds file location
    """
    return f"gcm_predict/{gcm_identifier}.zarr"


def make_bias_corrected_path(gcm_identifier: str = None, **kwargs) -> str:
    """Build the path for the bias corrected bcsd dataset

    Parameters
    ----------
    gcm_identifier : str
        Output from build_gcm_identifier. String to identify the gcm dataset used

    Returns
    -------
    bias_corrected_path: str
        Path to bcsd bias_corrected ds file location
    """
    return f"bias_corrected/{gcm_identifier}.zarr"


def make_bcsd_output_path(gcm_identifier: str = None, **kwargs) -> str:
    """Build the path for the bcsd output dataset

    Parameters
    ----------
    gcm_identifier : str
        Output from build_gcm_identifier. String to identify the gcm dataset used

    Returns
    -------
    bcsd_output_path: str
        Path to bcsd output ds file location
    """
    return f"bcsd_output/{gcm_identifier}.zarr"
