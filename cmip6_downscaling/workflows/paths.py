from typing import List, Optional, Tuple, Union


def build_obs_identifier(
    obs: str,
    train_period_start: str,
    train_period_end: str,
    variables: Union[str, List[str], Tuple[str]],
    **kwargs,
) -> str:
    """
    Build the common identifier for observation related data: the same pattern is used for: 1) chunked raw obs, 2) coarsened obs, and 3) coarsened then interpolated obs

    Parameters
    ----------
    obs : str
        From run hyperparameters
    train_period_start : str
        From run hyperparameters
    train_period_end : str
        From run hyperparameters
    variables : str
        From run hyperparameters

    Returns
    -------
    identifier : str
        string to be used in obs related paths as specified by the params
    """
    if isinstance(variables, str):
        variables = [variables]
    var_string = '_'.join(sorted(variables))
    return f'{obs}_{train_period_start}_{train_period_end}_{var_string}'


def build_gcm_identifier(
    gcm: str,
    scenario: str,
    train_period_start: str,
    train_period_end: str,
    predict_period_start: str,
    predict_period_end: str,
    variables: Union[str, List[str]],
    **kwargs,
) -> str:
    """
    Build the common identifier for GCM related data

    Parameters
    ----------
    gcm : str
        From run hyperparameters
    scenario : str
        From run hyperparameters
    train_period_start : str
        From run hyperparameters
    train_period_end : str
        From run hyperparameters
    predict_period_start : str
        From run hyperparameters
    predict_period_end : str
        From run hyperparameters
    variables : str
        From run hyperparameters

    Returns
    -------
    identifier : str
        string to be used in gcm related paths as specified by the params
    """
    if isinstance(variables, str):
        variables = [variables]
    var_string = '_'.join(variables)
    return f'{gcm}_{scenario}_{train_period_start}_{train_period_end}_{predict_period_start}_{predict_period_end}_{var_string}'


def make_rechunked_obs_path(
    chunking_approach: str, obs_identifier: Optional[str] = None, **kwargs
) -> str:
    """Build the path for rechunked observation

    Parameters
    ----------
    obs_identifier : str
        From run hyperparameters

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
        From run hyperparameters
    gcm_grid_spec: str
        Output of get_gcm_grid_spec

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
        From run hyperparameters
    gcm_grid_spec: str
        Output of get_gcm_grid_spec

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
    gcm_identifier : str
        From run hyperparameters

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
        From run hyperparameters

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
    return f"gard_pred_output/{gcm_identifier}_{bias_correction_method}_{model_type}_{label}.zarr"
