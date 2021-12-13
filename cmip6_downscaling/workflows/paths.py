from typing import List, Union


def build_obs_identifier(
    obs: str,
    train_period_start: str,
    train_period_end: str,
    variables: Union[str, List[str]],
    chunking_approach: str,
    **kwargs
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
    chunking_approach: str
        'full_space' or 'full_time'

    Returns
    -------
    identifier : str
        string to be used in obs related paths as specified by the params
    """
    if isinstance(variables, str):
        variables = [variables]
    var_string = '_'.join(variables)
    return f'{obs}_{train_period_start}_{train_period_end}_{var_string}_{chunking_approach}'


def make_rechunked_obs_path(
    obs_identifier: str,
    workdir: str = "az://cmip6",
    **kwargs
) -> str:
    """Build the path for rechunked observation

    Parameters
    ----------
    obs_identifier : str
        From run hyperparameters
    workdir : str, optional
        Intermediate files for caching (and might be used by other gcms), by default "az://cmip6"

    Returns
    -------
    rechunked_obs_path: str
        Path to which rechunked observation defined by the parameters should be stored
    """
    return f"{workdir}/intermediates/{obs_identifier}.zarr"


def make_coarse_obs_path(
    obs_identifier: str
    workdir: str = "az://cmip6",
    **kwargs
) -> str:
    """Build the path for coarsened observation

    Parameters
    ----------
    obs_identifier : str
        From run hyperparameters
    gcm_grid_spec: str
        Output of get_gcm_grid_spec
    workdir : str, optional
        Intermediate files for caching (and might be used by other gcms), by default "az://cmip6"

    Returns
    -------
    coarse_obs_path: str
        Path to which coarsened observation defined by the parameters should be stored
    """
    return f"{workdir}/intermediates/coarsened_{obs_identifier}_{gcm_grid_spec}.zarr"


def make_interpolated_obs_path(
    obs_identifier: str,
    gcm_grid_spec: str,
    workdir: str = "az://cmip6",
    **kwargs
) -> str:
    """Build the path for coarsened observation that has then been interpolated back to the observation grid

    Parameters
    ----------
    obs_identifier : str
        From run hyperparameters
    gcm_grid_spec: str
        Output of get_gcm_grid_spec
    workdir : str, optional
        Intermediate files for caching (and might be used by other gcms), by default "az://cmip6"

    Returns
    -------
    interpolated_obs_path: str
        Path to which interpolated observation defined by the parameters should be stored
    """
    return f"{workdir}/intermediates/interpolated_{obs_identifier}_{gcm_grid_spec}.zarr"
