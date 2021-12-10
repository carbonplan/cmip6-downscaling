from typing import List, Union


def build_obs_identifier(
    obs: str,
    train_period_start: str,
    train_period_end: str,
    variables: Union[str, List[str]],
    chunk_by: str,
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
    chunk_by: str
        Whether the data is chunked by time or space

    Returns
    -------
    identifier : str
        string to be used in obs related paths as specified by the params
    """
    if isinstance(variables, str):
        variables = [variables]
    var_string = '_'.join(variables)
    return f'{obs}_{train_period_start}_{train_period_end}_{var_string}_{chunk_by}_chunked'


def make_rechunked_obs_path(
    obs: str,
    train_period_start: str,
    train_period_end: str,
    variables: Union[str, List[str]],
    chunk_by: str,
    workdir: str = "az://cmip6",
) -> str:
    """Build the path for rechunked observation

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
    chunk_by: str
        Whether the data is chunked by time or space
    workdir : str, optional
        Intermediate files for caching (and might be used by other gcms), by default "az://cmip6"

    Returns
    -------
    rechunked_obs_path: str
        Path to which rechunked observation defined by the parameters should be stored
    """
    obs_identifier = build_obs_identifier(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variables=variables,
        chunk_by=chunk_by,
    )
    return f"{workdir}/intermediates/{obs_identifier}.zarr"


def make_coarse_obs_path(
    obs: str,
    train_period_start: str,
    train_period_end: str,
    variables: Union[str, List[str]],
    gcm_grid_spec: str,
    chunk_by: str,
    workdir: str = "az://cmip6",
) -> str:
    """Build the path for coarsened observation

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
    gcm_grid_spec: str
        Output of get_gcm_grid_spec
    chunk_by: str
        Whether the data is chunked by time or space
    workdir : str, optional
        Intermediate files for caching (and might be used by other gcms), by default "az://cmip6"

    Returns
    -------
    coarse_obs_path: str
        Path to which coarsened observation defined by the parameters should be stored
    """
    obs_identifier = build_obs_identifier(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variables=variables,
        chunk_by=chunk_by,
    )
    return f"{workdir}/intermediates/coarsened_{obs_identifier}_{gcm_grid_spec}.zarr"


def make_interpolated_obs_path(
    obs: str,
    train_period_start: str,
    train_period_end: str,
    variables: Union[str, List[str]],
    gcm_grid_spec: str,
    chunk_by: str,
    workdir: str = "az://cmip6",
) -> str:
    """Build the path for coarsened observation that has then been interpolated back to the observation grid

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
    gcm_grid_spec: str
        Output of get_gcm_grid_spec
    chunk_by: str
        Whether the data is chunked by time or space
    workdir : str, optional
        Intermediate files for caching (and might be used by other gcms), by default "az://cmip6"

    Returns
    -------
    interpolated_obs_path: str
        Path to which interpolated observation defined by the parameters should be stored
    """
    obs_identifier = build_obs_identifier(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variables=variables,
        chunk_by=chunk_by,
    )
    return f"{workdir}/intermediates/interpolated_{obs_identifier}_{gcm_grid_spec}.zarr"
