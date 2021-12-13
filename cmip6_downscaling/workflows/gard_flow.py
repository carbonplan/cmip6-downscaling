import os

from dask_kubernetes import KubeCluster, make_pod_spec
from prefect import Flow, task
from prefect.executors import DaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure

from cmip6_downscaling.methods.gard import gard_fit_and_predict, gard_postprocess, gard_preprocess

coarsen_obs_task = task(
    get_coarse_obs, 
    result=FunnelResult(cache_store, serializer=serializer), 
    target=make_coarse_obs_path
)

preprocess_bcsd_task = task(preprocess_bcsd, log_stdout=True, nout=2)

prep_bcsd_inputs_task = task(prep_bcsd_inputs, log_stdout=True, nout=3)

fit_and_predict_task = task(fit_and_predict, log_stdout=True)

postprocess_bcsd_task = task(postprocess_bcsd, log_stdout=True)


@task(result=FunnelResult(cache_store, serializer=serializer), target=make_interpolated_obs_path)
def coarsen_and_interpolate_obs(
    ds_obs, 
    gcm,
    connection_string,
    obs_identifier,
    gcm_grid_spec, 
    workdir,
):
    """
    # goal here is to cache: 1) the rechunked fine obs, 2) the coarsened obs, and 3) the regridded obs
    """
    # make the correct path patterns 
    path_dict = {
        'obs_identifier': obs_identifier,
        'gcm_grid_spec': gcm_grid_spec,
        'workdir': workdir,

    }
    rechunked_obs_path = make_rechunked_obs_path(**path_dict)

    # rechunked to full space 
    ds_obs_full_space = rechunk_zarr_array_with_caching(
        zarr_array=ds_obs,
        output_path=rechunked_obs_path,
        connection_string=connection_string,
        chunking_approach='full_space',
    )

    # regrid to coarse scale 
    ds_obs_coarse = coarsen_obs_task(
        ds_obs=ds_obs_full_space, 
        gcm=gcm,
        connection_string=connection_string,
        **path_dict
    )

    # interpolate to fine scale again 
    ds_obs_interpolated = regrid_ds(
        ds=ds_obs_coarse,
        target_grid_ds=ds_obs_full_space.isel(time=0),
        connection_string=connection_string,
    )

    return ds_obs_interpolated


def gard_preprocess(
    obs: str,
    gcm: str,
    train_period_start: str,
    train_period_end: str,
    variable: str,
    features: List[str],
    connection_string: str,
    workdir: str
):
    """


    Parameters
    ----------
    obs: str
        Name of observation dataset 
    gcm : str
        Name of GCM
    train_period_start : str
        Date for training period start (e.g. '1985')
    train_period_end : str
        Date for training period end (e.g. '2015')
    variable : str
        Variable of interest in CMIP conventions (e.g. 'tasmax')
    connection_string : str
        Connection string to give you read/write access to the out buckets specified above

    Returns
    -------

    """
    # get all the variables
    all_vars = list(set([variable] + features))

    # get observation and gcm 
    ds_obs = open_era5(all_vars, start_year=train_period_start, end_year=train_period_end)
    ds_gcm = get_gcm(
        gcm=gcm,
        scenario=scenario,
        variables=all_vars,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
    )

    # get gcm grid spec 
    gcm_grid_spec = get_gcm_grid_spec(gcm_ds=ds_gcm)

    # get interpolated observation 
    obs_identifier = build_obs_identifier(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variables=all_vars,
        chunking_approach='full_space',
    )
    ds_obs_regridded = coarsen_and_interpolate_obs(
        ds_obs=ds_obs, 
        obs_identifier=obs_identifier,
        gcm=gcm,
        gcm_grid_spec=gcm_grid_spec, 
        workdir=workdir,
        connection_string=connection_string
    )

    # 

    # bias correct the interpolated obs and gcm 
    ds_obs_bias_corrected, ds_obs_bias_corrected = bias_correct(
        ds_obs=ds_obs_regridded,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        variables=all_vars,

    )

    return ds_obs, ds_obs_bias_corrected, ds_gcm_bias_corrected


def gard_flow(
    model,
    label_name,
    feature_list=None,
    dim='time',
    bias_correction_method='quantile_transform',
    bc_kwargs=None,
    generate_scrf=True,
):
    """
    Parameters
    ----------
    model                 : a GARD model instance to be fitted pointwise
    feature_list          : a list of feature names to be used in predicting
    dim                   : string. dimension to apply the model along. Default is ``time``.
    bias_correction_method: string of the name of bias correction model
    bc_kwargs             : kwargs dict. directly passed to the bias correction model
    generate_scrf         : boolean. indicates whether a spatio-temporal correlated random field (scrf) will be
                            generated based on the fine resolution data provided in .fit as y. if false, it is
                            assumed that a pre-generated scrf will be passed into .predict as an argument that
                            matches the prediction result dimensions.
    spatial_feature       : (3, 3)
    """
    self._dim = dim
    if not isinstance(model, (AnalogBase, PureRegression)):
        raise TypeError('model must be part of the GARD family of pointwise models ')
    self.features = feature_list
    self.label_name = label_name
    self._model = model
    self.thresh = model.thresh

    # shared between multiple method types but point wise
    # TODO: spatial features
    # TODO: extend this to include transforming the feature space into PCA, etc
    # map + 1d component (pca) of ocean surface temperature for precip prediction
