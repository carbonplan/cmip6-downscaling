from prefect import Flow, Parameter

from cmip6_downscaling import runtimes
from cmip6_downscaling.methods.bcsd.tasks import (
    calc_spacial_anomalies,
    coarsen_obs,
    fit_and_predict,
    interpolate_obs,
    interpolate_prediction,
    postprocess_bcsd,
)
from cmip6_downscaling.methods.common.tasks import (
    annual_summary,
    get_experiment,
    get_obs,
    make_run_parameters,
    monthly_summary,
    pyramid,
    rechunk,
    run_analyses,
)
from cmip6_downscaling.workflows.utils import rechunk_zarr_array_with_caching

runtime = runtimes.get_runtime()
print(runtime)


with Flow(
    name="bcsd", storage=runtime.storage, run_config=runtime.run_config, executor=runtime.executor
) as flow:

    run_parameters = make_run_parameters(
        method=Parameter("method"),
        obs=Parameter("obs"),
        model=Parameter("model"),
        scenario=Parameter("scenario"),
        variable=Parameter("variable"),
        latmin=Parameter("latmin"),
        latmax=Parameter("latmax"),
        lonmin=Parameter("lonmin"),
        lonmax=Parameter("lonmax"),
        train_dates=Parameter("train_period"),
        predict_dates=Parameter("predict_period"),
    )

    # input datasets
    obs_path = get_obs(run_parameters)
    experiment_path = get_experiment(run_parameters)

    coarse_obs_path = coarsen_obs(obs_path, experiment_path, run_parameters)

    interpolated_obs_path = interpolate_obs(obs_path, coarse_obs_path, run_parameters)

    spatial_anomalies_path = calc_spacial_anomalies(obs_path, interpolated_obs_path, run_parameters)

    # TODO: add spatial_chunks to config and do all full_time rechunks according to that pattern
    coarse_obs_full_time_path = rechunk_zarr_array_with_caching(
        zarr_store=coarse_obs_path, chunking_approach='full_time'
    )
    experiment_full_time_path = rechunk_zarr_array_with_caching(
        zarr_store=experiment_path, chunking_approach='full_time'
    )

    bias_corrected_path = fit_and_predict(
        experiment_full_time_path, coarse_obs_full_time_path, run_parameters
    )

    bias_corrected_fine_full_space_path = interpolate_prediction(
        bias_corrected_path, obs_path, run_parameters
    )  # rechunks to full_space

    bias_corrected_fine_full_time_path = rechunk(
        bias_corrected_fine_full_space_path, pattern='full_time'
    )

    postprocess_bcsd_path = postprocess_bcsd(
        bias_corrected_fine_full_time_path, spatial_anomalies_path, run_parameters
    )  # fine-scale maps (full_space) (time: 365)

    # temporary aggregations
    monthly_summary_path = monthly_summary(postprocess_bcsd_path, run_parameters)
    annual_summary_path = annual_summary(postprocess_bcsd_path, run_parameters)

    # analysis notebook
    analysis_location = run_analyses(postprocess_bcsd_path, run_parameters)

    # pyramids
    daily_pyramid_path = pyramid(postprocess_bcsd_path, run_parameters, key='daily', levels=4)
    monthly_pyramid_path = pyramid(monthly_summary_path, run_parameters, key='monthly', levels=4)
    annual_pyramid_path = pyramid(annual_summary_path, run_parameters, key='annual', levels=4)

    # if config.get('run_options.cleanup_flag') is True:
    #     cleanup.run_rsfip(gcm_identifier, obs_identifier)
