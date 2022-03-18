from prefect import Flow, Parameter

from cmip6_downscaling import runtimes
from cmip6_downscaling.methods.bcsd.tasks import (
    coarsen_obs,
    fit_and_predict,
    interpolate_prediction,
    postprocess_bcsd,
    spatial_anomalies,
)
from cmip6_downscaling.methods.common.tasks import (
    annual_summary,
    get_experiment,
    get_obs,
    make_run_parameters,
    monthly_summary,
    pyramid,
    rechunk,
    regrid,
    run_analyses,
)

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

    experiment_train_path = get_experiment(run_parameters, time_subset='train_period')
    experiment_predict_path = get_experiment(run_parameters, time_subset='predict_period')

    coarse_obs_path = coarsen_obs(obs_path, experiment_train_path, run_parameters)

    interpolated_obs_path = regrid(source_path=obs_path, target_grid_path=obs_path)

    interpolated_obs_full_time_path = rechunk(
        path=interpolated_obs_path, pattern="full_time", run_parameters=run_parameters
    )
    obs_full_time_path = rechunk(path=obs_path, pattern="full_time", run_parameters=run_parameters)
    spatial_anomalies_path = spatial_anomalies(
        obs_full_time_path, interpolated_obs_full_time_path, run_parameters
    )
    coarse_obs_full_time_path = rechunk(coarse_obs_path, pattern='full_time')
    experiment_train_full_time_path = rechunk(
        experiment_train_path, pattern=coarse_obs_full_time_path
    )
    experiment_predict_full_time_path = rechunk(
        experiment_predict_path, pattern=coarse_obs_full_time_path
    )

    bias_corrected_path = fit_and_predict(
        experiment_train_full_time_path=experiment_train_full_time_path,
        experiment_predict_full_time_path=experiment_predict_full_time_path,
        coarse_obs_full_time_path=coarse_obs_full_time_path,
        run_parameters=run_parameters,
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
