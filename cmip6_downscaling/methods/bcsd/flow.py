from __future__ import annotations

import warnings

from prefect import Flow, Parameter

from cmip6_downscaling import runtimes
from cmip6_downscaling.methods.bcsd.tasks import (
    fit_and_predict,
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

warnings.filterwarnings(
    "ignore",
    "(.*) filesystem path not explicitly implemented. falling back to default implementation. This filesystem may not be tested",
    category=UserWarning,
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
        member=Parameter("member"),
        grid_label=Parameter("grid_label"),
        table_id=Parameter("table_id"),
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
    obs_full_space_path = rechunk(path=obs_path, pattern='full_space')
    experiment_train_path = get_experiment(run_parameters, time_subset='train_period')
    experiment_predict_path = get_experiment(run_parameters, time_subset='predict_period')

    # after regridding coarse_obs will have smaller array size in space but still
    # be chunked finely along time. but that's good to get it for regridding back to
    # the interpolated obs in next task
    coarse_obs_path = regrid(obs_full_space_path, experiment_train_path)

    # interpolated obs should have same exact chunking schema as ds at `obs_full_space_path`
    interpolated_obs_path = regrid(source_path=coarse_obs_path, target_grid_path=obs_path)

    interpolated_obs_full_time_path = rechunk(path=interpolated_obs_path, pattern="full_time")
    obs_full_time_path = rechunk(path=obs_path, pattern="full_time")
    spatial_anomalies_path = spatial_anomalies(obs_full_time_path, interpolated_obs_full_time_path)
    coarse_obs_full_time_path = rechunk(coarse_obs_path, pattern='full_time')
    experiment_train_full_time_path = rechunk(experiment_train_path, pattern='full_time')
    experiment_predict_full_time_path = rechunk(
        experiment_predict_path,
        pattern='full_time',
        template=coarse_obs_full_time_path,
    )
    bias_corrected_path = fit_and_predict(
        experiment_train_full_time_path=experiment_train_full_time_path,
        experiment_predict_full_time_path=experiment_predict_full_time_path,
        coarse_obs_full_time_path=coarse_obs_full_time_path,
        run_parameters=run_parameters,
    )
    bias_corrected_full_space_path = rechunk(
        bias_corrected_path,
        pattern='full_space',
        template=obs_full_space_path,
    )
    bias_corrected_fine_full_space_path = regrid(
        source_path=bias_corrected_full_space_path, target_grid_path=obs_path
    )

    bias_corrected_fine_full_time_path = rechunk(
        bias_corrected_fine_full_space_path,
        pattern='full_time',
        template=obs_full_time_path,
    )
    final_bcsd_full_time_path = postprocess_bcsd(
        bias_corrected_fine_full_time_path, spatial_anomalies_path
    )  # fine-scale maps (full_space) (time: 365)

    # temporary aggregations - these come out in full time
    monthly_summary_path = monthly_summary(final_bcsd_full_time_path, run_parameters)
    annual_summary_path = annual_summary(final_bcsd_full_time_path, run_parameters)

    # analysis notebook
    analysis_location = run_analyses(final_bcsd_full_time_path, run_parameters)

    # since pyramids require full space we now rechunk everything into full
    # space before passing into pyramid step. we probably want to add a cleanup
    # to this step in particular since otherwise we will have an exact
    # duplicate of the daily, monthly, and annual datasets
    final_bcsd_full_space_path = rechunk(final_bcsd_full_time_path, pattern='full_space')

    # make temporal summaries
    monthly_summary_full_space_path = rechunk(monthly_summary_path, pattern='full_space')
    annual_summary_full_space_path = rechunk(annual_summary_path, pattern='full_space')

    # pyramids
    daily_pyramid_path = pyramid(final_bcsd_full_space_path, levels=4)
    monthly_pyramid_path = pyramid(monthly_summary_full_space_path, levels=4)
    annual_pyramid_path = pyramid(annual_summary_full_space_path, levels=4)

    # # if config.get('run_options.cleanup_flag') is True:
    # #     cleanup.run_rsfip(gcm_identifier, obs_identifier)
