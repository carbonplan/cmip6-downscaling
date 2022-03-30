import warnings

from prefect import Flow, Parameter

from cmip6_downscaling import runtimes
from cmip6_downscaling.methods.common.tasks import (  # annual_summary,; monthly_summary,; pyramid,; run_analyses,
    get_experiment,
    get_obs,
    make_run_parameters,
    rechunk,
    regrid,
)
from cmip6_downscaling.methods.gard.tasks import (
    coarsen_and_interpolate,
    fit_and_predict,
    postprocess,
)

warnings.filterwarnings(
    "ignore",
    "(.*) filesystem path not explicitly implemented. falling back to default implementation. This filesystem may not be tested",
    category=UserWarning,
)

runtime = runtimes.get_runtime()
print(runtime)


with Flow(
    name="gard", storage=runtime.storage, run_config=runtime.run_config, executor=runtime.executor
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
    obs_full_space_path = rechunk(path=obs_path, chunking_pattern='full_space')
    experiment_train_path = get_experiment(run_parameters, time_subset='train_period')
    experiment_predict_path = get_experiment(run_parameters, time_subset='predict_period')

    # after regridding coarse_obs will have smaller array size in space but still
    # be chunked finely along time. but that's good to get it for regridding back to
    # the interpolated obs in next task
    # interpolated obs should have same exact chunking schema as ds at `obs_full_space_path`
    interpolated_obs_full_space_path = coarsen_and_interpolate(
        obs_full_space_path, experiment_train_path
    )

    # TODO: do we need the templates as well for the rechunking? probably defer to bcsd flow here
    interpolated_obs_full_time_path = rechunk(
        interpolated_obs_full_space_path, template=experiment_train_path, pattern='full_time'
    )

    # get gcm data into full space to prep for interpolation
    experiment_train_full_space_path = rechunk(experiment_train_path, chunking_pattern="full_space")
    experiment_predict_full_space_path = rechunk(
        experiment_predict_path, chunking_pattern="full_space"
    )

    # interpolate gcm to finescale
    experiment_train_fine_full_space_path = regrid(
        source_path=experiment_train_full_space_path, target_grid_path=obs_path
    )
    experiment_predict_fine_full_space_path = regrid(
        source_path=experiment_predict_full_space_path, target_grid_path=obs_path
    )
    # TODO: do we need the templates as well for the rechunking? probably defer to bcsd flow here
    experiment_train_fine_full_time_path = rechunk(
        experiment_train_fine_full_space_path, chunking_pattern="full_time"
    )
    experiment_predict_fine_full_time_path = rechunk(
        experiment_predict_fine_full_space_path,
        chunking_pattern="full_time",
        chunking_template_file=interpolated_obs_full_time_path,
    )

    # fit and predict (TODO: put the transformation steps currently in the prep_gard_input task into the fit and predict step)
    model_output_path = fit_and_predict(
        xtrain_path=interpolated_obs_full_time_path,
        ytrain_path=experiment_train_fine_full_time_path,
        xpred_path=experiment_predict_fine_full_time_path,
        run_parameters=run_parameters,
    )  # [transformation, gard_model_options])

    # # post process (add back in scrf)
    # # necessary input chunk pattern:
    final_gard_path = postprocess(
        model_output_path=model_output_path, run_parameters=run_parameters
    )  # this comes out in full-time currently

    # # temporary aggregations - these come out in full time
    # monthly_summary_path = monthly_summary(final_bcsd_full_time_path, run_parameters)
    # annual_summary_path = annual_summary(final_bcsd_full_time_path, run_parameters)

    # # analysis notebook (shared with BCSD)
    # analysis_location = run_analyses(final_bcsd_full_time_path, run_parameters)

    # # since pyramids require full space we now rechunk everything into full
    # # space before passing into pyramid step. we probably want to add a cleanup
    # # to this step in particular since otherwise we will have an exact
    # # duplicate of the daily, monthly, and annual datasets
    # final_bcsd_full_space_path = rechunk(final_bcsd_full_time_path, chunking_pattern='full_space')

    # # make temporal summaries
    # monthly_summary_full_space_path = rechunk(monthly_summary_path, chunking_pattern='full_space')
    # annual_summary_full_space_path = rechunk(annual_summary_path, chunking_pattern='full_space')

    # # pyramids
    # daily_pyramid_path = pyramid(final_bcsd_full_space_path, levels=4)
    # monthly_pyramid_path = pyramid(monthly_summary_full_space_path, levels=4)
    # annual_pyramid_path = pyramid(annual_summary_full_space_path, levels=4)

    # # if config.get('run_options.cleanup_flag') is True:
    # #     cleanup.run_rsfip(gcm_identifier, obs_identifier)
