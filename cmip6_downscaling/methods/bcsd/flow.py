from __future__ import annotations

import warnings

from prefect import Flow, Parameter

from cmip6_downscaling import runtimes
from cmip6_downscaling.methods.bcsd.tasks import (
    fit_and_predict,
    postprocess_bcsd,
    spatial_anomalies,
)
from cmip6_downscaling.methods.common.tasks import (  # run_analyses,; get_weights,
    finalize,
    get_experiment,
    get_obs,
    get_pyramid_weights,
    make_run_parameters,
    pyramid,
    rechunk,
    regrid,
    time_summary,
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
        features=Parameter("features"),
        latmin=Parameter("latmin"),
        latmax=Parameter("latmax"),
        lonmin=Parameter("lonmin"),
        lonmax=Parameter("lonmax"),
        train_dates=Parameter("train_dates"),
        predict_dates=Parameter("predict_dates"),
        bias_correction_method=Parameter("bias_correction_method"),
        bias_correction_kwargs=Parameter("bias_correction_kwargs"),
        model_type=Parameter("model_type"),
        model_params=Parameter("model_params"),
    )

    p = {}

    # input datasets
    p['obs_path'] = get_obs(run_parameters)

    p['obs_full_space_path'] = rechunk(path=p['obs_path'], pattern='full_space')
    p['experiment_train_path'] = get_experiment(run_parameters, time_subset='train_period')
    p['experiment_predict_path'] = get_experiment(run_parameters, time_subset='predict_period')

    # after regridding coarse_obs will have smaller array size in space but still
    # be chunked finely along time. but that's good to get it for regridding back to
    # the interpolated obs in next task

    p['coarse_obs_path'] = regrid(
        p['obs_full_space_path'], p['experiment_train_path'], weights_path=None
    )

    # interpolated obs should have same exact chunking schema as ds at `p['obs_full_space_path']`
    p['interpolated_obs_path'] = regrid(
        source_path=p['coarse_obs_path'],
        target_grid_path=p['obs_path'],
        weights_path=None,
    )

    p['interpolated_obs_full_time_path'] = rechunk(
        path=p['interpolated_obs_path'], pattern="full_time"
    )
    p['obs_full_time_path'] = rechunk(path=p['obs_path'], pattern="full_time")
    p['spatial_anomalies_path'] = spatial_anomalies(
        p['obs_full_time_path'], p['interpolated_obs_full_time_path']
    )
    p['coarse_obs_full_time_path'] = rechunk(p['coarse_obs_path'], pattern='full_time')
    p['experiment_train_full_time_path'] = rechunk(p['experiment_train_path'], pattern='full_time')

    p['experiment_predict_full_time_path'] = rechunk(
        p['experiment_predict_path'],
        pattern='full_time',
        template=p['coarse_obs_full_time_path'],
    )
    p['bias_corrected_path'] = fit_and_predict(
        experiment_train_full_time_path=p['experiment_train_full_time_path'],
        experiment_predict_full_time_path=p['experiment_predict_full_time_path'],
        coarse_obs_full_time_path=p['coarse_obs_full_time_path'],
        run_parameters=run_parameters,
    )
    p['bias_corrected_full_space_path'] = rechunk(
        p['bias_corrected_path'],
        pattern='full_space',
        template=p['obs_full_space_path'],
    )
    p['bias_corrected_fine_full_space_path'] = regrid(
        source_path=p['bias_corrected_full_space_path'],
        target_grid_path=p['obs_path'],
        weights_path=None,
    )

    p['bias_corrected_fine_full_time_path'] = rechunk(
        p['bias_corrected_fine_full_space_path'],
        pattern='full_time',
        template=p['obs_full_time_path'],
    )
    p['final_bcsd_full_time_path'] = postprocess_bcsd(
        p['bias_corrected_fine_full_time_path'], p['spatial_anomalies_path']
    )  # fine-scale maps (full_space) (time: 365)

    # temporary aggregations - these come out in full time
    p['monthly_summary_path'] = time_summary(p['final_bcsd_full_time_path'], freq='1MS')
    p['annual_summary_path'] = time_summary(p['final_bcsd_full_time_path'], freq='1AS')

    # analysis notebook
    # analysis_location = run_analyses(p['final_bcsd_full_time_path'], run_parameters)

    # since pyramids require full space we now rechunk everything into full
    # space before passing into pyramid step. we probably want to add a cleanup
    # to this step in particular since otherwise we will have an exact
    # duplicate of the daily, monthly, and annual datasets
    p['final_bcsd_full_space_path'] = rechunk(p['final_bcsd_full_time_path'], pattern='full_space')

    # make temporal summaries
    p['monthly_summary_full_space_path'] = rechunk(p['monthly_summary_path'], pattern='full_space')
    p['annual_summary_full_space_path'] = rechunk(p['annual_summary_path'], pattern='full_space')

    # pyramids

    p['pyramid_weights'] = get_pyramid_weights(run_parameters=run_parameters, levels=4)

    p['daily_pyramid_path'] = pyramid(
        p['final_bcsd_full_space_path'], weights_pyramid_path=p['pyramid_weights'], levels=4
    )
    p['monthly_pyramid_path'] = pyramid(
        p['monthly_summary_full_space_path'], weights_pyramid_path=p['pyramid_weights'], levels=4
    )
    p['annual_pyramid_path'] = pyramid(
        p['annual_summary_full_space_path'], weights_pyramid_path=p['pyramid_weights'], levels=4
    )

    # finalize
    finalize(p, run_parameters)
