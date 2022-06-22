import warnings

import dask
import xarray as xr
from prefect import Flow, Parameter
from sklearn.utils.validation import DataConversionWarning

from cmip6_downscaling import config, runtimes
from cmip6_downscaling.methods.common.tasks import (
    finalize,
    finalize_on_failure,
    get_experiment,
    get_obs,
    get_pyramid_weights,
    make_run_parameters,
    pyramid,
    rechunk,
    regrid,
    time_summary,
)
from cmip6_downscaling.methods.gard.tasks import coarsen_and_interpolate, fit_and_predict, read_scrf

xr.set_options(keep_attrs=True)
config.set({'run_options.use_cache': False})

dask.config.set({"array.slicing.split_large_chunks": False})
warnings.filterwarnings(
    "ignore",
    "(.*) filesystem path not explicitly implemented. falling back to default implementation. This filesystem may not be tested",
    category=UserWarning,
)
warnings.filterwarnings(
    action='ignore',
    category=DataConversionWarning,
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
    p['obs_full_time_path'] = rechunk(path=p['obs_path'], pattern='full_time')
    p['experiment_train_path'] = get_experiment(run_parameters, time_subset='train_period')
    p['experiment_predict_path'] = get_experiment(run_parameters, time_subset='predict_period')

    # after regridding coarse_obs will have smaller array size in space but still
    # be chunked finely along time. but that's good to get it for regridding back to
    # the interpolated obs in next task
    # interpolated obs should have same exact chunking schema as ds at `obs_full_space_path`
    p['interpolated_obs_full_space_path'] = coarsen_and_interpolate(
        p['obs_full_space_path'], p['experiment_train_path']
    )

    # just allow the interpolated obs full time rechunking determine the size of the subsequent full-time chunking routines
    p['interpolated_obs_full_time_path'] = rechunk(
        p['interpolated_obs_full_space_path'], pattern='full_time'
    )

    # get gcm data into full space to prep for interpolation
    p['experiment_predict_full_space_path'] = rechunk(
        p['experiment_predict_path'], pattern="full_space", template=p['obs_full_space_path']
    )

    # interpolate gcm to finescale. it will retain the same temporal chunking pattern (likely 25 timesteps)
    p['experiment_predict_fine_full_space_path'] = regrid(
        source_path=p['experiment_predict_full_space_path'],
        target_grid_path=p['obs_path'],
        weights_path=None,
    )
    p['experiment_predict_fine_full_time_path'] = rechunk(
        p['experiment_predict_fine_full_space_path'],
        pattern="full_time",
        template=p['interpolated_obs_full_time_path'],
    )

    p['scrf_path'] = read_scrf(p['experiment_predict_fine_full_time_path'], run_parameters)

    p['scrf_full_time_path'] = rechunk(
        p['scrf_path'],
        pattern="full_time",
        template=p['experiment_predict_fine_full_time_path'],
    )

    p['model_output_path'] = fit_and_predict(
        xtrain_path=p['interpolated_obs_full_time_path'],
        ytrain_path=p['obs_full_time_path'],
        xpred_path=p['experiment_predict_fine_full_time_path'],
        scrf_path=p['scrf_full_time_path'],
        run_parameters=run_parameters,
    )

    # temporary aggregations - these come out in full time
    p['monthly_summary_path'] = time_summary(p['model_output_path'], freq='1MS')
    p['annual_summary_path'] = time_summary(p['model_output_path'], freq='1AS')

    # analysis notebook (shared with BCSD)
    # analysis_location = run_analyses(model_output_path, run_parameters)

    if config.get('run_options.generate_pyramids'):

        # since pyramids require full space we now rechunk everything into full
        # space before passing into pyramid step. we probably want to add a cleanup
        # to this step in particular since otherwise we will have an exact
        # duplicate of the daily, monthly, and annual datasets
        p['full_space_model_output_path'] = rechunk(p['model_output_path'], pattern='full_space')

        # make temporal summaries
        p['monthly_summary_full_space_path'] = rechunk(
            p['monthly_summary_path'], pattern='full_space'
        )
        p['annual_summary_full_space_path'] = rechunk(
            p['annual_summary_path'], pattern='full_space'
        )

        # pyramids
        p['pyramid_weights'] = get_pyramid_weights(run_parameters=run_parameters, levels=4)

        # p['daily_pyramid_path'] = pyramid(
        #     p['full_space_model_output_path'], weights_pyramid_path=p['pyramid_weights'], levels=4
        # )
        p['monthly_pyramid_path'] = pyramid(
            p['monthly_summary_full_space_path'],
            weights_pyramid_path=p['pyramid_weights'],
            levels=4,
        )
        p['annual_pyramid_path'] = pyramid(
            p['annual_summary_full_space_path'], weights_pyramid_path=p['pyramid_weights'], levels=4
        )

    # finalize
    ref = finalize(run_parameters=run_parameters, **p)
    finalize_on_failure(run_parameters=run_parameters, **p)

flow.set_reference_tasks([ref])
