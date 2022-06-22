from prefect import Flow, Parameter, unmapped

from cmip6_downscaling import config, runtimes
from cmip6_downscaling.methods.common.tasks import (  # run_analyses,
    finalize,
    finalize_on_failure,
    get_experiment,
    get_obs,
    get_pyramid_weights,
    get_weights,
    make_run_parameters,
    pyramid,
    rechunk,
    regrid,
    time_summary,
)
from cmip6_downscaling.methods.maca.tasks import (
    bias_correction,
    combine_regions,
    construct_analogs,
    epoch_trend,
    split_by_region,
)

runtime = runtimes.get_runtime()
print(runtime)


with Flow(
    name='maca',
    storage=runtime.storage,
    run_config=runtime.run_config,
    executor=runtime.executor,
) as flow:
    # following https://climate.northwestknowledge.net/MACA/MACAmethod.php
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
        train_dates=Parameter("train_dates"),
        predict_dates=Parameter("predict_dates"),
        year_rolling_window=Parameter("year_rolling_window"),
        day_rolling_window=Parameter("day_rolling_window"),
    )

    p = {}

    ## Step 0: tasks to get inputs and set up
    ## Step 1: Common Grid -- this step is skipped since it seems like an unnecessary extra step for convenience

    # input datasets
    p['gcm_to_obs_weights'] = get_weights(run_parameters=run_parameters, direction='gcm_to_obs')
    p['obs_to_gcm_weights'] = get_weights(run_parameters=run_parameters, direction='obs_to_gcm')

    # get original resolution observations
    p['obs_path'] = get_obs(run_parameters)

    p['obs_full_space_path'] = rechunk(path=p['obs_path'], pattern='full_space')
    p['experiment_path'] = get_experiment(run_parameters, time_subset='both')

    # get coarsened resolution observations
    # this coarse obs is going to be used in bias correction next, so rechunk into full time first
    p['coarse_obs_full_space_path'] = regrid(
        p['obs_full_space_path'], p['experiment_path'], weights_path=p['obs_to_gcm_weights']
    )
    p['coarse_obs_full_time_path'] = rechunk(p['coarse_obs_full_space_path'], pattern='full_time')

    ## Step 2: Epoch Adjustment -- all variables undergo this epoch adjustment
    # TODO: in order to properly do a 31 year average, might need to run this step with the entire future period in GCMs
    # but this might be too memory intensive in the later task
    # BIG JOB
    p['coarse_epoch_trend_path'], p['detrended_data_path'] = epoch_trend(
        p['experiment_path'], run_parameters
    )
    p['fine_epoch_trend_path'] = regrid(
        p['coarse_epoch_trend_path'], p['obs_full_space_path'], weights_path=p['gcm_to_obs_weights']
    )

    # get gcm
    # 1981-2100 extent time subset
    p['experiment_predict_full_time_path'] = rechunk(p['experiment_path'], pattern='full_time')

    ## Step 3: Coarse Bias Correction
    # rechunk to make detrended data match the coarse obs
    p['detrend_gcm_full_time'] = rechunk(
        p['detrended_data_path'], template=p['coarse_obs_full_time_path']
    )

    # inputs should be in full-time
    p['bias_corrected_gcm_full_time_path'] = bias_correction(
        x_path=p['detrended_data_path'],
        y_path=p['coarse_obs_full_time_path'],
        run_parameters=run_parameters,
    )

    # rechunk into full space and cache the output
    p['bias_corrected_gcm_full_space_path'] = rechunk(
        p['bias_corrected_gcm_full_time_path'], pattern='full_space'
    )

    # # everything should be rechunked to full space and then subset
    p['bias_corrected_gcm_region_paths'] = split_by_region(p['bias_corrected_gcm_full_space_path'])
    p['coarse_obs_region_paths'] = split_by_region(p['coarse_obs_full_space_path'])
    p['obs_region_paths'] = split_by_region(p['obs_full_space_path'])
    p['find_trend_paths'] = split_by_region(p['coarse_epoch_trend_path'])

    # Step 4: Constructed Analogs
    # Step 5: Epoch Replacement
    p['constructed_analogs_region_paths'] = construct_analogs.map(
        gcm_path=p['bias_corrected_gcm_region_paths'],
        coarse_obs_path=p['coarse_obs_region_paths'],
        fine_obs_path=p['obs_region_paths'],
        fine_trend_path=p['find_trend_paths'],
        run_parameters=unmapped(run_parameters),
    )

    # Step 6: Fine Bias Correction
    p['final_maca_regions_paths'] = bias_correction.map(
        p['constructed_analogs_region_paths'], p['obs_region_paths'], run_parameters=run_parameters
    )

    p['full_grid_detrended_path'] = combine_regions(
        p['final_maca_regions_paths'], p['fine_obs_path']
    )

    # temporary aggregations - these come out in full time
    p['monthly_summary_path'] = time_summary(p['final_maca_full_time_path'], freq='1MS')
    p['annual_summary_path'] = time_summary(p['final_maca_full_time_path'], freq='1AS')

    # analysis notebook
    # analysis_location = run_analyses(p['final_maca_full_time_path'], run_parameters)

    if config.get('run_options.generate_pyramids'):

        p['final_maca_full_space_path'] = rechunk(
            p['final_maca_full_time_path'], pattern='full_space'
        )

        # make temporal summaries
        p['monthly_summary_full_space_path'] = rechunk(
            p['monthly_summary_path'], pattern='full_space'
        )
        p['annual_summary_full_space_path'] = rechunk(
            p['annual_summary_path'], pattern='full_space'
        )

        # pyramids
        p['pyramid_weights'] = get_pyramid_weights(run_parameters=run_parameters, levels=4)

        p['daily_pyramid_path'] = pyramid(
            p['final_maca_full_space_path'], weights_pyramid_path=p['pyramid_weights'], levels=4
        )
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
