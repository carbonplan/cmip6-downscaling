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
    get_region_numbers,
    replace_epoch_trend,
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
    p['obs_full_time_path'] = rechunk(path=p['obs_path'], pattern='full_time')

    p['obs_full_space_path'] = rechunk(path=p['obs_path'], pattern='full_space')
    p['experiment_path'] = get_experiment(run_parameters, time_subset='both')

    # get coarsened resolution observations
    # this coarse obs is going to be used in bias correction next, so rechunk into full time first
    p['coarse_obs_full_space_path'] = regrid(
        p['obs_full_space_path'], p['experiment_path'], weights_path=p['obs_to_gcm_weights']
    )

    ## Step 2: Epoch Adjustment -- all variables undergo this epoch adjustment
    # TODO: in order to properly do a 31 year average, might need to run this step with the entire future period in GCMs

    p['coarse_epoch_trend_path'], p['detrended_data_path'] = epoch_trend(
        p['experiment_path'], run_parameters
    )

    p['coarse_epoch_trend_full_space_path'] = rechunk(
        p['coarse_epoch_trend_path'], pattern='full_space'
    )

    p['fine_epoch_trend_full_space_path'] = regrid(
        p['coarse_epoch_trend_full_space_path'],
        p['obs_full_space_path'],
        weights_path=p['gcm_to_obs_weights'],
        pre_chunk_def={'time': 30},
    )

    p['coarse_obs_full_time_path'] = rechunk(
        p['coarse_obs_full_space_path'], pattern='full_time', template=p['detrended_data_path']
    )

    # get gcm
    # 1981-2100 extent time subset
    p['experiment_predict_full_time_path'] = rechunk(p['experiment_path'], pattern='full_time')

    ## Step 3: Coarse Bias Correction
    # rechunk to make detrended data match the coarse obs
    p['detrend_gcm_full_time'] = rechunk(
        p['detrended_data_path'],
        template=p[
            'coarse_obs_full_time_path'
        ],  # this is not working. time is chunked to match coarse obs
    )

    # inputs should be in full-time
    p['bias_corrected_gcm_full_time_path'] = bias_correction(
        x_path=p['detrend_gcm_full_time'],
        y_path=p['coarse_obs_full_time_path'],
        run_parameters=run_parameters,
    )

    # rechunk into full space and cache the output
    # p['bias_corrected_gcm_full_space_path'] = rechunk(
    #     p['bias_corrected_gcm_full_time_path'], pattern='full_space'
    # )

    p['region_numbers'] = get_region_numbers()

    p['bias_corrected_gcm_region_paths'] = split_by_region.map(
        p['region_numbers'], unmapped(p['bias_corrected_gcm_full_time_path'])
    )
    p['coarse_obs_region_paths'] = split_by_region.map(
        p['region_numbers'], unmapped(p['coarse_obs_full_time_path'])
    )
    p['obs_region_paths'] = split_by_region.map(
        p['region_numbers'], unmapped(p['obs_full_time_path'])
    )

    # Step 4: Constructed Analogs
    # Note: This is option was added to allow this step to be run on the local executor while the rest of the steps can be run with
    # The dask executor
    if config.get('run_options.construct_analogs'):
        p['constructed_analogs_region_paths'] = construct_analogs.map(
            gcm_path=p['bias_corrected_gcm_region_paths'],
            coarse_obs_path=p['coarse_obs_region_paths'],
            fine_obs_path=p['obs_region_paths'],
            run_parameters=unmapped(run_parameters),
        )

        if config.get('run_options.combine_regions'):

            p['combined_analogs_full_time_path'] = combine_regions(
                regions=p['region_numbers'],
                region_paths=p['constructed_analogs_region_paths'],
                template_path=p['fine_epoch_trend_full_space_path'],
            )

            p['fine_epoch_trend_full_time_path'] = rechunk(
                p['fine_epoch_trend_full_space_path'],
                pattern='full_time',
                template=p['combined_analogs_full_time_path'],
            )

            # Step 5: Epoch Replacement
            p['epoch_replaced_full_time_path'] = replace_epoch_trend(
                p['combined_analogs_full_time_path'], p['fine_epoch_trend_full_time_path']
            )

            # Step 6: Fine Bias Correction
            p['final_bias_corrected_full_time_path'] = bias_correction(
                p['epoch_replaced_full_time_path'],
                p['obs_full_time_path'],
                run_parameters=run_parameters,
            )

            # temporary aggregations - these come out in full time
            p['monthly_summary_path'] = time_summary(
                p['final_bias_corrected_full_time_path'], freq='1MS'
            )
            p['annual_summary_path'] = time_summary(
                p['final_bias_corrected_full_time_path'], freq='1AS'
            )

            # analysis notebook
            # analysis_location = run_analyses(p['final_bias_corrected_full_time_path'], run_parameters)

            if config.get('run_options.generate_pyramids'):

                # make temporal summaries
                p['monthly_summary_full_space_path'] = rechunk(
                    p['monthly_summary_path'], pattern='full_space'
                )
                p['annual_summary_full_space_path'] = rechunk(
                    p['annual_summary_path'], pattern='full_space'
                )

                # pyramids
                p['pyramid_weights'] = get_pyramid_weights(run_parameters=run_parameters, levels=4)

                p['monthly_pyramid_path'] = pyramid(
                    p['monthly_summary_full_space_path'],
                    weights_pyramid_path=p['pyramid_weights'],
                    levels=4,
                )
                p['annual_pyramid_path'] = pyramid(
                    p['annual_summary_full_space_path'],
                    weights_pyramid_path=p['pyramid_weights'],
                    levels=4,
                )

    # finalize
    ref = finalize(run_parameters=run_parameters, **p)
    finalize_on_failure(run_parameters=run_parameters, **p)

flow.set_reference_tasks([ref])
