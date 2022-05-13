from prefect import Flow, Parameter

from cmip6_downscaling import runtimes
from cmip6_downscaling.methods.common.tasks import (  # run_analyses,
    finalize,
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
) as maca_flow:
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
        train_dates=Parameter("train_period"),
        predict_dates=Parameter("predict_period"),
    )

    p = {}

    ## Step 0: tasks to get inputs and set up
    ## Step 1: Common Grid -- this step is skipped since it seems like an unnecessary extra step for convenience

    # input datasets
    # p['gcm_to_obs_weights'] = get_weights(run_parameters=run_parameters, direction='gcm_to_obs')
    p['obs_to_gcm_weights'] = get_weights(run_parameters=run_parameters, direction='obs_to_gcm')

    # get original resolution observations
    p['obs_path'] = get_obs(run_parameters)

    p['obs_full_space_path'] = rechunk(path=p['obs_path'], pattern='full_space')
    p['experiment_train_path'] = get_experiment(run_parameters, time_subset='train_period')
    p['experiment_predict_path'] = get_experiment(run_parameters, time_subset='predict_period')

    # get coarsened resolution observations
    # this coarse obs is going to be used in bias correction next, so rechunk into full time first
    p['coarse_obs_path'] = regrid(
        p['obs_full_space_path'], p['experiment_train_path'], weights_path=p['obs_to_gcm_weights']
    )

    ## Step 2: Epoch Adjustment -- all variables undergo this epoch adjustment
    # TODO: in order to properly do a 31 year average, might need to run this step with the entire future period in GCMs
    # but this might be too memory intensive in the later task
    # BIG JOB
    p['coarse_epoch_trend_path'], p['detrended_data_path'] = epoch_trend(
        p['experiment_train_path'], run_parameters
    )

    # get gcm
    # 1981-2100 extent time subset
    p['experiment_predict_full_time_path'] = rechunk(
        p['experiment_predict_path'], pattern='full_time'
    )

    ## Step 3: Coarse Bias Correction
    # rechunk to make detrended data match the coarse obs
    # TODO: check whether we want full-time AND template (or just template)
    p['detrend_gcm_rechunk_coarse_obs_path'] = rechunk(
        p['experiment_predict_path'], template=p['coarse_obs_path']
    )

    # inputs should be in full-time
    p['bias_corrected_gcm_path'] = bias_correction(
        gcm_path=p['detrended_data_path'],
        obs_path=p['coarse_obs_path'],
        run_parameters=run_parameters,
    )

    # do epoch adjustment again for multiplicative variables, see MACA v1 vs. v2 guide for details
    # if label in ['pr', 'huss', 'vas', 'uas']:
    #     coarse_epoch_trend_2 = epoch_trend_task(
    #         data=bias_corrected_gcm,
    #         train_period_start=train_period_start,
    #         train_period_end=train_period_end,
    #         day_rolling_window=epoch_adjustment_day_rolling_window,
    #         year_rolling_window=epoch_adjustment_year_rolling_window,
    #         gcm_identifier=f'{gcm_identifier}_2',
    #     )

    #     bias_corrected_gcm = remove_epoch_trend_task(
    #         data=bias_corrected_gcm,
    #         trend=coarse_epoch_trend_2,
    #         day_rolling_window=epoch_adjustment_day_rolling_window,
    #         year_rolling_window=epoch_adjustment_year_rolling_window,
    #         gcm_identifier=f'{gcm_identifier}_2',
    #     )

    ## Step 4: Constructed Analogs
    # rechunk into full space and cache the output
    p['bc_gcm_full_space'] = rechunk(p['bc_gcm'], pattern='full_space')

    # everything should be rechunked to full space and then subset
    p['coarse_obs_path'] = regrid(
        p['obs_full_space_path'], p['experiment_train_path'], weights_path=p['obs_to_gcm_weights']
    )

    p['bc_gcm_region_paths'] = split_by_region(p['bc_gcm_full_space_path'])
    p['coarse_obs_region_paths'] = split_by_region(p['coarse_obs_path'])
    p['fine_obs_region_paths'] = split_by_region(p['fine_obs_path'])

    p['constructed_analogs_path'] = construct_analogs.map(
        p['bc_gcm_region_paths'], p['coarse_obs_region_paths'], p['fine_obs_region_paths']
    )

    p['full_grid_detrended_path'] = combine_regions(
        p['constructed_analogs_path'], p['fine_obs_path']
    )

    # combine these two?
    # p['full_grid_path'] = maca_reapply_trend(p['full_grid_detrended_path'])
    # p['final_full_grid_path'] = bias_correct_final( p['full_grid_path'])

    # all inputs into the map function needs to be a list
    # ds_gcm_list, ds_obs_coarse_list, ds_obs_fine_list = subset_task(
    #     ds_gcm=bias_corrected_gcm_full_space,
    #     ds_obs_coarse=ds_obs_coarse_full_space,
    #     ds_obs_fine=ds_obs_full_space,
    #     subdomains_list=subdomains_list,
    # )

    # downscaling by constructing analogs
    # downscaled_gcm_list = maca_construct_analogs_task.map(
    #     ds_gcm=ds_gcm_list,
    #     ds_obs_coarse=ds_obs_coarse_list,
    #     ds_obs_fine=ds_obs_fine_list,
    #     subdomain_bound=subdomains_list,
    #     n_analogs=[constructed_analog_n_analogs] * n_subdomains,
    #     doy_range=[constructed_analog_doy_range] * n_subdomains,
    #     gcm_identifier=[gcm_identifier] * n_subdomains,
    #     label=[label] * n_subdomains,
    # )

    # combine back into full domain
    # combined_downscaled_output = combine_outputs_task(
    #     ds_list=downscaled_gcm_list,
    #     subdomains_dict=subdomains_dict,
    #     mask=mask,
    #     gcm_identifier=gcm_identifier,
    #     label=label,
    # )

    # ## Step 5: Epoch Replacement
    # if label in ['pr', 'huss', 'vas', 'uas']:
    #     combined_downscaled_output = maca_epoch_replacement_task(
    #         ds_gcm_fine=combined_downscaled_output,
    #         trend_coarse=coarse_epoch_trend_2,
    #         day_rolling_window=epoch_adjustment_day_rolling_window,
    #         year_rolling_window=epoch_adjustment_year_rolling_window,
    #         gcm_identifier=f'{gcm_identifier}_2',
    #     )

    # epoch_replaced_gcm = maca_epoch_replacement_task(
    #     ds_gcm_fine=combined_downscaled_output,
    #     trend_coarse=coarse_epoch_trend,
    #     day_rolling_window=epoch_adjustment_day_rolling_window,
    #     year_rolling_window=epoch_adjustment_year_rolling_window,
    #     gcm_identifier=gcm_identifier,
    # )

    ## Step 6: Fine Bias Correction
    p['final_maca_full_time_path'] = bias_correction()

    # final_output = maca_fine_bias_correction_task(
    #     ds_gcm=epoch_replaced_gcm,
    #     ds_obs=ds_obs_full_time,
    #     train_period_start=train_period_start,
    #     train_period_end=train_period_end,
    #     variables=[label],
    #     batch_size=bias_correction_batch_size,
    #     buffer_size=bias_correction_buffer_size,
    #     gcm_identifier=gcm_identifier,
    # )

    # temporary aggregations - these come out in full time
    p['monthly_summary_path'] = time_summary(p['final_maca_full_time_path'], freq='1MS')
    p['annual_summary_path'] = time_summary(p['final_maca_full_time_path'], freq='1AS')

    # analysis notebook
    # analysis_location = run_analyses(p['final_maca_full_time_path'], run_parameters)

    # since pyramids require full space we now rechunk everything into full
    # space before passing into pyramid step. we probably want to add a cleanup
    # to this step in particular since otherwise we will have an exact
    # duplicate of the daily, monthly, and annual datasets
    p['final_maca_full_space_path'] = rechunk(p['final_maca_full_time_path'], pattern='full_space')

    # make temporal summaries
    p['monthly_summary_full_space_path'] = rechunk(p['monthly_summary_path'], pattern='full_space')
    p['annual_summary_full_space_path'] = rechunk(p['annual_summary_path'], pattern='full_space')

    # pyramids
    p['pyramid_weights'] = get_pyramid_weights(run_parameters=run_parameters, levels=4)

    p['daily_pyramid_path'] = pyramid(
        p['final_maca_full_space_path'], weights_pyramid_path=p['pyramid_weights'], levels=4
    )
    p['monthly_pyramid_path'] = pyramid(
        p['monthly_summary_full_space_path'], weights_pyramid_path=p['pyramid_weights'], levels=4
    )
    p['annual_pyramid_path'] = pyramid(
        p['annual_summary_full_space_path'], weights_pyramid_path=p['pyramid_weights'], levels=4
    )

    # finalize
    finalize(p, run_parameters)
