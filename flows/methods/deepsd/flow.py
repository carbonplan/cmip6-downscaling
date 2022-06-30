from prefect import Flow, Parameter

from cmip6_downscaling import config, runtimes
from cmip6_downscaling.methods.common.tasks import (
    finalize,
    finalize_on_failure,
    get_experiment,
    get_obs,
    make_run_parameters,
    pyramid,
    rechunk,
    time_summary,
)
from cmip6_downscaling.methods.deepsd.tasks import (
    bias_correction,
    inference,
    normalize_gcm,
    rescale,
    shift,
    update_var_attrs,
)

runtime = runtimes.get_runtime()
print(runtime)

with Flow(
    name="deepsd", storage=runtime.storage, run_config=runtime.run_config, executor=runtime.executor
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
    p['obs_path'] = get_obs(run_parameters)
    p['obs_full_space_path'] = rechunk(path=p['obs_path'], pattern='full_space')
    p['shifted_obs_full_space_path'] = shift(
        path=p['obs_full_space_path'], path_type='obs', run_parameters=run_parameters
    )
    p['shifted_obs_full_time_path'] = rechunk(
        path=p['shifted_obs_full_space_path'], pattern='full_time'
    )

    # # Tasks for running inference on ERA5
    # p['shifted_experiment_train_path'] = coarsen_obs(
    #     path=p['shifted_obs_full_space_path'], output_degree=2.0
    # )
    # p['experiment_predict_path'] = get_validation(run_parameters)
    # p['experiment_predict_full_space_path'] = rechunk(
    #     path=p['experiment_predict_path'], pattern='full_space'
    # )
    # p['shifted_obs_predict_path'] = shift(
    #     path=p['experiment_predict_full_space_path'], path_type='obs', run_parameters=run_parameters
    # )
    # p['shifted_experiment_predict_path'] = coarsen_obs(
    #     p['shifted_obs_predict_path'], output_degree=2.0
    # )

    # Tasks for running inference on gcm
    p['experiment_train_path'] = get_experiment(run_parameters, time_subset='train_period')
    p['experiment_predict_path'] = get_experiment(run_parameters, time_subset='predict_period')
    p['shifted_experiment_predict_path'] = shift(
        path=p['experiment_predict_path'], path_type='gcm', run_parameters=run_parameters
    )
    p['shifted_experiment_train_path'] = shift(
        path=p['experiment_train_path'], path_type='gcm', run_parameters=run_parameters
    )

    ### Common steps for inference on obs or gcm
    p['normalized_shifted_experiment_predict_path'] = normalize_gcm(
        predict_path=p['shifted_experiment_predict_path'],
        historical_path=p['shifted_experiment_train_path'],
    )
    p['normalized_shifted_model_output_path'] = inference(
        gcm_path=p['normalized_shifted_experiment_predict_path'], run_parameters=run_parameters
    )
    p['normalized_shifted_model_output_full_time_path'] = rechunk(
        path=p['normalized_shifted_model_output_path'],
        pattern='full_time',
        template=p['shifted_obs_full_time_path'],
    )
    p['shifted_model_output_path'] = rescale(
        source_path=p['normalized_shifted_model_output_full_time_path'],
        obs_path=p['shifted_obs_full_time_path'],
        run_parameters=run_parameters,
    )
    p['bias_corrected_shifted_model_output_path'] = bias_correction(
        p['shifted_model_output_path'],
        p['shifted_obs_full_time_path'],
        run_parameters=run_parameters,
    )
    # Add attrs from rescaled product to bias corrected product
    p['bias_corrected_shifted_model_output_path'] = update_var_attrs(
        target_path=p['bias_corrected_shifted_model_output_path'],
        source_path=p['shifted_model_output_path'],
        run_parameters=run_parameters,
    )

    # temporary aggregations - these come out in full time
    p['raw_monthly_summary_path'] = time_summary(p['shifted_model_output_path'], freq='1MS')
    p['raw_annual_summary_path'] = time_summary(p['shifted_model_output_path'], freq='1AS')

    p['bias_corrected_monthly_summary_path'] = time_summary(
        p['bias_corrected_shifted_model_output_path'], freq='1MS'
    )
    p['bias_corrected_annual_summary_path'] = time_summary(
        p['bias_corrected_shifted_model_output_path'], freq='1AS'
    )
    # Add attrs from rescaled product to bias corrected product
    p['bias_corrected_monthly_summary_path'] = update_var_attrs(
        target_path=p['bias_corrected_monthly_summary_path'],
        source_path=p['raw_annual_summary_path'],
        run_parameters=run_parameters,
    )
    p['bias_corrected_annual_summary_path'] = update_var_attrs(
        target_path=p['bias_corrected_annual_summary_path'],
        source_path=p['raw_annual_summary_path'],
        run_parameters=run_parameters,
    )

    if config.get('run_options.generate_pyramids'):

        # since pyramids require full space we now rechunk everything into full
        # space before passing into pyramid step. we probably want to add a cleanup
        # to this step in particular since otherwise we will have an exact
        # duplicate of the daily, monthly, and annual datasets
        # p['full_space_model_output_path'] = rechunk(
        #     p['shifted_model_output_path'], pattern='full_space'
        # )

        # make temporal summaries
        p['bias_corrected_monthly_summary_full_space_path'] = rechunk(
            p['bias_corrected_monthly_summary_path'], pattern='full_space'
        )
        p['bias_corrected_annual_summary_full_space_path'] = rechunk(
            p['bias_corrected_annual_summary_path'], pattern='full_space'
        )

        p['raw_monthly_summary_full_space_path'] = rechunk(
            p['raw_monthly_summary_path'], pattern='full_space'
        )
        p['raw_annual_summary_full_space_path'] = rechunk(
            p['raw_annual_summary_path'], pattern='full_space'
        )
        # Add attrs from rescaled product to bias corrected product
        p['bias_corrected_monthly_summary_full_space_path'] = update_var_attrs(
            target_path=p['bias_corrected_monthly_summary_full_space_path'],
            source_path=p['raw_monthly_summary_full_space_path'],
            run_parameters=run_parameters,
        )
        p['bias_corrected_annual_summary_full_space_path'] = update_var_attrs(
            target_path=p['bias_corrected_annual_summary_full_space_path'],
            source_path=p['raw_annual_summary_full_space_path'],
            run_parameters=run_parameters,
        )

        # pyramids
        p['bias_corrected_monthly_pyramid_path'] = pyramid(
            p['bias_corrected_monthly_summary_full_space_path'], levels=4
        )
        p['bias_corrected_annual_pyramid_path'] = pyramid(
            p['bias_corrected_annual_summary_full_space_path'], levels=4
        )

        p['raw_monthly_pyramid_path'] = pyramid(p['raw_monthly_summary_full_space_path'], levels=4)
        p['raw_annual_pyramid_path'] = pyramid(p['raw_annual_summary_full_space_path'], levels=4)

    # finalize
    finalize(run_parameters=run_parameters, **p)
    finalize_on_failure(run_parameters=run_parameters, **p)
