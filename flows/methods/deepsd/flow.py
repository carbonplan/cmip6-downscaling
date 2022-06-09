from prefect import Flow, Parameter

from cmip6_downscaling.methods.common.tasks import get_obs, make_run_parameters
from cmip6_downscaling.methods.deepsd.tasks import shift_coarsen_interpolate

with Flow(name="deepsd") as flow:

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

    p = {}
    p['obs_path'] = get_obs(run_parameters)
    p['obs_shifted'] = shift_coarsen_interpolate(path=p['obs_path'], run_parameters=run_parameters)
