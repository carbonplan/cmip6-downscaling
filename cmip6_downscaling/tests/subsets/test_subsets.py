import json

import fsspec
import pytest

from cmip6_downscaling.workflows.bcsd_flow import bcsd_flow

# TODO: import other downscaling method flows

run_config = config.get_config()

list_of_keys = ['full_time', 'full_space']


@pytest.mark.parametrize("subset", list_of_keys)
def test_bcsd_flow_subset(subset):
    with open(f'{subset}_test_bcsd_params.json') as json_file:
        run_hyperparameters = json.load(json_file)

    gcm = run_hyperparameters["GCM"]
    scenario = run_hyperparameters["SCENARIO"]
    train_period_start = run_hyperparameters["TRAIN_PERIOD_START"]
    train_period_end = run_hyperparameters["TRAIN_PERIOD_END"]
    predict_period_start = run_hyperparameters["PREDICT_PERIOD_START"]
    predict_period_end = run_hyperparameters["PREDICT_PERIOD_END"]
    variable = run_hyperparameters["VARIABLE"]
    latmin = run_hyperparameters["LATMIN"]
    latmax = run_hyperparameters["LATMAX"]
    lonmin = run_hyperparameters["LONMIN"]
    lonmax = run_hyperparameters["LONMAX"]

    target_naming_str = f"{gcm}-{scenario}-{train_period_start}-{train_period_end}-{predict_period_start}-{predict_period_end}-{latmin}-{latmax}-{lonmin}-{lonmax}-{variable}.zarr"

    bcsd_flow.run(parameters=run_hyperparameters)
    # check for completed files? or check some output in the analysis script?
    # or just check that it doesn't fail?

    assert fsspec.open(
        run_config.results_cache_path + f'/postprocess-results-{target_naming_str}/' + '.zmetadata'
    ).read()


# TODO: test_gard_flow_subset(subset):
#
