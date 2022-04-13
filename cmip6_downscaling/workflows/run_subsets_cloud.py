import glob
import os

# --- Settings ---
# downscaling_methods = ['bcsd', 'gard', 'maca']
downscaling_methods = ['bcsd']

_prefect_register_str = (
    """prefect register --project "envs" -p ../methods/{downscaling_method}/flow.py"""
)
_prefect_run_str = """prefect run -i "{flow_run_id}" --param-file {param_file}"""

# --- Funcs ---


def retrieve_test_parms():
    """retrieve list of all .json param files in method subdir"""
    return glob.glob('../configs/*.json')


def register_flow():
    """Register flow with prefect cloud and return flow_run_id for running flows"""
    print('registering flow on prefect cloud')
    return os.popen(_prefect_register_str).read().split("ID: ")[1].split("\n")[0]


def run_flows(downscaling_methods, json_list):
    """Iterate through possible tests and run in prefect cloud (parallel-ish)"""
    for method in downscaling_methods:
        flow_id = (
            os.popen(_prefect_register_str.format(downscaling_method=method))
            .read()
            .split("ID: ")[1]
            .split("\n")[0]
        )
        print(f'running available flow parameters for {flow_id}')
        for test_fil in json_list:
            print(test_fil)
            sys_output = os.popen(
                _prefect_run_str.format(flow_run_id=flow_id, param_file=test_fil)
            ).read()
            print(sys_output)


def main():
    json_list = retrieve_test_parms()
    run_flows(downscaling_methods, json_list)


main()
