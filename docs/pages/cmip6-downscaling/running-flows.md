import Section from '../../components/section'

# Running Flows

In this project each downscaling method [BCSD, GARD, MACA, DEEPSD] has it's own workflow for generating results. These data production workflows are handled by the python library, prefect, which encapsulates the data processing steps into individual tasks, which are organized into a 'Flow'.

Prefect allows us to run these downscaling flows with many different parameter combinations (gcms, observations, training period, prediction period) without modifying the specific downscaling method script.

## Choosing a Runtime

Prefect has the ability to run flows with different `runtimes`. Choosing the correct runtime can be crucial to help with scaling multiple flows or debugging a single issue.

Pre-configured runtimes are stored in [`cmip6_downscaling.runtimes.py`](https://github.com/carbonplan/cmip6-downscaling/blob/main/cmip6_downscaling/runtimes.py)

The current runtime options are:

[`cloud`](https://github.com/carbonplan/cmip6-downscaling/blob/a0379110c33b557f959a1d6fa53e9f93891a45b3/cmip6_downscaling/runtimes.py#L57) `executor: dask-distrubted` - Runtime for queuing multiple flows on prefect cloud.

[`local`](https://github.com/carbonplan/cmip6-downscaling/blob/a0379110c33b557f959a1d6fa53e9f93891a45b3/cmip6_downscaling/runtimes.py#L113) `executor: local` - Runtime for developing on local machine

[`CI`](https://github.com/carbonplan/cmip6-downscaling/blob/a0379110c33b557f959a1d6fa53e9f93891a45b3/cmip6_downscaling/runtimes.py#L130) `executor: local` - Runtime used for Continuous Integration

[`pangeo`](https://github.com/carbonplan/cmip6-downscaling/blob/a0379110c33b557f959a1d6fa53e9f93891a45b3/cmip6_downscaling/runtimes.py#L140) `executor: dask-distrubted` - Runtime for processing on jupyter-hub

## Modifying Flow Config

Project level configuration settings are in [`cmip6_downscaling.config.py`](https://github.com/carbonplan/cmip6-downscaling/blob/main/cmip6_downscaling/config.py) and configured using the python package [`donfig`](https://donfig.readthedocs.io/en/latest/). Default configuration options can be overwritten in multiple ways with donfig. Below are two options for specifying use of the cloud runtime. Note: any `connection_strings` or other sensitive information is best stored in a local .yaml or as an environment variable.

#### Python Context

[donfig python context configuration options](https://donfig.readthedocs.io/en/latest/configuration.html#directly-within-python)

In a python context with `config.set()`. ex:

```python

config.set({'runtime': 'cloud'})

```

#### yaml

[donfig yaml configuration options](https://donfig.readthedocs.io/en/latest/configuration.html#specify-configuration). Default config options can be overwritten with a top-level yaml file. Details on setup are provided in the donfig docs above.

```yaml
run_options:

runtime: 'cloud'
```

#### Environment Variables

Config options can also be set with specifically formatted environment variables. Details can be found below.

[environment variables](https://donfig.readthedocs.io/en/latest/configuration.html#environment-variables)

## Parameter Files

All downscaling flows require run parameters to be passed in as a `.json file`. These parameter files contain arguments to the flows, specifying which downscaling method, which variable etc. Example config files can be found in `cmip6_downscaling.configs.generate_valid_configs.<method>`. Future configs can be generated manually or using the notebook template [generate_valid_json_parameters.ipynb](https://github.com/carbonplan/cmip6-downscaling/blob/main/configs/generate_valid_configs/generate_valid_json_parameters.ipynb).

Example config file:

```json
{
  "method": "gard",
  "obs": "ERA5",
  "model": "BCC-CSM2-MR",
  "member": "r1i1p1f1",
  "grid_label": "gn",
  "table_id": "day",
  "scenario": "historical",
  "features": ["pr"],
  "variable": "pr",
  "latmin": "-90",
  "latmax": "90",
  "lonmin": "-180",
  "lonmax": "180",
  "bias_correction_method": "quantile_mapper",
  "bias_correction_kwargs": {
    "pr": { "detrend": false },
    "tasmin": { "detrend": true },
    "tasmax": { "detrend": true },
    "psl": { "detrend": false },
    "ua": { "detrend": false },
    "va": { "detrend": false }
  },
  "model_type": "PureRegression",
  "model_params": { "thresh": 0 },
  "train_dates": ["1981", "2010"],
  "predict_dates": ["1950", "2014"]
}
```

## Runtimes

### Cloud

**Use Cases: Parallel Production Runs**

The Cloud runtime uses a [dask executor](https://docs.prefect.io/api/latest/executors.html#daskexecutor) with kubernetes as an orchestrator paired to cloud storage to run multiple parallel flows with the ability to scale worker resources and machines to match the flow compute demands.

This environment is meant for parallel production runs with multiple parameter files. The prefect cloud dashboard allows a user to monitor flows in real time and inspect flow run diagrams.

While this runtime excels at resource scaling and parallel runs, debugging with it can be difficult and unexpected worker resource errors have been known to crop up. If errors are found in a flow, dropping down to a `pangeo` or `local` runtime may help.

#### Registering a Flow

With the prefect cloud runtime selected, flows can be registered and run with the prefect [CLI](https://docs.prefect.io/orchestration/concepts/cli.html).

To register a flow:

```bash

#prefect register --project "<project name>" -p <python file for prefect flow>

prefect register --project "cmip6" -p flow.py

```

This should output information about the flow, including ID.

```bash

Collecting flows...

<class 'cmip6_downscaling.runtimes.CloudRuntime'>

Storage    : <class 'prefect.storage.azure.Azure'>

Run Config : <class 'prefect.run_configs.kubernetes.KubernetesRun'>

Executor   : <class 'prefect.executors.dask.DaskExecutor'>

Processing 'flow.py':

Building `Azure` storage...

Registering 'bcsd'... Done

└── ID: e4d94ccd-a3f7-4024-8944-3b5b65914372

└── Version: 1

```

#### Running a Flow

Once the flow is registered, multiple flows can be run with different parameter files.

To run a flow:

```bash

#prefect run -p <python file for prefect flow> --param-file <path to json parameter file> --watch (optional)

prefect run -p flow.py --param-file bcsd_ERA5_ACCESS-CM2_ssp585_tasmax_-90_90_-180_180_1981_2010_2015_2099.json

```

Running a prefect cloud flow will output a url to the prefect dashboard where the flow status can be watched.

### Pangeo

**Usecases: Individual Runs - Detailed Logs - Debugging**

The Pangeo runtime is meant for running prefect flows on a jupyter-hub cloud instance. This runtime is great for individual runs, debugging flow issues and getting detailed real-time logs via the dask dashboard. It is important to know your expected resource usage when selecting which size of jupyter-hub you are using.

#### Running a Flow

With the pangeo runtime selected, flows can be run using the prefect CLI. Unlike the cloud runtime, flows are not registered with prefect cloud.

To run a flow:

```bash

#prefect run -p <python file for prefect flow> --param-file <path to json parameter file> --watch (optional)

prefect run -p flow.py --param-file bcsd_ERA5_ACCESS-CM2_ssp585_tasmax_-90_90_-180_180_1981_2010_2015_2099.json

```

#### Dask Dashboard

One of the benefits of using the pangeo runtime is the easy access to the dask dashboard. In the dashboard, you can monitor memory and cpu usage, track task progress in a flow and see multiple visualizations of the runs progress.

Once your flow is running, navigate to this url.

Note: This url is specific to username, jupyter-hub name and port.

`https://prod.azure.carbonplan.2i2c.cloud/user/<username>/<jupyter-hub name>/proxy/<port>/status`

ex:

`https://prod.azure.carbonplan.2i2c.cloud/user/norlandrhagen/bcsd/proxy/8787/status`

export default ({ children }) => <Section name='Running Prefect Flows'>{children}</Section>
