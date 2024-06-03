<p align="left" >
<a href='https://carbonplan.org'>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://carbonplan-assets.s3.amazonaws.com/monogram/light-small.png">
  <img alt="CarbonPlan monogram." height="48" src="https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png">
</picture>
</a>
</p>

# carbonplan / cmip6-downscaling

**climate downscaling using cmip6 data**

[![CI](https://github.com/carbonplan/cmip6-downscaling/actions/workflows/main.yaml/badge.svg)](https://github.com/carbonplan/cmip6-downscaling/actions/workflows/main.yaml)
[![PyPI](https://img.shields.io/pypi/v/cmip6-downscaling)](https://pypi.org/project/cmip6-downscaling)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<img
src='https://images.carbonplan.org/highlights/cmip6-downscaling-dark.png'
/>

This repository includes our tools/scripts/models/etc for climate downscaling. This work is described in more detail in a [web article](https://carbonplan.org/research/cmip6-downscaling-explainer) with
a companion [map tool](https://carbonplan.org/research/cmip6-downscaling) to explore the data. We encourage you to reach out if you are interested in using the code or datasets by [opening an issue](https://github.com/carbonplan/cmip6-downscaling/issues/new) or [sending us an email](mailto:hello@carbonplan.org).

## install

```shell
python -m pip install cmip6_downscaling
```

## usage

```python
from cmip6_downscaling.methods import ...
```

## data access

There are two ways to access the data using Python.

First, the entire collection of datasets at daily timescales is available through an `intake` catalog using the following code snippet.

```
import intake
cat = intake.open_esm_datastore(
  'https://rice1.osn.mghpcc.org/carbonplan/cp-cmip/version1/catalog/osn-rechunked-global-downscaled-cmip6.json'
)
```

You can check out this example [Jupyter notebook](https://github.com/carbonplan/cmip6-downscaling/blob/main/notebooks/accessing_data_example.ipynb) to see how to access the data, perform some simple analysis, and download subsets.

You can also access the data by using the URL of an individual dataset. See [the datasets page](https://github.com/carbonplan/cmip6-downscaling/blob/main/datasets.md) for a table of all available datasets in this collection with storage locations and other metadata. A code snippet showing how to use the URL is shown below:

````
import xarray as xr
xr.open_zarr('https://rice1.osn.mghpcc.org/carbonplan/cp-cmip/version1/rechunked_data/DeepSD/ScenarioMIP.CCCma.CanESM5.ssp245.r1i1p1f1.day.DeepSD.pr.zarr',chunks={})```

## license

All the code in this repository is [MIT](https://choosealicense.com/licenses/mit/)-licensed. Some of the data provided by this API is sourced from content made available under a [CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/) license. We include attribution for this content, and we please request that you also maintain that attribution if using this data.

## about us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions with open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/cmip6-downscaling/issues/new) or [sending us an email](mailto:hello@carbonplan.org).
````
