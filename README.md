<img
  src='https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png'
  height='48'
/>

# carbonplan / cmip6-downscaling

**climate downnscaling using cmip6 data**

_Note: This project is under active development. We expect to make many breaking changes to the utilities and APIs included in this repository. Feel free to look around, but use at your own risk._

[![GitHub][github-badge]][github]
[![Build Status]][actions]
![MIT License][]

[github]: https://github.com/carbonplan/cmip6-downscaling
[github-badge]: https://badgen.net/badge/-/github?icon=github&label
[build status]: https://github.com/carbonplan/cmip6-downscaling/actions/workflows/main.yaml/badge.svg
[actions]: https://github.com/carbonplan/cmip6-downscaling/actions/workflows/main.yaml
[mit license]: https://badgen.net/badge/license/MIT/blue

This repository includes our tools/scripts/models/etc for mapping forest carbon potential and risks.

## install

```shell
pip install -e .
```

## usage

```python
from cmip6_downscaling.methods import ...
```

## license

All the code in this repository is [MIT](https://choosealicense.com/licenses/mit/) licensed. Some of the data provided by this API is sourced from content made available under a [CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/) license. We include attribution for this content, and we please request that you also maintain that attribution if using this data.

## about us

CarbonPlan is a non-profit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of carbon removal and climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/cmip6-downscaling/issues/new) or [sending us an email](mailto:hello@carbonplan.org).
