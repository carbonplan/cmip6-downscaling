import Section from '../../components/section'

# Input Datasets

..

## Observations

The multiple downscaling methods used in this project require observation datasets to train against. In the current iteration of this project, the primary observation dataset used was the ERA5 reanalysis dataset.

### ERA5 Reanalysis

The ERA5 reanalasis dataset is a 40+ year global hourly climate product produced by the European Centre for Medium-Range Weather Forecasts (ECMWF). It contains multiple variables produced on 30km grid at 137 atmopsheric levels.

A version of the ERA5 dataset is stored in Zarr format on the was [AWS open data regsitry](https://registry.opendata.aws/ecmwf-era5/). For the downscaling project a subset of this dataset was transfered to stores on Microsoft Azure West. In the transfer, hourly data was resampled to daily temporal resoltuion.

This dataset can be accessed/explored via an intake catalog.

```python
# !pip install intake-esm

import intake
cat = intake.open_esm_datastore("https://cmip6downscaling.blob.core.windows.net/training/ERA5-azure.json")

```

or via our webmap (LINK TO FUTURE WEBMAP)

The ERA5 data [transfering](https://github.com/carbonplan/cmip6-downscaling/blob/4bf65c61f7192908cca81fe94cda3b94931586f0/flows/ERA5/ERA5_transfer.py) and [processing](https://github.com/carbonplan/cmip6-downscaling/blob/4bf65c61f7192908cca81fe94cda3b94931586f0/flows/ERA5/ERA5_resample.py) scripts can be found on [Github](https://github.com/carbonplan/cmip6-downscaling).

The ECMWF ERA5 dataset is licencesed under Creative Commons Attribution 4.0 International (CC BY 4.0).
https://apps.ecmwf.int/datasets/licences/general/

## GCMs

...Data transfered from multiple CMIP6 ensemble members.

Zarr stores of the cmip6 archive are available in a pangeo intake catalog.

A subset of this dataset was transfered to Azure Europe West.

The transfer script can be found [here](https://github.com/carbonplan/cmip6-downscaling/blob/main/flows/cmip6_transfer.py)

export default ({ children }) => <Section name='Input Datasets'>{children}</Section>
