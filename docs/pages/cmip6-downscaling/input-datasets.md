import Section from '../../components/section'

# Input Datasets

..# Methods

## CMIP6 raw datasets

The downscaled datasets shown here are derived from results from the [Coupled Model Intercomparison Project Phase 6](https://doi.org/10.5194/gmd-9-1937-2016). Raw datasets are also available in the web catalog (labeled “Raw”). GCMs are run at different spatial resolutions, and the data presented here are displayed in their original spatial resolution. The raw CMIP6 datasets were accessed via the Pangeo data catalog.

## Reference dataset

All downscaled datasets here were trained on the [ERA5](https://doi.org/10.1002/qj.3803) global reanalysis product at the 0.25 degree (~25 km) spatial resolution. All downscaling methods used daily temperature maxima and minima and precipitation for the period 1981-2010. One algorithm (GARD-MV) also used wind.

## Reference dataset

### ERA5 Reanalysis

All downscaled datasets here were trained on the [ERA5](https://doi.org/10.1002/qj.3803) global reanalysis product at the 0.25 degree (~25 km) spatial resolution. All downscaling methods used daily temperature maxima and minima and precipitation for the period 1981-2010. One algorithm (GARD-MV) also used wind.

This dataset can be accessed/explored via an [intake](https://intake-esm.readthedocs.io/en/stable/) catalog.

```python
# !pip install intake-esm

import intake
cat = intake.open_esm_datastore("https://cmip6downscaling.blob.core.windows.net/training/ERA5-azure.json")

```

The ERA5 data [transfering](https://github.com/carbonplan/cmip6-downscaling/blob/4bf65c61f7192908cca81fe94cda3b94931586f0/flows/ERA5/ERA5_transfer.py) and [processing](https://github.com/carbonplan/cmip6-downscaling/blob/4bf65c61f7192908cca81fe94cda3b94931586f0/flows/ERA5/ERA5_resample.py) scripts can be found on [Github](https://github.com/carbonplan/cmip6-downscaling).

The ECMWF ERA5 dataset was produced by the European Centre for Medium-Range Weather Forecasts (ECMWF) and is licencesed under Creative Commons Attribution 4.0 International (CC BY 4.0).
https://apps.ecmwf.int/datasets/licences/general/

## CMIP6 raw datasets

The downscaled datasets shown here are derived from results from the [Coupled Model Intercomparison Project Phase 6](https://doi.org/10.5194/gmd-9-1937-2016). Raw datasets are also available in the web catalog (labeled “Raw”). GCMs are run at different spatial resolutions, and the data presented here are displayed in their original spatial resolution. The raw CMIP6 datasets were accessed via the Pangeo data catalog.

The transfer script can be found [here](https://github.com/carbonplan/cmip6-downscaling/blob/main/flows/cmip6_transfer.py)

export default ({ children }) => <Section name='Input Datasets'>{children}</Section>
