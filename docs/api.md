# API

```{eval-rst}
.. currentmodule:: cmip6_downscaling
```

## Data

```{eval-rst}
.. autosummary::
   :toctree: generated/

   data.cmip.postprocess
   data.cmip.load_cmip
   data.cmip.get_gcm

   data.observations.open_era5

   data.utils.to_standard_calendar
   data.utils.lon_to_180
```

## Downscaling Methods

```{eval-rst}
.. currentmodule:: cmip6_downscaling.methods
```

### BCSD

```{eval-rst}
.. autosummary::
   :toctree: generated/

   bcsd.tasks.spatial_anomalies
   bcsd.tasks.fit_and_predict
   bcsd.tasks.postprocess_bcsd
   bcsd.utils.reconstruct_finescale
```

### GARD

```{eval-rst}
.. autosummary::
   :toctree: generated/

   gard.tasks.coarsen_and_interpolate
   gard.tasks.fit_and_predict
   gard.tasks.read_scrf
   gard.utils.get_gard_model
   gard.utils.add_random_effects

```

### Common Tasks

```{eval-rst}
.. autosummary::
   :toctree: generated/

   common.tasks.make_run_parameters
   common.tasks.get_obs
   common.tasks.get_experiment
   common.tasks.rechunk
   common.tasks.time_summary
   common.tasks.get_weights
   common.tasks.get_pyramid_weights
   common.tasks.regrid
   common.tasks.pyramid
   common.tasks.run_analyses
   common.tasks.finalize
```
