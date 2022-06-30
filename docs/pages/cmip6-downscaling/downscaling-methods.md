import Section from '../../components/section'

## Downscaling methods

We implemented four downscaling methods globally. Descriptions of these implementations are below, along with references to further information. Our [explainer article](https://carbonplan.org/research/cmip6-downscaling-explainer) discusses the importance of downscaling, and describes some of the key methodological differences, in more detail.

### MACA

The Multivariate Adaptive Constructed Analogs method [(Abatzoglou and Brown, 2012)](https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/joc.2312) finds common spatial patterns among GCM and reference datasets to construct downscaled future projections from actual weather patterns from the past. The method involves a combination of coarse and fine-scale bias-correction, detrending of GCM data, and analog selection, steps which are detailed thoroughly in the [MACA Datasets documentation](https://climate.northwestknowledge.net/MACA/MACAmethod.php). MACA is designed to operate at the regional scale. As a result, we split the global domain into smaller regions using the AR6 delineations from the `regionmask` [package](https://regionmask.readthedocs.io/en/stable/) and downscaled each region independently. We then stitched the regions back together to create a seamless global product. Of the methods we have implemented, MACA is the most established.

### GARD-SV

The Generalized Analog Regression Downscaling (GARD) (Guttmann et al., in review) approach is a downscaling sandbox that allows scientists to create custom downscaling implementations, supporting single or multiple predictor variables, pure regression and pure analog approaches, and different bias-correction routines. At its core, GARD builds a linear model for every pixel relating the reference dataset at the fine-scale to the same data coarsened to the scale of the GCM. The downscaled projections are then further perturbed by spatially-correlated random fields to reflect the error in the regression models. Our GARD-SV (single-variate) implementation uses the same variable for training and prediction (e.g. precipitation is the only predictor for downscaling precipitation). For regression, we used the PureRegression method, building a single model for each pixel from the entire timeseries of training data. The precipitation model included a logistic regression component, with a threshold of 0 mm/day for constituting a precipitation event.

### GARD-MV

The GARD-MV (multi-variate) implementation follows the same process as the GARD-SV method but uses multiple predictor variables for model training and inference. Specifically, we used three predictors for each downscaling model, adding the two directions of 500mb winds to each model. Thus, the predictors for precipitation in this model are precipitation, longitudinal wind, and latitudinal wind.

### DeepSD

DeepSD uses a computer vision approach to learn spatial patterns at multiple resolutions [Vandal et al., 2017](https://dl.acm.org/doi/10.1145/3097983.3098004). Specifically, DeepSD is a stacked super-resolution convolutional neural network. We adapted the [open-source DeepSD implementation](https://github.com/tjvandal/deepsd) for downscaling global ensembles by updating the source code for Python 3 and TensorFlow2, removing the batch normalization layer, normalizing based on historical observations, training models for temperature and precipitation, and training on a global reanalysis product (ERA5). In addition, we trained the model for fewer iterations than in Vandal et al., 2017 and clipped aphysical precipitation values at 0. Our dataset includes an additional bias-corrected product (DeepSD-BC). Given its origin in deep learning, this method is the most different from those included here, and is an experimental contribution to our dataset.
