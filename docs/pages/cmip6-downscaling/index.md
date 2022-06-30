import Section from '../../components/section'

# CMIP6-Downscaling

This project implements a collection of statistical climate downscaling methods and provides workflows to generate downscaled climate data.

Statistical climate downscaling refers to a collection of algorithms that, when applied to raw climate model output, correct for systemic biases and generate data at higher spatial resolutions. These methods are an important part of many analytical workflows in climate impacts and climate mitigation.

The project was designed with two primary goals in mind:

1. to provide a sandbox for developing and comparing across a wide range of downscaling methods
2. to develop production-ready pipelines for downscaling CMIP6 model output.

The package currently provides [Prefect flows](https://docs.prefect.io/core/) for the following methods:

- [Bias-correction and Statistical Downscaling (BCSD)](https://link.springer.com/article/10.1023/B:CLIM.0000013685.99609.9e)
- [Ensemble Generalized Statistical Downscaling (En-GARD)](https://gard.readthedocs.io/en/develop/)
- [Multivariate Adapted Constructed Analogs (MACA)](https://doi.org/10.1002/joc.2312)
- [DeepSD](https://dl.acm.org/doi/10.1145/3097983.3098004)

Many of these workflows build on or extend functionality in [Scikit-downscale](https://scikit-downscale.readthedocs.io/en/latest/), providing complete end-to-end pipelines for generating downscaled climate data. The workflows are designed to be run on the cloud (Azure West Europe for best performance) but smaller applications should be possible using Prefect's local deployment methods.

export default ({ children }) => <Section name='intro'>{children}</Section>
