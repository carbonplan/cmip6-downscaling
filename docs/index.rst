.. _Intro:

CMIP6-Downscaling
-----------------

This project implements a collection of statistical climate downscaling methods and provides workflows to generate downscaled climate data.

Statistical climate downscaling refers to a collection of algorithms that, when applied to raw climate model output, correct for systemic biases
and generate data at higher spatial resolutions. These methods are an important part of many analytical workflows in climate impacts and climate mitigation.

The project was designed with two primary goals in mind:

1. to provide a sandbox for developing and comparing across a wide range of downscaling methods
2. to develop production-ready pipelines for downscaling CMIP6 model output.

The package currently provides `Prefect flows <https://docs.prefect.io/core/>`_ for the following methods:

- `Bias-correction and Statistical Downscaling (BCSD) <https://link.springer.com/article/10.1023/B:CLIM.0000013685.99609.9e>`_
- `Ensemble Generalized Statistical Downscaling (En-GARD) <https://gard.readthedocs.io/en/develop/>`_
- `Multivariate Adapted Constructed Analogs (MACA) <https://doi.org/10.1002/joc.2312>`_
- `DeepSD <https://dl.acm.org/doi/10.1145/3097983.3098004>`_

Many of these workflows build on or extend functionality in `Scikit-downscale <https://scikit-downscale.readthedocs.io/en/latest/>`_,
providing complete end-to-end pipelines for generating downscaled climate data. The workflows are designed to be run on the cloud
(Azure West Europe for best performance) but smaller applications should be possible using Prefect's local deployment methods.

.. toctree::
   :hidden:

   self

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started

    Quickstart <quick-start>

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: How-To Guides

    Running Flows <running-flows>


.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Explanation

    Input Datasets <input-datasets>
    Downscaling Methods <downscaling-methods>

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Reference

    API <api>
