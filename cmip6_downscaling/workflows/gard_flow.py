import os

from dask_kubernetes import KubeCluster, make_pod_spec
from prefect import Flow, task
from prefect.executors import DaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure

from cmip6_downscaling.methods.gard import (
    gard_preprocess,
    gard_fit_and_predict,
    gard_postprocess,
)

make_flow_paths_task = task(make_flow_paths, log_stdout=True, nout=4)

preprocess_bcsd_task = task(preprocess_bcsd, log_stdout=True, nout=2)

prep_bcsd_inputs_task = task(prep_bcsd_inputs, log_stdout=True, nout=3)

fit_and_predict_task = task(fit_and_predict, log_stdout=True)

postprocess_bcsd_task = task(postprocess_bcsd, log_stdout=True)


def gard_flow(
    model,
    label_name,
    feature_list=None,
    dim='time',
    bias_correction_method='quantile_transform',
    bc_kwargs=None,
    generate_scrf=True,
):
    """
    Parameters
    ----------
    model                 : a GARD model instance to be fitted pointwise
    feature_list          : a list of feature names to be used in predicting
    dim                   : string. dimension to apply the model along. Default is ``time``.
    bias_correction_method: string of the name of bias correction model
    bc_kwargs             : kwargs dict. directly passed to the bias correction model
    generate_scrf         : boolean. indicates whether a spatio-temporal correlated random field (scrf) will be
                            generated based on the fine resolution data provided in .fit as y. if false, it is
                            assumed that a pre-generated scrf will be passed into .predict as an argument that
                            matches the prediction result dimensions.
    spatial_feature       : (3, 3)
    """
    self._dim = dim
    if not isinstance(model, (AnalogBase, PureRegression)):
        raise TypeError('model must be part of the GARD family of pointwise models ')
    self.features = feature_list
    self.label_name = label_name
    self._model = model
    self.thresh = model.thresh

    # shared between multiple method types but point wise
    # TODO: spatial features
    # TODO: extend this to include transforming the feature space into PCA, etc
    # map + 1d component (pca) of ocean surface temperature for precip prediction
