from proxy_builder.median_cs_regression import MedianCSRegression
from proxy_builder.linear_cs_regression import LinearCSRegression
from proxy_builder.data_processing_methods import LinearNonlinearMixin1
from proxy_builder.prediction_methods import LinearNonlinearMixinCDS


class MedianCSRegressionCDS(LinearNonlinearMixinCDS, MedianCSRegression, LinearNonlinearMixin1, LinearCSRegression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
