from proxy_builder.ridge_cs_regression import RidgeCSRegression
from proxy_builder.linear_cs_regression_cds import LinearCSRegressionCDS


class RidgeCSRegressionCDS(RidgeCSRegression, LinearCSRegressionCDS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
