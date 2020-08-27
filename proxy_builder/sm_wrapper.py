import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class SMWrapper(BaseEstimator, RegressorMixin):
    """ A sklearn-style wrapper for statsmodels' OLS and Ridge regression.
        Also implements sklearn's native models such as
        KernelRidge, SVR, Gradient Boosting.
        Implemented to unify the APIs of statsmodels and sklearn.
    """
    def __init__(self, model='ols', fit_intercept=False, l1_wt=0.0, alpha=1.0, max_features=2,
                 n_estimators=100, max_depth=80, learning_rate=0.1, min_samples_leaf=2,
                 min_samples_split=2, random_state=None):
        assert model in ['ols', 'median', 'ridge']
        self.model = model
        if self.model == 'ols':
            self.model_ = OLS
        elif self.model == 'median':
            self.model_ = GradientBoostingRegressor
        else:
            self.model_ = OLS
            self.is_regularized = True
        self.fit_intercept = fit_intercept
        self.l1_wt = l1_wt
        self.alpha = alpha
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.results = None

    def fit(self, x, y):
        # --- add intercept to the model
        if self.fit_intercept:
            x = sm.add_constant(x)
        # --- fit method for the OLS model
        if self.model == 'ols':
            self.results = self.model_(y, x).fit()
        # --- fit method for gradient-boosted tree quantile regression
        elif self.model == 'median':
            self.results = self.model_(loss='quantile', alpha=0.5, max_depth=self.max_depth,
                                       min_samples_split=self.min_samples_split,
                                       min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                                       n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                       random_state=self.random_state).fit(x, y)
        # --- fit method for ridge regression
        else:
            self.results = self.model_(y, x).fit_regularized(L1_wt=self.l1_wt, alpha=self.alpha)

    def predict(self, x):
        if self.fit_intercept:
            x = sm.add_constant(x)
        return self.results.predict(x)
