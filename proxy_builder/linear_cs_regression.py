import pandas as pd
from proxy_builder.cs_regression import CrossSectionalRegression
from proxy_builder.data_processing_methods import LinearNonlinearMixin2
from proxy_builder.sm_wrapper import SMWrapper
from proxy_builder.utils import lazy_property


class LinearCSRegression(LinearNonlinearMixin2, CrossSectionalRegression):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = None

    def fit(self):
        # --- fit the model
        self._model = SMWrapper()
        self._model.fit(self.x_train, self.y_train)

    @lazy_property
    def train_data(self):
        x_train, y_train = self._get_train_input_data()
        return x_train.values, y_train.values

    @property
    def x_train(self):
        return self.train_data[0]

    @property
    def y_train(self):
        return self.train_data[1]

    @property
    def model(self):
        return self._model

    @property
    def betas(self):
        return pd.DataFrame(
            {'beta': self.model.results.params.tolist() + [0] * len(self._x_columns_dummy),
             'name': self._fill_names() + self._names_to_drop}).sort_values('name').set_index('name')

    @property
    def pvalues(self):
        return pd.DataFrame(
            {'pvalue': self.model.results.pvalues,
             'name': self._fill_names()}).set_index('name')

    @property
    def residuals(self):
        return self.model.results.resid

    @property
    def predictions(self):
        return self.model.predict(self.x_train)

