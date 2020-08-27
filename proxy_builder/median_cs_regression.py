from collections.abc import Mapping
from proxy_builder.cs_regression import CrossSectionalRegression
from proxy_builder.data_processing_methods import LinearNonlinearMixin1
from proxy_builder.sm_wrapper import SMWrapper
from proxy_builder.utils import get_cross_validated_model, lazy_property


class MedianCSRegression(LinearNonlinearMixin1, CrossSectionalRegression):

    admissible_params_to_tune = {"max_depth", "min_samples_split", "min_samples_leaf",
                                 "max_features", "n_estimators", "learning_rate"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = None

    def fit(self, *args, **kwargs):
        # --- fit the model
        self._model = SMWrapper(model="median", *args, **kwargs)
        self._model.fit(self.x_train, self.y_train)

    def fit_cv(self, params_to_tune):
        assert isinstance(params_to_tune, Mapping), "The set of parameters to tune must be a dict"
        assert set(params_to_tune.keys()).issubset(self.admissible_params_to_tune), \
            f"Parameters allowed for tuning are: \n" \
            f"{', '.join(self.admissible_params_to_tune)}"
        # --- fit the model
        self._model = get_cross_validated_model(SMWrapper(model="median"), self.x_train,
                                                self.y_train, params_to_tune)
        self._model.fit(self.x_train, self.y_train)

    @lazy_property
    def train_data(self):
        x_train, y_train = self._get_train_input_data()
        x_train.drop(columns=self._x_columns_dummy + ['const_fame'], inplace=True)
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
    def rsquared(self):
        return self.model.score(self.x_train, self.y_train)

    @property
    def predictions(self):
        return self.model.predict(self.x_train)

    @property
    def predictions_cv(self):
        return self.model.predict(self.x_train)
