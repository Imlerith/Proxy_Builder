from collections.abc import Mapping
from proxy_builder.linear_cs_regression import LinearCSRegression
from proxy_builder.sm_wrapper import SMWrapper
from proxy_builder.utils import get_cross_validated_model


class RidgeCSRegression(LinearCSRegression):

    admissible_params_to_tune = {"alpha"}

    def __init__(self, *args, **kwargs):
        super(RidgeCSRegression, self).__init__(*args, **kwargs)
        self._model = None

    def fit(self, *args, **kwargs):
        # --- fit the model
        self._model = SMWrapper(model="ridge", *args, **kwargs)
        self._model.fit(self.x_train, self.y_train)

    def fit_cv(self, params_to_tune):
        assert isinstance(params_to_tune, Mapping), "The set of parameters to tune must be a dict"
        assert set(params_to_tune.keys()).issubset(self.admissible_params_to_tune), \
            f"Parameters allowed for tuning are: \n" \
            f"{', '.join(self.admissible_params_to_tune)}"
        # --- fit the model
        self._model = get_cross_validated_model(SMWrapper(model="ridge"), self.x_train,
                                                self.y_train, params_to_tune)
        self._model.fit(self.x_train, self.y_train)

    @property
    def rsquared(self):
        return self.model.score(self.x_train, self.y_train)
