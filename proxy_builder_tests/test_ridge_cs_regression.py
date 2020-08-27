import os
from importlib import util
from unittest import TestCase

import pandas as pd

from proxy_builder import RidgeCSRegression


class TestRidgeCSRegression(TestCase):

    def setUp(self) -> None:
        pkg_name = "proxy_builder_tests.test_files"
        file_train_input = "cds_spreads_data_all.csv"
        spec = util.find_spec(pkg_name)
        path = spec.submodule_search_locations[0]
        path_to_train = os.path.join(path, file_train_input)
        self.train_input = pd.read_csv(path_to_train, index_col=0)
        self.key_cols = ['region', 'sector', 'rating', 'tenor']
        self.y_column = "benchmark"
        self.reg_params = {"alpha": 0.1}
        self.params_to_tune = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}
        self.reg_model_xsec_ridge = RidgeCSRegression(train_df=self.train_input, cols_x_dummy=self.key_cols,
                                                      col_y=self.y_column)
        self.reg_model_xsec_ridge_2 = RidgeCSRegression(train_df=self.train_input, cols_x_dummy=self.key_cols,
                                                        col_y=self.y_column)
        self.reg_model_xsec_ridge_2.fit(**self.reg_params)

    def test_fit_1(self):
        self.reg_model_xsec_ridge.fit(**self.reg_params)
        self.assertIsNotNone(self.reg_model_xsec_ridge.model)

    def test_fit_2(self):
        self.reg_model_xsec_ridge.fit(**self.reg_params)
        self.assertEqual(self.reg_model_xsec_ridge.model.model, "ridge")

    def test_fit_cv_1(self):
        self.reg_model_xsec_ridge.fit_cv(self.params_to_tune)
        obtained_alpha = self.reg_model_xsec_ridge.model.alpha
        self.assertIn(obtained_alpha, self.params_to_tune.get("alpha"))

    def test_rsquared(self):
        rsquared = self.reg_model_xsec_ridge_2.rsquared
        self.assertLess(rsquared, 1)

    def test_data(self):
        x_df = self.reg_model_xsec_ridge_2.x_train
        y_df = self.reg_model_xsec_ridge_2.y_train
        self.assertEqual(x_df.shape[0], y_df.shape[0])

    def test_predictions(self):
        predictions = self.reg_model_xsec_ridge_2.predictions
        self.assertLess(abs(predictions[0] - -5.6549942406), 1e-06)
