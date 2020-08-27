import os
from importlib import util
from unittest import TestCase

import pandas as pd

from proxy_builder import MedianCSRegression


class TestMedianCSRegression(TestCase):

    def setUp(self) -> None:
        pkg_name = "proxy_builder_tests.test_files"
        file_train_input = "cds_spreads_data_all.csv"
        spec = util.find_spec(pkg_name)
        path = spec.submodule_search_locations[0]
        path_to_train = os.path.join(path, file_train_input)
        self.train_input = pd.read_csv(path_to_train, index_col=0)
        self.key_cols = ['region', 'sector', 'rating', 'tenor']
        self.y_column = "benchmark"
        self.tree_params = {"max_features": 2, "n_estimators": 100, "max_depth": 20,
                            "learning_rate": 0.1, "min_samples_leaf": 2, "min_samples_split": 2}
        self.params_to_tune = {"max_depth": [3, 5, 10], "min_samples_split": [2],
                               "min_samples_leaf": [1], "max_features": ["sqrt"],
                               "n_estimators": [100], "learning_rate": [0.1]}
        self.reg_model_xsec_med = MedianCSRegression(train_df=self.train_input, cols_x_dummy=self.key_cols,
                                                     col_y=self.y_column)
        self.reg_model_xsec_med_2 = MedianCSRegression(train_df=self.train_input, cols_x_dummy=self.key_cols,
                                                       col_y=self.y_column)
        self.reg_model_xsec_med_2.fit(**self.tree_params, random_state=17)

    def test_fit_1(self):
        self.reg_model_xsec_med.fit(**self.tree_params)
        self.assertIsNotNone(self.reg_model_xsec_med.model)

    def test_fit_2(self):
        self.reg_model_xsec_med.fit(**self.tree_params)
        self.assertEqual(self.reg_model_xsec_med.model.model, "median")

    def test_fit_cv_1(self):
        self.reg_model_xsec_med.fit_cv(self.params_to_tune)
        obtained_depth = self.reg_model_xsec_med.model.max_depth
        self.assertIn(obtained_depth, self.params_to_tune.get("max_depth"))

    def test_rsquared(self):
        rsquared = self.reg_model_xsec_med_2.rsquared
        self.assertLess(rsquared, 1)

    def test_data(self):
        x_df = self.reg_model_xsec_med_2.x_train
        y_df = self.reg_model_xsec_med_2.y_train
        self.assertEqual(x_df.shape[0], y_df.shape[0])

    def test_predictions(self):
        predictions = self.reg_model_xsec_med_2.predictions
        self.assertLess(abs(predictions[0] - -6.0999140735), 1e-06)
