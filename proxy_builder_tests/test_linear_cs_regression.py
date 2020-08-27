import os
from importlib import util
from unittest import TestCase

import pandas as pd

from proxy_builder import LinearCSRegressionCDS


class TestLinearCSRegressionCDS(TestCase):

    def setUp(self) -> None:
        pkg_name = "proxy_builder_tests.test_files"
        file_train_input = "cds_spreads_data_train.csv"
        file_test_input = "cds_spreads_data_test.csv"
        spec = util.find_spec(pkg_name)
        path = spec.submodule_search_locations[0]
        path_to_train = os.path.join(path, file_train_input)
        path_to_test = os.path.join(path, file_test_input)
        self.train_input = pd.read_csv(path_to_train, index_col=0)
        self.test_input = pd.read_csv(path_to_test, index_col=0)
        self.key_cols = ['region', 'sector', 'rating', 'tenor']
        self.y_column = "benchmark"
        self.reg_model_xsec_lin_cds = LinearCSRegressionCDS(train_df=self.train_input, cols_x_dummy=self.key_cols,
                                                            col_y=self.y_column)
        self.reg_model_xsec_lin_cds.fit()

    def test_get_proxies_1(self):
        proxies = self.reg_model_xsec_lin_cds.bucket_values
        cols_actual = set(proxies.columns)
        cols_expected = {"region", "sector", "rating", "tenor", self.y_column + "_pred"}
        self.assertEqual(cols_actual, cols_expected)

    def test_get_proxies_2(self):
        proxies = self.reg_model_xsec_lin_cds.bucket_values
        find_mask = (proxies["region"] == "EUR") & (proxies["sector"] == "FIN") & \
                    (proxies["rating"] == "A") & (proxies["tenor"] == "Y05")
        value_actual = proxies.loc[find_mask, self.y_column + "_pred"].values[0]
        value_expected = 0.0045339951
        self.assertLess(abs(value_actual - value_expected), 1e-06)

    def test_predict_1(self):
        predictions = self.reg_model_xsec_lin_cds.predict(self.test_input)
        pred_col_set = set(predictions.columns)
        required_cols = {self.y_column, self.y_column + "_pred"}
        self.assertTrue(required_cols.issubset(pred_col_set))

    def test_predict_2(self):
        predictions = self.reg_model_xsec_lin_cds.predict(self.test_input)
        const_mask = predictions['const_fame'] == "CL_CORP_BONDS'MARKIT.CDS.EUR.AIRLIQ.SNRFOR.MM.Y10.SPR.CLUB"
        value_actual = predictions.loc[const_mask, self.y_column].values[0]
        value_expected = 0.00551095
        self.assertLess(abs(value_actual - value_expected), 1e-06)

    def test_predict_3(self):
        predictions = self.reg_model_xsec_lin_cds.predict(self.test_input)
        const_mask = predictions['const_fame'] == "CL_CORP_BONDS'MARKIT.CDS.EUR.AIRLIQ.SNRFOR.MM.Y10.SPR.CLUB"
        value_actual = predictions.loc[const_mask, self.y_column + "_pred"].values[0]
        value_expected = 0.0050587283
        self.assertLess(abs(value_actual - value_expected), 1e-06)
