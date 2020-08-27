import numpy as np
import pandas as pd
from proxy_builder.cs_regression import CrossSectionalRegression


class LinearNonlinearMixin1(CrossSectionalRegression):

    def _fill_one_hot_encoded(self):
        dummies_final = pd.get_dummies(self._train_df, columns=self._x_columns_dummy,
                                       prefix='', prefix_sep='')
        return dummies_final

    def _fill_one_hot_encoded_test(self, test_df):
        dummies_final = pd.get_dummies(test_df, columns=self._x_columns_dummy,
                                       prefix='', prefix_sep='')
        return dummies_final

    def _fill_names(self):
        names = [col for col in self._fill_one_hot_encoded().columns
                 if col not in set(self._train_df.columns.tolist()).difference(['const_fame'])]
        return names

    def _fill_names_test(self, test_df):
        names = [col for col in self._fill_one_hot_encoded_test(test_df).columns
                 if col not in set(test_df.columns.tolist()).difference(['const_fame'])]
        return names

    def _get_train_input_data(self):
        # --- one-hot-encode input data
        x = self._fill_one_hot_encoded().loc[:, self._fill_names()]\
            .reset_index(drop=True)
        y = self._fill_one_hot_encoded()[self._y_column]\
            .reset_index(drop=True)
        bkts_df = self._train_df.loc[:, self._x_columns_dummy]\
            .reset_index(drop=True)
        x = pd.concat([x, bkts_df], axis=1)
        # --- apply log/exp/lin transformation
        if self._type_y == 'log':
            y = y.apply(np.log)
        elif self._type_y == 'exp':
            y = y.apply(np.exp)
        else:
            pass
        # --- remove lines with NaN or Inf in y
        if sum(np.isnan(y)) > 0 or sum(np.isinf(y)) > 0:
            no_nan_inf_mask = ~(np.isnan(y) | np.isinf(y))
            y = y[no_nan_inf_mask]
            x = x[no_nan_inf_mask]
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        return x, y

    def _get_test_input_data(self, test_df):
        # --- one-hot-encode input data
        x = self._fill_one_hot_encoded_test(test_df)\
                .loc[:, self._fill_names_test(test_df)]\
            .reset_index(drop=True)
        y = self._fill_one_hot_encoded_test(test_df)[self._y_column]\
            .reset_index(drop=True)
        bkts_df = test_df.loc[:, self._x_columns_dummy]\
            .reset_index(drop=True)
        x = pd.concat([x, bkts_df], axis=1)
        # --- apply transformation
        if self._type_y == 'log':
            y = y.apply(np.log)
        elif self._type_y == 'exp':
            y = y.apply(np.exp)
        else:
            pass
        # --- remove lines with NaN or Inf in y
        if ((sum(np.isnan(y)) > 0) or
                (sum(np.isinf(y)) > 0)):
            no_nan_inf_mask = ~(np.isnan(y) | np.isinf(y))
            y = y[no_nan_inf_mask]
            x = x[no_nan_inf_mask]
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        return x, y

    def fit(self, *args, **kwargs):
        pass


class LinearNonlinearMixin2(CrossSectionalRegression):

    def _fill_one_hot_encoded(self):
        dummies_final = pd.get_dummies(self._train_df, columns=self._x_columns_dummy,
                                       prefix='', prefix_sep='')
        dummies_final['general'] = 1
        return dummies_final

    def _fill_one_hot_encoded_test(self, test_df):
        dummies_final = pd.get_dummies(test_df, columns=self._x_columns_dummy,
                                       prefix='', prefix_sep='')
        dummies_final['general'] = 1
        return dummies_final

    def _fill_names(self):
        names = [col for col in self._fill_one_hot_encoded().columns
                 if col not in self._names_to_drop + self._train_df.columns.tolist()]
        return names

    def _fill_names_test(self, test_df):
        names = [col for col in self._fill_one_hot_encoded_test(test_df).columns
                 if col not in self._names_to_drop + test_df.columns.tolist()]
        return names

    def _get_train_input_data(self):
        # --- one-hot-encode input data
        x = self._fill_one_hot_encoded().loc[:, self._fill_names() + ["const_fame"]]
        y = self._fill_one_hot_encoded()[self._y_column]
        # --- apply log/exp/lin transformation
        if self._type_y == 'log':
            y = y.apply(np.log)
        elif self._type_y == 'exp':
            y = y.apply(np.exp)
        else:
            pass
        # --- remove lines with NaN or Inf in y
        if sum(np.isnan(y)) > 0 or sum(np.isinf(y)) > 0:
            no_nan_inf_mask = ~(np.isnan(y) | np.isinf(y))
            y = y[no_nan_inf_mask]
            x = x[no_nan_inf_mask]
        x.drop(columns=["const_fame"], inplace=True)
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        return x, y

    def _get_test_input_data(self, test_df):
        # --- one-hot-encode input data
        x = self._fill_one_hot_encoded_test(test_df)\
                .loc[:, self._fill_names_test(test_df)]\
            .reset_index(drop=True)
        y = self._fill_one_hot_encoded_test(test_df)[self._y_column]\
            .reset_index(drop=True)
        # --- apply transformation
        if self._type_y == 'log':
            y = y.apply(np.log)
        elif self._type_y == 'exp':
            y = y.apply(np.exp)
        else:
            pass
        # --- remove lines with NaN or Inf in y
        if (sum(np.isnan(y)) > 0) or (sum(np.isinf(y)) > 0):
            no_nan_inf_mask = ~(np.isnan(y) | np.isinf(y))
            y = y[no_nan_inf_mask]
            x = x[no_nan_inf_mask]
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        return x, y

    def fit(self, *args, **kwargs):
        pass
