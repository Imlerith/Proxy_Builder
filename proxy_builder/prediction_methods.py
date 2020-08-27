import itertools
import numpy as np
import pandas as pd
from proxy_builder.linear_cs_regression import LinearCSRegression
from proxy_builder.utils import lazy_property


class LinearNonlinearMixinCDS(LinearCSRegression):

    @lazy_property
    def bucket_values(self):
        """
        Function to calculate CCA CDS bucket values
        :return: a dataframe of CCA CDS bucket values
        """
        # --- indicator matrix to obtain predictions for buckets
        dummy_arrays = [self._train_df[col].unique() for col in self._x_columns_dummy]
        buckets = list(set(itertools.product(*dummy_arrays)))
        x_train, _ = self._get_train_input_data()
        x_train = x_train.drop(columns=self._x_columns_dummy + ['const_fame'])
        indicator_df = pd.DataFrame(0, index=pd.MultiIndex.from_tuples(buckets, names=self._x_columns_dummy),
                                    columns=x_train.columns)
        for bkt in buckets:
            for dim in bkt:
                indicator_df.loc[bkt, dim] = 1
        # --- get predictions
        y_pred = self._model.predict(indicator_df.values)
        indicator_df['bkt'] = indicator_df.index.tolist()
        indicator_df = indicator_df.reset_index(drop=True)
        bucket_values = indicator_df['bkt'].apply(pd.Series)
        bucket_values.rename(columns=dict(zip(bucket_values.columns, self._x_columns_dummy)), inplace=True)
        bucket_values[self._y_column + '_pred'] = y_pred
        bucket_values.sort_values(by=self._x_columns_dummy, inplace=True)
        # --- transform back to exp/log if needed
        if self._type_y == 'log':
            bucket_values[self._y_column + '_pred'] = bucket_values[self._y_column + '_pred'].apply(np.exp)
        elif self._type_y == 'exp':
            bucket_values[self._y_column + '_pred'] = bucket_values[self._y_column + '_pred'].apply(np.log)
        else:
            pass
        bucket_values.reset_index(drop=True, inplace=True)
        return bucket_values

    def predict(self, test_df):
        x_test, y_test = self._get_test_input_data(test_df)
        x_test_values = x_test.drop(columns=self._x_columns_dummy + ['const_fame'])
        y_pred = self._model.predict(x_test_values).flatten()
        # --- transform back to exp/log if needed
        test_results = x_test.loc[:, self._x_columns_dummy + ['const_fame']]
        if self._type_y == 'log':
            test_results[self._y_column] = np.exp(y_test.values)
            test_results[self._y_column + '_pred'] = np.exp(y_pred)
        elif self._type_y == 'exp':
            test_results[self._y_column] = np.log(y_test.values)
            test_results[self._y_column + '_pred'] = np.log(y_pred)
        else:
            test_results[self._y_column] = y_test.values
            test_results[self._y_column + '_pred'] = y_pred
        # --- save the results
        test_results.sort_values(by=self._x_columns_dummy, inplace=True)
        test_results = test_results.loc[:, ['const_fame', self._y_column,
                                            self._y_column + '_pred']]\
            .reset_index(drop=True)
        return test_results
