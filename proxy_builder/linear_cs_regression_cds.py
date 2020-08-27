import numpy as np
import pandas as pd
import itertools
from proxy_builder.linear_cs_regression import LinearCSRegression
from proxy_builder.prediction_methods import LinearNonlinearMixinCDS
from proxy_builder.utils import lazy_property


class LinearCSRegressionCDS(LinearNonlinearMixinCDS, LinearCSRegression):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @lazy_property
    def bucket_values(self):
        """
        Function to calculate CCA CDS bucket values
        :return: a dataframe of CCA CDS bucket values
        """
        # --- indicator matrix to obtain predictions
        dummy_arrays = [self._train_df[col].unique() for col in self._x_columns_dummy]
        buckets = list(set(itertools.product(*dummy_arrays)))
        indicator_df = pd.DataFrame(0, index=pd.MultiIndex.from_tuples(buckets, names=self._x_columns_dummy),
                                    columns=self.betas.index.values)
        indicator_df['general'] = 1
        for bkt in buckets:
            for dim in bkt:
                indicator_df.loc[bkt, dim] = 1
        # --- raw predictions
        predictions = np.matmul(indicator_df.values, self.betas.beta.values.reshape(-1, 1))
        # --- final results
        indicator_df['bkt'] = indicator_df.index.tolist()
        indicator_df = indicator_df.reset_index(drop=True)
        bucket_values = pd.DataFrame(0, index=range(indicator_df.shape[0]),
                                     columns=self._x_columns_dummy + [self._y_column])
        bucket_values[self._x_columns_dummy] = indicator_df['bkt'].apply(pd.Series)
        bucket_values[self._y_column] = predictions
        bucket_values = bucket_values.sort_values(by=self._x_columns_dummy).reset_index(level=0, drop=True)
        # --- transform back to exp/log if needed
        if self._type_y == 'log':
            bucket_values[self._y_column] = bucket_values[self._y_column].apply(np.exp)
        elif self._type_y == 'exp':
            bucket_values[self._y_column] = bucket_values[self._y_column].apply(np.log)
        else:
            pass
        bucket_values.rename(columns={self._y_column: self._y_column + '_pred'}, inplace=True)
        return bucket_values

    def predict(self, test_df):
        # --- indicator matrix to obtain predictions
        dummy_arrays = [test_df[col].unique() for col in self._x_columns_dummy]
        buckets = list(set(itertools.product(*dummy_arrays)))
        indicator_df = pd.DataFrame(0, pd.MultiIndex.from_tuples(buckets, names=self._x_columns_dummy),
                                    columns=self.betas.index.values)
        indicator_df['general'] = 1
        for bkt in buckets:
            for dim in bkt:
                indicator_df.loc[bkt, dim] = 1
        # --- raw predictions
        predictions = np.matmul(indicator_df.values, self.betas.beta.values.reshape(-1, 1))
        # --- final results
        indicator_df['bkt'] = indicator_df.index.tolist()
        indicator_df = indicator_df.reset_index(drop=True)
        test_results = pd.DataFrame(0, index=range(indicator_df.shape[0]),
                                    columns=self._x_columns_dummy + [self._y_column])
        test_results[self._x_columns_dummy] = indicator_df.bkt.apply(pd.Series)
        test_results[self._y_column] = predictions
        test_results = test_results.sort_values(by=self._x_columns_dummy).reset_index(level=0, drop=True)
        # --- transform back to exp/log if needed
        if self._type_y == 'log':
            test_results[self._y_column] = test_results[self._y_column].apply(np.exp)
        elif self._type_y == 'exp':
            test_results[self._y_column] = test_results[self._y_column].apply(np.log)
        else:
            pass
        test_results.rename(columns={self._y_column: self._y_column + '_pred'}, inplace=True)
        test_results = pd.merge(test_results, test_df, how='inner', on=self._x_columns_dummy) \
                           .dropna() \
                           .drop_duplicates() \
                           .loc[:, ['const_fame', self._y_column, self._y_column + '_pred']]
        return test_results
