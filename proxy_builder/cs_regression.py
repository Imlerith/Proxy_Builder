from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Union

ListStr0 = Union[str, List[str]]


class CrossSectionalRegression(ABC):
    def __init__(self,
                 train_df: pd.DataFrame,
                 cols_x_dummy: List[str],
                 col_y: str,
                 drop_method: ListStr0 = 'smallest',
                 type_y: str = 'log'):

        assert type(train_df) is pd.DataFrame
        assert type(cols_x_dummy) is list
        assert type(col_y) is str
        assert type(drop_method) is list and len(drop_method) == len(cols_x_dummy) or \
               type(drop_method) is str and drop_method in ['first', 'last', 'smallest']
        assert type(type_y) is str and type_y in ['lin', 'log', 'exp']

        self._train_df = train_df
        self._x_columns_dummy = cols_x_dummy
        self._y_column = col_y
        self._drop_method = drop_method
        self._type_y = type_y
        self._check_input()

    def _check_input(self):
        for column in self._x_columns_dummy + [self._y_column]:
            if column not in self._train_df.columns:
                raise NameError(f"column {column} is not in the input df")
        if type(self._drop_method) is list:
            for i, val in enumerate(self._drop_method):
                if val not in set(self._train_df[self._x_columns_dummy[i]]):
                    raise NameError(f"baseline value {val} absent in the column {self._x_columns_dummy[i]}")
                assert val in set(self._train_df[self._x_columns_dummy[i]])
        else:
            if self._drop_method == 'first':
                self._names_to_drop = [list(set(self._train_df[column]))[0] for column in self._x_columns_dummy]
            elif self._drop_method == 'last':
                self._names_to_drop = [list(set(self._train_df[column]))[-1] for column in self._x_columns_dummy]
            else:
                self._names_to_drop = list()
                for column in self._x_columns_dummy:
                    counts = pd.value_counts(self._train_df[column])
                    counts_sorted_by_index = counts.iloc[np.lexsort((counts.index, -counts.values))]
                    self._names_to_drop.append(counts_sorted_by_index.index[-1])

    def _get_names_to_drop(self):
        return self._names_to_drop

    @abstractmethod
    def _fill_one_hot_encoded(self):
        pass

    @abstractmethod
    def _fill_one_hot_encoded_test(self, test_df):
        pass

    @abstractmethod
    def _fill_names(self):
        pass

    @abstractmethod
    def _fill_names_test(self, test_df):
        pass

    @abstractmethod
    def _get_train_input_data(self):
        pass

    @abstractmethod
    def _get_test_input_data(self, test_df):
        pass

    @abstractmethod
    def fit(self):
        pass

    @property
    def train_df(self):
        return self._train_df
