import logging
from functools import wraps

from sklearn.model_selection import GridSearchCV, ShuffleSplit


def get_cross_validated_model(model, x, y, params_to_tune, n_splits=5, score='r2'):
    sscv = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)
    gscv = GridSearchCV(model, param_grid=params_to_tune, cv=sscv, scoring=score)
    gscv.fit(x, y)
    model_opt = gscv.best_estimator_
    return model_opt


def lazy_property(func):
    name = '_lazy_' + func.__name__

    @property
    def lazy(self):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            value = func(self)
            setattr(self, name, value)
            return value
    return lazy


def log_exception(level, default_result=None):
    # logger with a default value
    def log_internal(func):
        lgr = logging.getLogger("my_application")
        lgr.setLevel(level)
        fh = logging.FileHandler("my_logger.log")
        fh.setLevel(level)
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        if lgr.hasHandlers():
            lgr.handlers.clear()
        lgr.addHandler(fh)

        @wraps(func)
        def wrapper(*a, **kw):
            lgr.log(level, "Ran with args: {} and kwargs: {}".format(a, kw))
            try:
                return func(*a, **kw)
            except Exception as e:
                err = "There was an exception in  "
                err += func.__name__
                lgr.exception(err)
                lgr.exception(e)
                return default_result

        return wrapper

    return log_internal
