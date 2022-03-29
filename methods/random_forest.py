from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from .base_model import BaseModel
from  sklearn.ensemble import ExtraTreesRegressor
import numpy as np

class RandomForest(BaseModel):
    def __init__(self, name):  # , full_name):
        """
        Class initialization.
        Args
            :attr:`name` (string):
            the name attribute of the method.
        TODO: add more generic args?
        """
        super(RandomForest, self).__init__(name)

        self.name = name
        # super(RandomForest)
        est = ExtraTreesRegressor(n_estimators=100)
        self.model = IterativeImputer(skip_complete=True, estimator=est, verbose=False, missing_values=np.nan)

    def fit(self,X,  **kwargs):
        self.model.fit(X)

    def fit_transform(self,X, **kwargs):
        return self.model.fit_transform(X)

    def impute(self,X, **kwargs):
        return self.model.fit_transform(X)

    def transform(self,X, **kwargs):
        return self.model.transform(X)