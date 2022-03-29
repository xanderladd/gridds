
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from .base_model import BaseModel
from sklearn.linear_model import BayesianRidge as BRIDGE


class BayesianRidge(BaseModel):
    def __init__(self, name):  # , full_name):
        """
        Class initialization.
        Args
            :attr:`name` (string):
            the name attribute of the method.
        TODO: add more generic args?
        """
        self.name = name
        super(BayesianRidge, self).__init__(name)

        # super(BayesianRidge)
        state=np.random.randint(0,1000)
        est = BRIDGE()
        self.model = IterativeImputer(estimator=est,\
            random_state=state,sample_posterior=False,verbose=False)
        #self.model = IterativeImputer(random_state=0, verbose=False)

    def fit(self,X, **kwargs):
        self.model.fit(X)

    def predict(self,X, **kwargs):
        return self.model.predict(X)

    def fit_transform(self,X, **kwargs):
        return self.model.fit_transform(X)

    def impute(self,X, **kwargs):
        return self.model.fit_transform(X.reshape(-1,1))

    def transform(self,X, **kwargs):
        return self.model.transform(X)


