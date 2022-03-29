from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from .base_model import BaseModel
import numpy as np


class KNN(BaseModel):
    def __init__(self, name, n_neighbors=10):  # , full_name):
        """
        Class initialization.
        Args
            :attr:`name` (string):
            the name attribute of the method.
        TODO: add more generic args?
        """
        self.name = name
        super(KNN, self).__init__(name)
        '''  
        3 options below for self.model
        '''
        #self.model = KNeighborsRegressor()
        self.model = KNeighborsRegressor()
        # self.model = KNNImputer(n_neighbors=n_neighbors)

    def fit(self,X,  **kwargs):
        self.model.fit(X)
    
    def predict(self,X,  **kwargs):
        return self.model.predict(X)

    
    def fit_transform(self,X,  **kwargs):
        self._model = IterativeImputer(random_state=np.random.randint(0,10000),skip_complete=True,estimator=self.model,verbose=False)
        return self._model.fit_transform(X)
    
    # def fit_transform(self,X):
    #     return self.model.fit_transform(X)

    def impute(self,X,  **kwargs):
        self.model = KNNImputer(n_neighbors=n_neighbors)
        return self.model.fit_transform(X)

    def transform(self,X,  **kwargs):
        return self.model.transform(X)
