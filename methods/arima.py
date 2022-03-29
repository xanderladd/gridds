import os
from statsmodels.tsa.arima.model import ARIMA as sm_ARIMA
import time
import numpy as np
from .base_model import BaseModel

class ARIMA(BaseModel):
    def __init__(self, name, p=5,d=1,q=0 ):
        super(ARIMA, self).__init__(name)
        self.name = name
        self.p = p
        self.d = d
        self.q = q

    def fit_transform(self,X,**kwargs):
        print('univariate for now')
        # model=sm_ARIMA(site_data[:,0],order=(1,1,1))
        preds = []
        for X_dim in range(X.shape[1]):
            self.fit(X[:,X_dim])
            predictions = np.expand_dims(self.predict(X[:,X_dim]), axis=-1)
            preds.append(predictions)
        predictions = np.hstack(preds)#np.expand_dims(predictions, axis=-1)
        return predictions

    def fit(self,X):
        self.model= sm_ARIMA(X, order=(self.p, self.d, self.q))
        self.model=self.model.fit()
        # print("ARIMA FIT:" , model_fit.summary())

    
    def predict(self,X):
        # todo: add a check to ensure fit transform procedure is being called
        return self.model.predict()

    def set_autoregression_controls(self, delay, horizon):
        self.p = delay  # AR terms
        self.q = delay # moving average 
        assert horizon == 1, 'cannot use horizon more than 1 for ARIMA'