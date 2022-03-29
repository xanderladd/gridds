import os
from ddcrf.run_ddcrf import run_ddcrf
import time
from .base_model import BaseModel

class ddcrf(BaseModel):
    def __init__(self, name, n_iters=10, alpha=10, gamma=1 ):
        super(BaseModel, self).__init__()
        self.n_iters = n_iters
        self.alpha = alpha
        self.gamma = gamma
        self.name = name

    def predict(self,site_data,**kwargs):
        start = time.time()
        # TODO: later enter a save path to handle this
        if os.path.isfile('./ddcrf_sample.hdf5'):
            os.remove('./ddcrf_sample.hdf5')
        res = run_ddcrf(site_data, n_iters=self.n_iters, alpha=self.alpha, gamma=self.gamma, multivariate=True)
        end = time.time()
        
        return res