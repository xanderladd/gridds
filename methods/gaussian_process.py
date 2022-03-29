from sklearn.gaussian_process import GaussianProcessRegressor
from .base_model import BaseModel
from sklearn.gaussian_process.kernels import RBF, DotProduct, ConstantKernel, ExpSineSquared, \
                                 Matern, WhiteKernel, RationalQuadratic
import numpy as np

class GP(BaseModel):
    def __init__(self, name):  # , full_name):
        """
        Class initialization.
        Args
            :attr:`name` (string):
            the name attribute of the method.
        TODO: add more generic args?
        """
        self.name = name
        super(GP, self).__init__(name)
        '''  
        3 options below for self.model
        '''
        #self.model = KNeighborsRegressor()
        
        # kernel 1
        long_term_trend_kernel = 50.0 ** 2 * RBF(length_scale=50.0)
        seasonal_kernel = (
            2.0 ** 2
            * RBF(length_scale=100.0)
            * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
        )
        irregularities_kernel = 0.5 ** 2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
        noise_kernel = 0.1 ** 2 * RBF(length_scale=0.1) + WhiteKernel(
            noise_level=0.1 ** 2, noise_level_bounds=(1e-5, 1e5)
        )
        non_stationary_kernel =  DotProduct()
        # kernel 2
        kernel = (
            long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel
        )

        # kernel 3
        # smaller lengthscale does better on reconstruction
        kernel = 2 * RBF(length_scale=0.1)  + DotProduct() + WhiteKernel() + ConstantKernel(constant_value=2)
        # kernel 4
        kernel = 1 * RBF(length_scale=.1, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel()

        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, normalize_y=True)

    def fit(self,X, **kwargs):
        y = X 
        x = kwargs['timestamps'].reshape(-1,1)
        self.model.fit(x,y)
    
    def predict(self,X, **kwargs):
        y = X 
        x = kwargs['timestamps'].reshape(-1,1)
        return self.model.predict(x)

    def fit_transform(self,X, **kwargs):
        mask = np.where(np.isfinite(X))[0]
        X_masked = X[mask]
        timestamps_masked = kwargs['timestamps'][mask]
        self.fit(X_masked, timestamps=timestamps_masked)
        return self.predict(X, **kwargs)
    
    # def fit_transform(self,X):
    #     return self.model.fit_transform(X)

    def impute(self,X, **kwargs):
        self.model = KNNImputer(n_neighbors=n_neighbors)

        return self.model.fit_transform(X)

    def transform(self,X, **kwargs):
        return self.model.transform(X)
