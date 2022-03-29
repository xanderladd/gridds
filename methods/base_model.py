from abc import ABCMeta, abstractmethod
import statsmodels.tsa.tsatools as tsatools
import numpy as np
import copy

class BaseModel(object):
    """ 
    Abstract class representing a generic GP 
    
    """
    __metaclass__ = ABCMeta

    def __init__(self, name):  # , full_name):
        """
        Class initialization.
        Args
            :attr:`name` (string):
            the name attribute of the method.
        TODO: add more generic args?
        """
        self.name = name
        self.multivariate = False
        self.loss = []
        self.horizon = None

    # def __str__(self):
    #     return self.name

    @abstractmethod
    def fit(self, X, y, **kwargs):
        """ 
        Fit hyperparameters using Maximum Likelihood Estimation 
        
        Args:
            X (2d torch.tensor): training data in 2d array form (nsamples,nfeatures)
            y (1d torch.tensor): training data in list form, (nsameples) for each response
        """
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        """         
        Args:
            X (2d torch.tensor): training data in 2d array form (nsamples,nfeatures)
        """
        pass

    @abstractmethod
    def fit_transform(self, X, **kwargs):
        """         
        Args:
            X (2d torch.tensor): training data in 2d array form (nsamples,nfeatures)
        """
        pass

    @staticmethod
    def lagmat_backwards(timeseries, lag):
        if len(timeseries.shape) > 1:
            assert timeseries.shape[1] < 2
        ts_copy = copy.deepcopy(timeseries)
        trim_len = len(timeseries) - lag
        res = np.zeros(shape=(trim_len, lag))
        for l in range(lag):
            res[:,l] = timeseries[l:trim_len+l,0]
        return res, ts_copy[lag:]
    # static?
    def lag_transform(self,timeseries, lag=20, horizon=1):
        orig_dim =  timeseries.shape[-1]
        lagged_timeseries, timeseries = tsatools.lagmat(timeseries, maxlag=int(lag), trim='backward', original='sep')#self.lagmat_backwards(timeseries, lag) #tsatools.lagmat(timeseries, maxlag=int(-lag), trim='backward', original='sep')
        if orig_dim > 1: 
            self.multivariate = True
            lagged_timeseries = lagged_timeseries.reshape(lagged_timeseries.shape[0], lag, orig_dim)
        if  horizon > 1:
            timeseries, _ = tsatools.lagmat(timeseries, maxlag=horizon, trim='backward', original='sep')
        return timeseries, lagged_timeseries
    
    # static?
    def batch_shape(self, timeseries, batch_size=20):
        # if len(timeseries.shape) == 3: # already in batch shape
        #     print("already in batch shape",timeseries.shape )
        #     return timeseries
        keep_inds = len(timeseries) - len(timeseries) % batch_size
        trimmed_timeseries = timeseries[:keep_inds]
        self.trimmed =  len(timeseries) - len(trimmed_timeseries)
        # RETURN shp:   (num_batches, batch_size, num_feats)
        return trimmed_timeseries.reshape(-1, batch_size, trimmed_timeseries.shape[1])

    def batch_reshape(self, batch_timeseries, num_feats=1, nan_pad=True):
        assert batch_timeseries.shape[1] == self.batch_size
        reshaped_ts = batch_timeseries.reshape(-1, batch_timeseries.shape[-1]) # (timeseries len, num_samples)
        if nan_pad: # pad with nans w. shape (trmmed, n_feats)
            print(np.empty((self.trimmed,reshaped_ts.shape[-1])).shape, )
            reshaped_ts = np.append(reshaped_ts,np.empty((self.trimmed,reshaped_ts.shape[-1])), axis=0)
            # reshaped_ts = np.insert(reshaped_ts,0, np.empty((self.trimmed,reshaped_ts.shape[-1])), axis=0)

        return reshaped_ts
        
    @abstractmethod
    def set_autoregression_controls(self, delay, horizon):
        self.batch_size = delay
        self.horizon = horizon

    
    
    
    
    # @abstractmethod
    # def evaluate(self):
    #     """
    #     TODO: decide if we want to keep or leave?
            
    #     """
    #     pass

    # @abstractmethod
    # def set_params(self):
    #     """
    #     TODO: decide if we want to use optuna?
    #     """
    #     pass
    
    # def addTensorBoard(self, writer):
    #     """
    #     Every class will have this function so they can initialize a summary writer
        
    #     Args:
    #         writer (tensboard.SummaryWriter) writer object passed in via experimenter
        
    #     """
    #     self.writer = writer
    #     self.recording = True
        
    # def getNamedParams(self):
    #     """
    #     Every class will have this function so we can log the parameters from the model. 
    #     Also checks that every class has a "self.model" is this a good practice? I think so,
    #     but it could also be a limiting restraint.
        
    #     Returns:
    #         res (dict): mapping of named model parameters to their values
    #     """
    #     assert hasattr(self, "model")
    #     res = {}
    #     for name,parameter in self.model.named_parameters():
    #         try:
    #             res[name] = parameter.item()
    #         except ValueError:
    #             pass # ignoring parameters that don't just have one item for now!
    #     return res
