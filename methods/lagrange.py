from .base_model import BaseModel
import numpy as np
from scipy import interpolate

class Lagrange(BaseModel):

    def __init__(self, name):  # , full_name):
        """
        Class initialization.
        Args
            :attr:`name` (string):
            the name attribute of the method.
        TODO: add more generic args?
        """
        self.name = name

    def predict(self, y, res=10):
        x = np.arange(len(y))
        # y = y[:,np.where(np.isnan(y).any(axis=0))[0]] 
        """
        There is a problem with using this method -- it does interpolation not extrapolation
        should be a different category
        """
        
        pred_inds = np.arange(0,len(y),2) #np.where(np.isnan(y))[0]
        train_inds = np.arange(0,len(y),res) # use res as interpolation step
        train_inds = np.append(train_inds, len(y) - 1)
        interp = interpolate.lagrange(train_inds,y[train_inds,0])
        curr_interp = interp(pred_inds)
        # y[pred_inds,0] = curr_interp
        import pdb; pdb.set_trace()
        import matplotlib.pyplot as plt
        plt.plot(y[:,0])#[0,:,0])
        plt.plot(curr_interp, label='itp')
        plt.legend()
        plt.ylim(-3,3)

        plt.savefig('tst.png')
        return y[:,0]
