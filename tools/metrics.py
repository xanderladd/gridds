
   
import numpy as np
import sklearn.metrics 


def rmse(y_pred, y):
    return np.nanmean((y - y_pred)**2)

def mae(y_pred, y):
    return np.nanmean(np.abs(y - y_pred))

    
def binary_crossentropy(y_pred,y):
    raise NotImplementedError