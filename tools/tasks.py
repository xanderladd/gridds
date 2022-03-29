
import abc
from abc import ABCMeta, abstractmethod
import numpy as np
import os
from gridds.tools.metrics import *



TASK_REQUIRED_KEYS = ['name', 'procedure',  'metrics'] # think about what else might go in here
ALLOWED_PROCEDURES = ['fit','fit_transform','predict']

default_autoregression = {
    'name': 'autoregression',
    'procedure': ['fit_transform'],
    'metrics': [mae, rmse],
    'delay': 5, 
    'horizon': 1 
}    

long_autoregression = {
    'name': 'autoregression',
    'procedure': ['fit_transform'],
    'metrics': [mae, rmse],
    'delay': 20,
    'horizon': 5
}    


default_impute = {
    'name': 'impute',
    'procedure': ['fit_transform'],
    'metrics': [mae, rmse]
}    

default_fault_pred = {
    'name': 'fault_pred',
    'procedure': ['fit','predict'],
    'metrics': [binary_crossentropy]
}    

unsupervised_fault_pred = {
    'name': 'fault_pred',
    'procedure': ['predict'],
    'metrics': [binary_crossentropy]
}   