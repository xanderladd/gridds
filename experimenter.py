#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ladd12
"""
import os
import types
import shutil
import json
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
matplotlib.rcParams.update({'font.size': 11})
from matplotlib.ticker import FormatStrFormatter
matplotlib.rcParams.update({'figure.autolayout': True})
from abc import ABCMeta, abstractmethod
import numpy as np
from ddcrf.run_ddcrf import run_ddcrf
import gridds.tools.utils as utils
from sklearn import preprocessing
from gridds.data.db_interface import DbObject
import time
import gridds.tools.config as cfg 
# from gridds.tools.metrics import *

import gridds.viz.viz as viz
import copy
import pickle






class Experimenter(object):
    """ 
    """

    def __init__(self, name, runs=1):

        assert isinstance(name, str)

        self.name = name
        self.dataset = None
        self.methods = None
        self.metrics = None
        self.nb_runs = runs
        self.run_num = 0
        


    def cache_data(self, res, site_data, method, output_path):
        save_path = os.path.join(output_path, os.path.basename(output_path) + '.pkl' )
        
        data_cache  = {}
        data_cache['ground_truth'] = site_data
        data_cache['predicted'] = res
        data_cache['method_class'] = type(method)
        data_cache['method_name'] = method.name
        data_cache['train_loss'] = method.loss
        
        with open(save_path, 'wb') as f:
            pickle.dump(data_cache,f)


    """
    TODO: variable naming here is a disaster
    replace res,  df, site_data
    """
    def postprocess_result(self, df, predictions, ground_truth, \
                            method, task, roi_inds=None):
        output_path = os.path.join('outputs',self.name,str(self.run_num),method.name)
        os.makedirs(output_path, exist_ok=True)


        # df update
        dct = {}
        dct['method_name'] = method.name
        for metric_func in task['metrics']:
            dct[metric_func.__name__] = metric_func(predictions,ground_truth)
        # site_data = site_data[:len(res)]
        df = df.append(dct, ignore_index=True)
        # saving DF all the time
        df.to_csv(os.path.join(os.path.dirname(output_path),'results.csv'), index=False)
        # cache data
        self.cache_data(predictions, ground_truth, method, output_path)
        
        # reset train loss 
        method.loss = []
        
        return df
        
    @staticmethod
    def cache_task(task, name):
        output_path = os.path.join('outputs', name,'task.pkl')
        with open(output_path,'wb') as f:
            pickle.dump(task, f)


    
    def run_experiment(self, dataset, methods, task, clean=False):
        assert utils.check_task(task), f"task {task} does not meet specifications"\
        
        if clean: # deletes previous run directory
            shutil.rmtree(os.path.join('outputs', self.name), ignore_errors=True)
        roi_inds = []
        df = pd.DataFrame(columns=['method_name'] + [met.__name__ for met in task['metrics']])
        for run in range(self.nb_runs):
            dataset.shuffle_and_split()
            for method in methods:
                if 'delay' in task.keys():
                    assert 'horizon' in task.keys(), "must specify horizon with lag"
                    method.set_autoregression_controls(task['delay'], task['horizon'])
                # TODO: might need to redo this logic
                if 'fit_transform' in task['procedure']:
                    prediction = method.fit_transform(dataset.X, timestamps=dataset.timestamps) # feeds entire timeseries (no train/test)
                    ground_truth = dataset.X
                    if np.isnan(ground_truth).any():
                        roi_inds = np.where(np.isnan(ground_truth))[0]
                        ground_truth = dataset._X # pulls back up X val
                elif 'predict' in task['procedure']:
                    method.fit(dataset.X_train, dataset.y_train, timestamps=dataset.timestamps)
                    prediction = method.predict(dataset.X_test, timestamps=dataset.timestamps)
                    ground_truth = dataset.y_test
                
                df = self.postprocess_result(df, prediction, ground_truth, method, task, roi_inds=roi_inds)
            # run iteration loop
            self.run_num += 1
        self.cache_task(task, self.name)
        print(df)
            # res is a binary timeseries indicating 0,1 (fault detected fault not detected)
            # res could be a confidence score as well


       

