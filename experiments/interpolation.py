
import sys

sys.path.append('../')
from experimenter import Experimenter
from data.synthetic_data import SyntheticData
from gridds.viz.viz import visualize_output
import gridds.tools.tasks as tasks
import os
import matplotlib.pyplot as plt
from methods import *


if __name__ == '__main__':
    # run experiments from root dir ( one up)
    os.chdir('../')
    dataset = SyntheticData('test')

    # must present multiple features
    reader_instructions = {
        'sources': ['fake_source'],
        'modalities': ['ami'],
        'target': 'fault_present',  ## THIS IS CHALLENGE
        'features': ['feature0','feature1', 'feature2', 'feature3']
    }

    # reader_instructions = {
    #     'sources': ['fake_source1', 'fake_source2'],
    #     'modalities': ['ami'],
    #     'target': 'fault_present',  ## THIS IS CHALLENGE
    #     'features': ['feature0']
    # }



    dataset.prepare_data(reader_instructions)
    dataset.remove_data(chunksize=20, chunks=12)
    methods = [KNN('KNN'),BayesianRidge('Bayesian Ridge'),RandomForest('Random Forest')] #  GP('GP')

    task = tasks.default_impute

    exp = Experimenter('interpolation', runs=3)
    exp.run_experiment(dataset,methods,task=task, clean=True)
    visualize_output(os.path.join('outputs', exp.name))

    