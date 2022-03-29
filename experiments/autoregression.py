
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

    reader_instructions = {
        'sources': ['fake_source'],
        'modalities': ['ami'],
        'target': 'fault_present',  ## THIS IS CHALLENGE
        'features': ['feature0','feature1', 'feature2', 'feature3']
    }

    # reader_instructions = {
    #     'sources': ['fake_source'],
    #     'modalities': ['ami'],
    #     'target': 'fault_present',  ## THIS IS CHALLENGE
    #     'features': ['feature0']
    # }


    dataset.prepare_data(reader_instructions)
 

    # methods = [LSTM('LSTM', train_iters=1),  ARIMA('ARIMA'),  VanillaRNN('RNN', train_iters=1)]#, LSTM('LSTM')]
    methods = [VRAE('VRAE',train_iters=2, batch_size=5)]
    
    settings = {
                
                'plot_ami_meters': False,
                'overlay_faults': True, # only works if you plot AMI meters
                'plot_faults': True, # not in use
                'sankey': False,
                'temporal': False, # only works if you do sankey
                'oms_scatter': False,
                
                }
    
    exp = Experimenter('autoregression')
    exp.run_experiment(dataset,methods, task=tasks.default_autoregression, clean=True)
    visualize_output(os.path.join('outputs', exp.name))

    