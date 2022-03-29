
import sys

from gridds.experimenter import Experimenter
from gridds.data import *
from gridds.viz.viz import visualize_output
import os
import matplotlib.pyplot as plt
from gridds.methods import *
import gridds.tools.tasks as tasks


if __name__ == '__main__':
    # run experiments from root dir ( one up)
    os.chdir('../')
    dataset = SmartDS('test', sites=1, normalize=False, size=2000)

    reader_instructions = {
        'sources': ['P1U'],
        'modalities': ['load_data'],
        'target': '',  ## how to get faults  for NREL?
        # 'features': ['feature0','feature1', 'feature2', 'feature3']
        'replicates': ['customers']
    }

    dataset.prepare_data(reader_instructions)
    # real task
    # methods = [ARIMA('ARIMA')]
    methods = [ VanillaRNN('RNN', train_iters=600, learning_rate=.001, batch_size=5, hidden_size=32)]
    methods += [ARIMA('ARIMA')]
    methods += [VRAE('VRAE',train_iters=100, batch_size=5)]
    methods  += [LSTM('LSTM', train_iters=500, batch_size=10, learning_rate=.005, layer_dim=2, hidden_size=16)]
    # smoke test
    # methods = [ VanillaRNN('RNN', train_iters=10, learning_rate=.01, batch_size=10, hidden_size=16)]
    # methods += [ARIMA('ARIMA')]
    # methods += [VRAE('VRAE',train_iters=3, batch_size=5)]
    # methods  += [LSTM('LSTM', train_iters=3, batch_size=3, learning_rate=.008, layer_dim=2, hidden_size=16)]

    exp = Experimenter('nrel', runs=1)
    exp.run_experiment(dataset,methods,task=tasks.default_autoregression, clean=False)
    visualize_output(os.path.join('outputs', exp.name))

    