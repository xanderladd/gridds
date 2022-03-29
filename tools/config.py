import os
import pathlib
from .private import *

path_to_config = pathlib.Path(__file__).parent.resolve()
base_dir =  path_to_config
data_dir = os.path.join(path_to_config, 'data')


SMOKE_TEST = True

AMI_CANONICAL = ['element_id' , 'feeder' , \
    'map_location', 'KWH' , 'time']

SCADA_CANONICAL = ['element_id','element_name', 'outage_start', \
    'outage_end',  'duration', 'customers_affected',   'cause',  \
        'cause_code',
        'map_location']



method_colors = {
                'RNN':'red',
                 'ARIMA':'blue',
                  'LSTM':'green',
                  'LSTM_VAE':'purple',
                  'KNN':'red',
                  'VRAE': 'brown',
                 'iterative':'blue',
                  'Random Forest':'green',
                  'GP': 'orange',
                  'Bayesian Ridge': 'grey'
                  }

method_markers = {'RNN':'o', 
                'ARIMA':'v', 
                'LSTM':'P',
                'LSTM_VAE':'+',
                'KNN':'o', 
                'VRAE': 's',
                'iterative':'v', 
                'RF':'P',
                'GP': '-'
                }


method_styles = {'RNN':'dotted',
                 'ARIMA':'dashed',
                 'LSTM':'dashdot',
                 'KNN':'dotted', 
                 'LSTM_VAE':'solid',
                'iterative':'dashed', 
                'VRAE': 'solid',
                'GP': 'solid',
                'Bayesian Ridge': 'dashdot',
                'Random Forest':'dashdot', }
