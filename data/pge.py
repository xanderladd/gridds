import os
import numpy as np
import pandas as pd

class PMUFault(Dataset):
    def __init__(self, options=[], fault=False, root='/usr/gapps/gmlc/pge/amlies/'):
        
        self.fault = fault
        self.root = root
        self.folder = 'faulty' if fault == True else 'normal'
        self.load_path = os.path.join(self.root, f'{self.folder}_hours')
        self.filenames = os.listdir(self.load_path)
        if not len(options):
            self.options = options
        else:
            self.options  = os.listdir()
        
        self.prepare_data(options)
        
    def _prepare_data(self, options):
        timesteps, data, targets, dfs = [], [], [], []
        for option in options:
            if f'{option}.csv' in self.filenames:
                filepath = os.path.join(self.load_path, f'{option}.csv')
                df = pd.read_csv(filepath)
                
                timesteps.append(df[['second']].to_numpy())
                data.append(df[['L1E Avg Voltage',
                                'L2E Avg Voltage',
                                'L3E Avg Voltage',
                                'L1 Avg Current',
                                'L2 Avg Current',
                                'L3 Avg Current']].to_numpy())
                targets.append(df[['fault_present']])
            else:
                raise ValueError(f"Option not found: {option}. Available options are: {self.filenames}")
        self.timesteps = np.stack(timesteps)
        self.x = np.stack(data)
        self.y = np.stack(targets)
        self.dfs = dfs
        print(self.timesteps.shape, self.data.shape, self.targets.shape)
    
    def get_data(self, idx=None):
        if idx is not None:
            return np.expand_dims(self.data[idx], 0)
        return self.data

    def get_timesteps(self, idx=None):
        if idx is not None:
            return np.expand_dims(self.timesteps[idx], 0)
        return self.timesteps

    def get_faults(self, idx=None):
        if idx is not None:
            return np.expand_dims(self.targets[idx], 0)
        return self.targets
