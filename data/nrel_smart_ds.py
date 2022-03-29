import numpy as np
import h5py
from sklearn import preprocessing
from gridds.data.base_dataset import Dataset
import pandas as pd
import os
from torch.nn import functional
from sklearn.preprocessing import normalize
from gridds.utils.utils import *



class SmartDS(Dataset):
    def __init__(self, dataset_name, location='AUS', size=1000, sites=3, test_pct=.2, normalize=False):
        super().__init__(dataset_name)
        self.verbose = True
        self.base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nrel_smart_ds', location)
        self.size = size
        self.sites = sites
        self.X = []
        self.y  = []
        self.timestamps =  []
        self.test_pct = test_pct
        self.dt  = []
        self.names = []
        self.normalize =  normalize
    

    def download_data(self):
        raise NotImplementedError
    
    def _prepare_data(self, reader_instructions):
        for source in reader_instructions['sources']:
            for mode in reader_instructions['modalities']:
                wrkDir = os.path.join(self.base_path,source, mode)
                max_val =  1000
                while max_val > 30:
                    parquet_dfs = []
                    for pqt in np.random.choice(os.listdir(wrkDir),self.sites): # choose self. random sites
                        curr_df  =pd.read_parquet(os.path.join(wrkDir,pqt), engine='pyarrow')
                        curr_df['ID'] = pqt.replace('.parquet','')
                        site_len = len(curr_df)
                        parquet_dfs.append(curr_df)
                        full_df = pd.concat(parquet_dfs)
                        max_val = full_df['total_site_electricity_kw'].values.max()

        # df = pd.concat(parquet_dfs)
        selected_ind = np.random.choice(np.arange(self.size,site_len))
        selected_inds = np.arange(selected_ind-self.size, selected_ind)
        

        for df in parquet_dfs:
            # TODO: could parameterize selected param
            X = df['total_site_electricity_kw'].values[selected_inds].reshape(-1,1)
            y = np.zeros_like(X).reshape(-1,1)
            timestamps = df['Time'].values[selected_inds].reshape(-1,1)
            dt = .02
            name = df['ID']
            self.X = flexible_concat(self.X, X, dim=-1)
            self.y  = flexible_concat(self.y, y, dim=-1)
            self.timestamps =  flexible_concat(self.timestamps, timestamps, dim=1)
            self.dt  = flexible_concat(self.dt, [dt], dim=-1)
            self.names = flexible_concat(self.names,[name], dim=-1)
        if self.normalize:
            self.X = normalize(self.X, axis=1)
            self.y = normalize(self.y, axis=1)
               




        reader_instructions['modalities']
        reader_instructions['replicates']

        # {'sources': ['P1U'], 'modalities': ['load_data'], 'target': '', 'replicates': ['customers']}

        
    
