import abc
from abc import ABCMeta, abstractmethod
import numpy as np
import os
from gridds.data.db_interface import DbObject
import copy
from sklearn.model_selection import train_test_split

class Dataset(object):
    """ """
    __metaclass__ = ABCMeta

    def __init__(self, dataset_name):
        """."""
        self.name = dataset_name
        self.data = None
        self.faults = None
        self.timestamps = None
        self.dataset_name = dataset_name

    # TODO: unlock attributes when you can accomodate them
    # @property
    # def data(self):
    #     """
    #     get multi-stream timeseries data
    #     TODO: ensure it is always formatted S x N x D
    #     where S is # of samples, N is # of time points and D is number of streams
    #     values are some sensor measurement. Here is where we have abstraction
    #     """
    #     return self.data
    
    # @property
    # def faults(self):
    #     """
    #     get timestamps of faults
    #     each should be a tuple of start and end
    #     """
    #     return self.faults

    # @property
    # def timesteps(self):
    #     """
    #     get timeseries across entire data
    #     """
    #     return self.timesteps
    def prepare_data(self, reader_instructions=None):
        """
        wrapper function to check prepare data
        """
        self._prepare_data(reader_instructions=reader_instructions)

    @abstractmethod
    def _prepare_data(self):
        """."""
        pass
    

    @abstractmethod
    def shuffle_and_split(self, strategy='shuffle'):
        """."""
        if strategy == 'shuffle':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_pct, random_state=42)
        else:
            train_num = (1-self.test_pct) * len(self.X)
            self.X_train, self.X_test = self.X[:train_num], self.X[train_num:]
            self.y_train, self.y_test = self.y[:train_num], self.y[train_num:]

    @staticmethod
    def save_dfs(df_list, modality, name, save_path='data/csvs'):
        total_dfs = 0
        os.makedirs(os.path.join(save_path, modality, name), exist_ok=True)
        if df_list is None:
            print(f'Nothing to report for {name} {modality}')
            return
        for df in df_list:
            curr_save_path = os.path.join(save_path, modality, name, name + str(total_dfs) + '.csv')
            df.to_csv(curr_save_path)
            total_dfs += 1

        print(f'saved {total_dfs} {name} {modality} dfs to ' + \
            f'{os.path.abspath(os.path.join(save_path,modality,name))}')


    @staticmethod
    def summarize_dfs(df_list, modality, name):
        columns = []
        total_rows = 0
        if df_list is None:
            print(f'Nothing to report for {name} {modality}')
            return

        for df in df_list:
            try:
                columns.append(df.columns)
            except:
                import pdb; pdb.set_trace()
            total_rows += len(df)
        # columns = np.unique(np.array(columns))
        # print('=========================================================')
        # print(f'{name} {modality} columns: {columns} \n')
        #print('----------------------------------------------------------')
        print(f'{name} {modality} has {total_rows} rows \n')
        # print('========================================================= \n\n')

    def ingest_data(self, types=['ami','scada','oms'], ingest_path = "data/csvs", close=True, src_ignore=[]):
        db_obj = DbObject()
        db_obj.connect()
        src_ignore =  src_ignore + ['.ipynb_checkpoints']
        for data_type in types:
            pth = os.path.join(ingest_path,data_type)
            sources = [os.path.join(pth,f) for f in\
                      os.listdir(pth) if f not in  src_ignore]
            fp = True
            if data_type == 'oms': fp = False
            for source_path in sources:
                table_prefix = os.path.basename(source_path)
                code = db_obj.create(data_type=data_type,path=source_path,\
                                prefix=table_prefix,replace=True, field_permissive=fp)
                if not code:
                    db_obj.ingest(data_type=data_type,path=source_path, \
                    prefix=table_prefix,replace=True, field_permissive=fp)
        if close: # set close to false if you wanna do things after this cmd
            db_obj.conn.close()

    def remove_data(self, chunksize=20, chunks=10):
        self.y = self.y.astype(float)
        self._X = copy.deepcopy(self.X )
        self._y =  copy.deepcopy(self.y)
        for chunk in range(chunks):
            chunk_end = int(np.random.uniform(chunksize,len(self.X)))
            chunk_start = chunk_end - chunksize
            self.X[chunk_start:chunk_end] = np.nan
            self.y[chunk_start:chunk_end] = np.nan
    