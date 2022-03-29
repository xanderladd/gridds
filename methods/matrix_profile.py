import os
from ddcrf.run_ddcrf import run_ddcrf
import time
from sklearn import preprocessing
import matrixprofile as mp


class matrix_profile:
    def __init__(self):
        pass

    def predict(self,site_data, **kwargs):
        # roi_start = kwargs['roi_start'] 
        # roi_end = kwargs['roi_end'] 
        data_scaled = preprocessing.scale(site_data)
        # ami_df = df_pair[0].loc[roi_start:roi_end-1]
        profile, figures = mp.analyze(data_scaled, n_jobs=-1)
        # ami_id = ami_df['element_id'].unique()[0]
        # os.makedirs(f'figures/southside_ec/ami/{ami_id}_profile', exist_ok=True)
        # save_path = f'figures/southside_ec/ami/{ami_id}_profile'
        # for idx, figure in enumerate(figures):
            # figure.savefig(os.path.join(save_path,f'figure{idx}.png'))

        import pdb; pdb.set_trace()
        return res