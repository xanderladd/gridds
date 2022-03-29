import numpy as np
import h5py
from sklearn import preprocessing
from scipy.stats import invwishart, multivariate_normal
from sklearn.gaussian_process.kernels import Matern
from data.base_dataset import Dataset
import pandas as pd
################################################################################
# synthetic Data
################################################################################
def generate_coefficients(num_coeff):
    filter_stable = False
    while not filter_stable:
        true_theta = np.random.random(num_coeff) - 0.5
        coefficients = np.append(1, -true_theta)
        # check if magnitude of all poles is less than one
        if np.max(np.abs(np.roots(coefficients))) < 1:
            filter_stable = True
    coeffs = true_theta
    return coeffs

def generate_ar_data(num_coeff,pts_per_cycle, scale=1, noise_var=.01, num_cycles=3, coeffs=[], initial_pts=[]):
    # we add 3 cycles which we will later cut off later
    num_cycles += 3 
    y = np.zeros(pts_per_cycle*num_cycles)
    # initialize data points for first cycle
    if len(initial_pts) < 1:
        y[:pts_per_cycle] = np.random.randn(pts_per_cycle)
    else:
        y[:pts_per_cycle] = initial_pts
    if len(coeffs) < 1:
        coeffs = generate_coefficients(num_coeff)
    bayes_preds = np.copy(y)
    
    # + 3*self.num_prev because we want to cut first (3*self.num_prev) datapoints later
    # so dist is more stationary (else initial num_prev datapoints will stand out as diff dist)
    for i in range(num_cycles*pts_per_cycle - num_coeff ):
        # Generate y value if there was no noise
        # (equivalent to Bayes predictions: predictions from oracle that knows true parameters (coefficients))
        bayes_preds[i + num_coeff] = np.dot(y[i: num_coeff+i][::-1], coeffs)
        # Add noise
        y[i + num_coeff] = bayes_preds[i + num_coeff] + noise_var * np.random.randn()
        if i > 1000:
            noise_var = .1
        if i < 1025:
            noise_var = .01

    # Cut first  3 cycles so dist is roughly stationary
    bayes_preds = bayes_preds[(pts_per_cycle*(num_cycles-3)):] * scale
    
    y = y[(pts_per_cycle*(num_cycles-3)):] * scale
    return y, bayes_preds
    
def generate_coefficients(num_coeff):
    filter_stable = False
    while not filter_stable:
        true_theta = np.random.random(num_coeff) - 0.5
        coefficients = np.append(1, -true_theta)
        # check if magnitude of all poles is less than one
        if np.max(np.abs(np.roots(coefficients))) < 1:
            filter_stable = True
    coeffs = true_theta
    return coeffs

def generate_multivar_ar_data(num_coeff,pts_per_cycle, num_feat=3, scale=1, noise_var=.001, correlation="High", num_cycles=3):
    # we add 3 cycles which we will later cut off later
    num_cycles += 3 
    y = np.zeros((pts_per_cycle*num_cycles, num_feat))
    # initialize data points for first cycle
    y[:pts_per_cycle,:] = np.random.randn(pts_per_cycle,num_feat)
    coeffs = np.repeat(generate_coefficients(num_coeff).reshape(-1,1),num_feat,axis=1) # coeffs are same for each feature on purpose
    bayes_preds = np.copy(y)
    
    # how to define MVN? TODO: set this param better
    df = 10**noise_var * num_feat
    scale1 = num_feat * np.ones((num_feat, num_feat))
    scale1[np.diag_indices_from(scale1)] = df
    Sigma_k = invwishart(df=df, scale=scale1).rvs()
       
    if correlation == "Low":
        Sigma_k = Sigma_k * np.eye(num_feat)
    else:
        Sigma_k =  Sigma_k * np.array([[1,1,0,0],[1,1,0,0],[0,0,1,1], [0,0,1,1]]) 


    means = np.zeros(num_feat)
    
    mvn = multivariate_normal(means, Sigma_k) # using one multivariate normal in place of f_0 for now... (note no base dist)
    # + 3*self.num_prev because we want to cut first (3*self.num_prev) datapoints later
    # so dist is more stationary (else initial num_prev datapoints will stand out as diff dist)
    for i in range(num_cycles*pts_per_cycle - num_coeff ):
        # Generate y value if there was no noise
        # (equivalent to Bayes predictions: predictions from oracle that knows true parameters (coefficients))
        bayes_preds[i + num_coeff] = np.sum(np.multiply(y[i: num_coeff+i][::-1], coeffs), axis=0)
        # Add noise
        noise = mvn.rvs(size=1)    
        y[i + num_coeff] = bayes_preds[i + num_coeff] + noise_var * noise

    # Cut first  3 cycles so dist is roughly stationary
    bayes_preds = bayes_preds[(pts_per_cycle*(num_cycles-3)):] * scale
    y = y[(pts_per_cycle*(num_cycles-3)):] * scale
    return y, bayes_preds

def npSigmoid(x):
    """
    numpy sigmoid: \frac{1}{1 + e^{-x}}
    """
    return 1/(1 + np.exp(-x))

def autocorr_transformer_data(n, d, failures=4, interval_choice="random",correlation="High",show=False):
    """
    geenerate fake transformer data using 
    Args:
        - n (int): number of time points
        - d (int): dimensionality of meausrements (features)
        - f0 (function): function that defines base distribution
        - f1 (function): function that defines failure distribution
    
    TOOD: consider args that govern failure time, means/modes, stddevs
        
    """
    max_fail_time = int((n/(failures/2)) / (failures)) # can only be in failure state for ? of time for a given failure
    min_fail_time = int((n/failures) / (failures))

    timestamps = np.arange(n)
    if interval_choice == "random":
        # TODO: chance correlation across code
        start_cps = np.random.choice(n, size=failures)

        # if correlation == "High":
        #     start_cps = np.random.choice(n, size=failures)
        # elif correlation == "Medium":
        #     start_cps = np.random.choice(n//2, size=failures)
        #     start_cps = np.append(start_cps, )
    elif interval_choice == "uniform":
        start_cps = np.linspace(0, n-max_fail_time, failures).astype(int)
    else:
        assert 1==0, "not valid interval choice"
    
    all_cps = []
    means = [.03,.05, .07,.08] # TODO: random parameter choice, allow to be set 
    fail_means = np.array(means) * 1.8# TODO: random parameter choice, allow to be set 
    x = np.linspace(0, 8*np.pi, d)
    y = np.linspace(0, 8*np.pi, n)
    xv, yv = np.meshgrid(x, y)
    # min(fail_means)*np.sin(yv) + 
    data,_ = generate_multivar_ar_data(num_coeff=5, pts_per_cycle=n,\
                                                  num_cycles=-2, num_feat=d, scale=means, correlation=correlation, noise_var=1)
    for failure_idx in range(failures):
        curr_fail_time = max(min_fail_time,int(min(np.random.wald(mean=max_fail_time, scale=30),max_fail_time))) # fail at least 6 seconds  but less than 10
        start_idx, end_idx = start_cps[failure_idx],  start_cps[failure_idx] + curr_fail_time
        all_cps += [start_idx,end_idx]
        if end_idx >= n:
            end_idx = n
            curr_fail_time = n - start_cps[failure_idx]
            
        failure, _ = generate_multivar_ar_data(num_coeff=5, pts_per_cycle=curr_fail_time,\
                                                  num_cycles=-2, num_feat=d, scale=fail_means, noise_var=.3)
        failure = np.cumsum(np.abs(failure),axis=0)
        s = .6
        g_x = np.linspace(-.5*np.pi, 1.4*np.pi, curr_fail_time)
        
        fail_weight = np.repeat(npSigmoid((np.sin(g_x))/s).reshape(-1,1), d, axis=1) 
#debug
#         print(failure.shape, "FAIL SHAPE")
#         print(start_idx, end_idx)
#         print(fail_weight.shape, "Fail wt shp")
#         print((fail_weight*failure).shape)
        data[start_idx:end_idx] = fail_weight*failure + (1-fail_weight)*data[start_idx:end_idx]
    data = data[:n,:]
    return data, all_cps
     

def regular_downsample(data,cps,save_to_f=None, postfix=None):
    keep_pctg =int(np.random.uniform(.6,.8) *len(data))# percentage of data to keep for this feature
    sampled_inds = np.sort(np.random.choice(np.arange(len(data)),replace=False, size=keep_pctg))
    downsampled_ts = data[sampled_inds]
    adjusted_cps = []
    for cp in cps:
        missing_inds = np.sum([1 for i in range(cp) if i not in sampled_inds])
        adjusted_cps.append(cp-missing_inds)
    if save_to_f:
        sampled_inds = np.repeat(sampled_inds.reshape(-1,1),data.shape[1], axis=1)
        #stacked_adjusted_cps = np.repeat(np.array(adjusted_cps).reshape(-1,1),data.shape[1], axis=1)

        save_to_f.create_dataset("sampled_inds_{}".format(postfix), data=sampled_inds)
    return downsampled_ts, adjusted_cps  

def irregular_downsample(data,cps,save_to_f=None, postfix=None):
    sampled_inds = []
    all_adjusted_cps = []
    all_downsampled_ts = []
    for feature_idx in range(data.shape[1]):
        keep_pctg =int(np.random.uniform(.6,.8) *len(data))# percentage of data to keep for this feature
        curr_sampled_inds = np.sort(np.random.choice(np.arange(len(data)),replace=False, size=keep_pctg))
        full_ts = np.zeros(data.shape[0])
        full_ts[curr_sampled_inds] = data[curr_sampled_inds,feature_idx]
        # TODO: value for missing data is weird - 999 ?
        full_ts[~np.isin(np.arange(len(full_ts)),curr_sampled_inds)] = 0
        #downsampled_ts = data[curr_sampled_inds,feature_idx]
        # TODO: clean this and above
        downsampled_ts = full_ts
        adjusted_cps = []
        for cp in cps:
            missing_inds = np.sum([1 for i in range(cp) if i not in curr_sampled_inds])
            adjusted_cps.append(cp - missing_inds)
        sampled_inds.append(curr_sampled_inds)
        all_adjusted_cps.append(adjusted_cps)
        all_downsampled_ts.append(downsampled_ts)
    if save_to_f:
        dt = h5py.special_dtype(vlen=np.dtype('float64'))
        save_to_f.create_dataset("sampled_inds_{}".format(postfix),len(sampled_inds),  dtype=dt)
        save_to_f["sampled_inds_{}".format(postfix)][...] = sampled_inds
    return np.array(all_downsampled_ts).T, np.array(all_adjusted_cps)   

def generate_site_data(num_sites=1000, timesteps=500,num_feats=4, irregular=False, correlation="High", save_to_f=None):
    full_site_data = []
    full_cp_data = []
    site_data = []
    cp_data = []
    
    for site in range(num_sites):
        if not site % 50:
            print(site)
        curr_feature, curr_cps = autocorr_transformer_data(timesteps, num_feats,correlation=correlation, interval_choice="uniform" )
        full_site_data.append(curr_feature)
        full_cp_data.append(curr_cps)
        # save downsampled
        if not irregular:
            curr_feature, curr_cps = regular_downsample(curr_feature,curr_cps, save_to_f, postfix=site)
        else:
            curr_feature, curr_cps = irregular_downsample(curr_feature,curr_cps, postfix=site)
        site_data.append(curr_feature)
        cp_data.append(curr_cps)
    if save_to_f:
        save_to_f.create_dataset("full_site_data", data=full_site_data)
        save_to_f.create_dataset("full_cp_data", data=full_cp_data)
        save_to_f.create_dataset("cp_data", data=cp_data)
        save_to_f.create_dataset("site_data", data=site_data)
    return np.array(site_data), np.array(cp_data), np.array(full_site_data), np.array(full_cp_data)
    

class SyntheticData(Dataset):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.verbose = True
        # un hardcode later
        self.test_pct = .3
        
    def generate_data(self, n_hours, n_feats):
        # ADD more features like correlation
        """
        PSUEDO:

        1. generate data(args)
        2. make data into similarly formatted table
        3. make faulty by holding out hours
        4. pd.save_csv(load_path) for faulty and normal
        """
        num_sites = 1
        site_data, cp_data, full_site_data, full_cp_data = generate_site_data(1, n_hours, num_feats=n_feats)
        # TODO
        # works only for one site
        cp_data = cp_data[0]
        # TODO: pass in column names once args are set up
        site_data = {f'feature{i}': site_data[0,:,i]  for i in range(site_data.shape[-1])}
        site_df = pd.DataFrame(site_data)
        site_df['fault_present'] = 0
        status = 0
        for i in range(len(cp_data)-1):
            site_df.loc[cp_data[i]: cp_data[i+1], 'fault_present'] = status 
            status = (status + 1) % 2
        self.site_dfs = [site_df]


    def prepare_data(self, reader_instructions):
        # hardcoded
        n_hours=2000
        n_feats = len(reader_instructions['features'])
        n_sites = len(reader_instructions['sources'])
        # MONKEY PATCH
        if n_feats < 4:
            keep_fts = n_feats
            n_feats = 4
        else:
            keep_fts = n_feats
        self.generate_data(n_hours, n_feats)
        # later do loop and irregular concatenate
        # for now we have just one site.
        self.X = self.site_dfs[0][reader_instructions['features']].values[:,:keep_fts]
        self.y = self.site_dfs[0][reader_instructions['target']].values
        # timesteps are just  sequential
        self.timestamps = np.arange(len(self.y))


    def get_data(self, idx=None):
        if idx is not None and len(self.data.shape) == 3:
            return np.expand_dims(self.X[idx], 0)
        return self.data

    def get_timesteps(self, idx=None):
        if idx is not None:
            return np.expand_dims(self.timesteps[idx], 0)
        return self.timesteps

    def get_faults(self, idx=None):
        if idx is not None:
            return np.expand_dims(self.y[idx], 0)
        return self.y

