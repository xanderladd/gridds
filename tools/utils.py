import pandas as pd
import numpy as np
import re
import os


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def un_nest_dict(dct):
    df = pd.json_normalize(dct, sep='_')
    dct = df.to_dict(orient='records')[0]
    return dct 

def add_gitignore(path):
    with open(os.path.join(path,'.gitignore'),'w') as f:
        f.write('*')

def pad_lists(lst1,lst2, axis=0):
    npad = [(0, 0)] * lst1.ndim
    npad[axis] = (0, abs(len(lst2) - len(lst1)))
    if len(lst1) > len(lst2):
        lst2 = np.pad(lst2, pad_width= npad, mode='constant', constant_values=0 )
    elif len(lst1) < len(lst2):
        lst1 = np.pad(lst1, pad_width = npad, mode='constant', constant_values=0  )
    return lst1, lst2

def flexible_concat(lst1, lst2, dim=0):
    lst1, lst2 =  np.array(lst1), np.array(lst2)
    if len(lst1.shape) == 1:
        lst1 = lst1.reshape(-1,1)
    if len(lst2.shape) == 1:
        lst2 = lst2.reshape(-1,1)
    
    if len(lst1) < 2 :
        return lst2
    elif len(lst2) < 2:
        return lst1
    else:
        # res = np.concatenate([lst1, lst2], axis=dim)
        lst1, lst2 = pad_lists(lst1, lst2)
        res = np.concatenate([lst1, lst2], axis=dim)

    return res
    
def drop_level_combine(df):
    df.columns = ['_'.join((col[0], str(col[1]))) for col in df.columns]
    return df

def plus_minus_cols(df, main, std, drop=True, ci=False):
    # if ci:
        # df[std] = 1.96*df[std] / np.sqrt(df[std.replace('std','count')].unique()[0])
    df[main] = df[main].astype(str).apply(lambda x: x[:5]) \
    + " ± " + df[std].astype(str).apply(lambda x: x[:5])
    if drop:
        df = df.drop(std,axis=1)
        df = df.drop(std.replace('std','count'),axis=1)
    return df


def listdir_only(base_path):
    return [elem for elem in os.listdir(base_path) if os.path.isdir(os.path.join(base_path,elem))]

def check_task(task):
    """
    Returns true for now. TODO: implement task check later
    """
    return True
def merge_ami_oms(ami_df, oms_df):
    '''
    TODO: once you can meaningfully merge in OMS data using Ids
    then you will need this for confirmed outages
    '''
    raise NotImplementedError
    extra_columns = ['duration', 'customers_affected', 'cause', 'cause_code', 'map_location']
    import pdb; pdb.set_trace()
    for row in oms_df.to_dict(orient='records'):
        start, end = row['start_time'], row['end_time']

def add_fault_valleys(ami_df, oms_df):
    # vectorized approach, faster, but it had a bug -- could it be the parens around datetime addition?
    # ami_df['feeder_fault_present'] = ami_df.assign(key=1).merge(oms_df.assign(key=1),on='key').drop('key', 1)\
    #                                     .assign(Event=lambda v: (v['end_time_x'] <= v['end_time_y'] + pd.DateOffset(hours=10)) & (v['start_time_x'] >= v['start_time_y'] - pd.DateOffset(hours=10)))\
    #                                     .groupby('start_time_x', as_index=False)['Event'].any()['Event']
    ami_df['feeder_fault_present'] = False
    for row in oms_df.to_dict(orient="records"):
        condition1 = ((ami_df['end_time'] <= (row['end_time'] + pd.DateOffset(hours=10)))\
            & (ami_df['start_time'] >= (row['start_time'] - pd.DateOffset(hours=10))))
        ami_df.loc[condition1, 'feeder_fault_present'] = True

    ami_df['fault_present'] = False
    ami_df.loc[(ami_df['feeder_fault_present'] == True) & (ami_df['kwh'].astype(float) == 0.0), 'fault_present'] = True
    return ami_df

def flatten_list(lst):
    res = []
    [res.append(elem[0]) for elem in lst]
    return res

def add_feeder_sub(fs_dict, feeder, substation, item):
    if substation in fs_dict:
        if feeder in fs_dict[substation]:
            fs_dict[substation][feeder].append(item)
        else:
            fs_dict[substation][feeder] = [item]
    else:
        if type(item) != list:
            fs_dict[substation] = {feeder: [item]}
        else:
            fs_dict[substation] = {feeder: item}
    return fs_dict

def format_sub_feeder_tuples(lst):
    res = []
    for elem in lst:
        elem = elem[1:-1] # remove parens
        # elem = re.sub(r"'","", elem)
        elem_list = elem.split(',')
        res.append((int(elem_list[0]), int(elem_list[1]), elem_list[2]))
    return res

