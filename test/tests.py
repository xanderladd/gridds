 
 
    
    
# test case
# ami_df[ami_df['start_time'] == pd.to_datetime('2019-07-30 20:00:00')]['start_time'].values <  oms_df[oms_df['customers_affected'] == '583']['start_time'].values
    
# ami_df[ami_df['start_time'] == pd.to_datetime('2019-07-30 20:00:00')]['end_time'].values >  oms_df[oms_df['customers_affected'] == '583']['end_time'].values

#lagged_timeseries[20] == time_series[0] for delay 20


# test if VAE can handle several different lengths to be robust