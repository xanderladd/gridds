import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
from chart_studio import plotly as pltly
import plotly.graph_objs as go
# Offline mode
import copy
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go
import urllib, json
import gridds.tools.utils as utils
import matplotlib.cm as cm
import matplotlib
import pickle
import gridds.tools.config as config
# init_notebook_mode(connected=True)

# def plot_fault(ami_df, oms_df, window=180, name='name'):
#     oms_df['outage_start'] = pd.to_datetime(oms_df['outage_start'])
#     oms_df['outage_end'] = pd.to_datetime(oms_df['outage_end'])
#     c1 = (oms_df['Sub'] == 8)
#     c2 = (oms_df['feeder'] == 2)
#     c3 = ~oms_df['cause'].str.contains('Scheduled')
#     # this should probably go in a better spot
#     ami_df['start_time'] = pd.to_datetime(ami_df['start_time'])
#     ami_df['end_time'] = pd.to_datetime(ami_df['end_time'])
#     grp_df = []
#     choice_range = np.arange(len(oms_df[c1 & c2 & c3]))

#     # while len(grp_df) < 1:
#         # random_idx = np.random.choice(choice_range)
#     for random_idx in choice_range:
#         t1, t2 = oms_df[c1 & c2 & c3].reset_index()['outage_start'].iloc[random_idx], oms_df[c1 & c2 & c3].reset_index()['outage_end'].iloc[random_idx]
#         sub = oms_df[c1 & c2 & c3].reset_index().iloc[random_idx]['Sub']
#         feeder = oms_df[c1 & c2 & c3].reset_index().iloc[random_idx]['feeder']

#         t1_orig, t2_orig = t1, t2
#         t1, t2 = t1 - datetime.timedelta(minutes = window), t2 + datetime.timedelta(minutes = window)
        
#         c4 = (ami_df['Substation'] == sub)
#         c5 = (ami_df['feeder'] == feeder)
#         c6 = (ami_df['start_time'] > t1)
#         c7 = (ami_df['end_time'] < t2)
#         # print(t1,t2, (c6 & c7).sum(), "SUM")
#         curr_time_df = ami_df[c4 & c5 & c6 & c7]
        
#         if len(curr_time_df) < 1:
#             continue

#         print(random_idx)
#         grp_df = curr_time_df.groupby('start_time').agg(sum).reset_index()

#         curr_outage = oms_df[c1 & c2 & c3].reset_index().loc[random_idx]
#         fig = plt.figure()
#         plt.plot(grp_df['start_time'], grp_df['KWH'].values)
#         # plt.plot(curr_time_df['start_time'], curr_time_df['KWH'].values)
#         plt.title(name + f' Sum of Meters for Feeder {feeder}  \n Cause: ' + curr_outage['cause'] + \
#         f' \n element:  name {curr_outage["element_name"]}'  )
#         plt.xticks(rotation=90)
#         plt.axvline(t1_orig)
#         plt.axvline(t2_orig)
#         os.makedirs(f'figures/{name}/faults', exist_ok=True)
#         plt.savefig(f'figures/{name}/faults/fault{random_idx}.png')
#         plt.close(fig)


def plot_fault(ami_df, oms_df, sub, feeder, window=300, name='name'):
    oms_df['outage_start'] = pd.to_datetime(oms_df['outage_start'])
    oms_df['outage_end'] = pd.to_datetime(oms_df['outage_end'])
    os.makedirs(f'figures/{name}/faults_agg', exist_ok=True)
    os.makedirs(f'figures/{name}/faults_indv', exist_ok=True)

    if sub:
        c1 = (oms_df['substation'] == sub)
    else:
        c1 = (oms_df['substation'] == '21 Powhatan')
    if feeder:
        c2 = (oms_df['feeder'] == feeder)
    else:
        c2 = (oms_df['feeder'] == '02 Ballsville' )
        
    c3 = ~oms_df['cause'].str.contains('Scheduled')

    # this should probably go in a better spot
    ami_df['start_time'] = pd.to_datetime(ami_df['start_time'])
    ami_df['end_time'] = pd.to_datetime(ami_df['end_time'])
    grp_df = []
    choice_range = np.arange(len(oms_df[c1 & c2 & c3]))


    for random_idx in choice_range:
        t1, t2 = oms_df[c1 & c2 & c3].reset_index()['outage_start'].iloc[random_idx], oms_df[c1 & c2 & c3].reset_index()['outage_end'].iloc[random_idx]
        curr_sub = oms_df[c1 & c2 & c3].reset_index().iloc[random_idx]['substation']
        curr_feeder = oms_df[c1 & c2 & c3].reset_index().iloc[random_idx]['feeder']

        t1_orig, t2_orig = t1, t2
        t1, t2 = t1 - datetime.timedelta(minutes = window), t2 + datetime.timedelta(minutes = window)
        
        if sub:
            c4 = (ami_df['substation'] == curr_sub)
        else:
            c4 = (ami_df['substation'] != 1090230)
        if feeder:
            c5 = (ami_df['feeder'] == curr_feeder)
        else:
            c5 = (ami_df['feeder'] != 1090230)
        
    
        c6 = (ami_df['start_time'] > t1)
        c7 = (ami_df['end_time'] < t2)
        
    #     print(t1,t2,(c4 & c5).sum(), (c6 & c7).sum(), "SUM")
        curr_time_df = ami_df[c4 & c5 & c6 & c7]

        if len(curr_time_df) < 1:
            continue

        grp_df = curr_time_df.groupby(['start_time','end_time']).agg(sum).reset_index()
        curr_outage = oms_df[c1 & c2 & c3].reset_index().loc[random_idx]
        # ensures there is a start and end time in our interval
        if (grp_df['start_time'] < t2_orig).sum() == 0 or (grp_df['end_time'] > t1_orig).sum() == 0:
            continue
        # ensure we exceed interval too
        if (grp_df['start_time'] > t2_orig).sum() == 0 or (grp_df['end_time'] < t1_orig).sum() == 0:
            continue
            
            
        fig = plt.figure(facecolor='white')
        plt.plot(grp_df['start_time'], grp_df['KWH'].values)
        # plt.plot(curr_time_df['start_time'], curr_time_df['KWH'].values)
        plt.title(name + f' Sum of Meters for Feeder 21 Powhatan  \n Cause: ' + curr_outage['cause'] + \
        #plt.title(name + f' Sum of Meters for Feeder {feeder}  \n Cause: ' + curr_outage['cause'] + \
        f' \n element name: {curr_outage["element_name"]}'  )
        plt.xticks(rotation=90)
        plt.axvspan(t1_orig, t2_orig,0,np.max(grp_df['KWH'].values), alpha=0.5, color='red')
        plt.axvline(t1_orig, label='fault start / end', marker='1', color='green', linewidth=2)
        plt.axvline(t2_orig, marker='2',  color='green', linewidth=2)
        plt.legend()
        fig.savefig(f'figures/{name}/faults_agg/fault{random_idx}.png', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        
        fig = plt.figure(facecolor='white')
        plt.plot(curr_time_df['start_time'], curr_time_df['KWH'].values)
        # plt.plot(curr_time_df['start_time'], curr_time_df['KWH'].values)
        plt.title(name + f' Sum of Meters for Feeder 21 Powhatan  \n Cause: ' + curr_outage['cause'] + \
        #plt.title(name + f' Sum of Meters for Feeder {feeder}  \n Cause: ' + curr_outage['cause'] + \
        f' \n element name: {curr_outage["element_name"]}'  )
        plt.axvspan(t1_orig, t2_orig,0,np.max(curr_time_df['KWH'].values), alpha=0.5, color='red')
        plt.xticks(rotation=90)
        plt.axvline(t1_orig, label='fault start / end', marker='1', color='green', linewidth=2)
        plt.axvline(t2_orig, marker='2',  color='green', linewidth=2)
        plt.legend()
        fig.savefig(f'figures/{name}/faults_indv/fault{random_idx}.png', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)


# def plot_timeseries_by_id(table, name, elem_id, id_df, yaxis='y1'):
#     id_df = id_df.sort_values('start_time', ascending=True)
#     id_df['kwh'] = id_df['kwh'].astype(float)
#     energy_data = go.Scatter(x=id_df.start_time,
#                             y=id_df.kwh,
#                             yaxis=yaxis)

#     layout = go.Layout(title=f'Energy Plot for {name} {table} elem id : {elem_id}', xaxis=dict(title='Date'),
#                     yaxis=dict(title='(kWh)'))
#     fig = go.Figure(data=[energy_data], layout=layout)
#     # iplot(fig)
#     os.makedirs(f'figures/{name}/{table}', exist_ok=True)
#     fig.write_html(f'figures/{name}/{table}/elem_{elem_id}.html')

def plot_timeseries_by_id(table, name, elem_id, id_df, oms_df=None, yaxis='y1'):
    id_df = id_df.sort_values('start_time', ascending=True)
    id_df['kwh'] = id_df['kwh'].astype(float)
    roof = id_df['kwh'].max()
    energy_data = go.Scatter(x=id_df.start_time,
                            y=id_df.kwh,
                            yaxis=yaxis)

    layout = go.Layout(title=f'Energy Plot for {name} {table} elem id : {elem_id}', xaxis=dict(title='Date'),
                    yaxis=dict(title='(kWh)'))
    fig = go.Figure(data=[energy_data], layout=layout)
    # iplot(fig)
    start_time = id_df['start_time'].min()
    end_time = id_df['end_time'].max()
    if not oms_df is None:
        overlay_faults(fig, oms_df, start_time, end_time, yaxis)
    if 'fault_present' in id_df.columns:
        fig = shade_faults(fig, id_df, yaxis, roof)
    if 'inferred_fault' in id_df.columns:
        fig = shade_faults(fig, id_df, yaxis, roof, fault_label='inferred_fault', color='yellow')
    os.makedirs(f'figures/{name}/{table}', exist_ok=True)
    fig.write_html(f'figures/{name}/{table}/elem_{elem_id}.html')

def shade_faults(fig, id_df, yaxis, roof, fault_label='fault_present', color='red'):
    xStart = id_df.loc[id_df[fault_label] == True, 'start_time'] - pd.DateOffset(hours=20)# #['2015-01-11', '2015-02-08', '2015-03-08', '2015-04-05']
    # stop times come with two hour offset for viz purposes
    xStop = id_df.loc[id_df[fault_label] == True, 'end_time'] + pd.DateOffset(hours=20)#['2015-01-25', '2015-02-22', '2015-03-22', '2015-04-10']
    xStart, xStop = xStart.values, xStop.values
    # get dict from tuple made by vspan()
    #xElem = fig['layout']['shapes'][0]
    # specify the corners of the rectangles
    seen_intervals = []
    for x_start, x_stop in zip(xStart, xStop):
        # print(x_start, x_stop )
        skip = False
        for interval in seen_intervals:
            if x_start < interval[1] and x_stop > interval[0]: # end of curr interval overlaps:
                skip = True
            if  x_stop > interval[0] and x_start < interval[1]: # begining of curr interval overlaps:
                skip = True
        if skip:
            # print('skipped')
            continue
        else:
            seen_intervals.append((x_start,x_stop))
        x_start = pd.Timestamp(x_start)
        x_stop = pd.Timestamp(x_stop)
        # fig = go.Figure(go.Scatter(x=[x_start,x_stop,x_stop, x_start], y=[0,0, 20, 20],fill="toself", fillcolor='rgba(2550,0,0,0.3)', yaxis=yaxis))

        #fig.add_trace(go.Scatter(x=[x_start,x_start,x_stop, x_stop], y=[0,0, 20, 20],fill="toself", fillcolor='red', yaxis=yaxis))
        fig.add_shape(
                type="rect",
                xref="x",
                yref=yaxis,
                x0=x_start,
                y0=0,
                x1=x_stop,
                y1=roof,
                fillcolor = color,
                # fillcolor="lightgray",
                opacity=0.3,
                line_width=0,
                layer="below"

            )
    # fig.update_shapes(dict(xref='x', yref='y'))
    return fig

def overlay_faults(fig, oms_df, min_time, max_time, yaxis, top_coord=5):
    seen = []
    for row in oms_df.to_dict(orient="records"):
        start_time, end_time = row['start_time'], row['end_time']
        if start_time in seen:
            continue
        else:
            seen.append(start_time)

        if start_time < min_time or end_time > max_time:
            continue
        # y_val_idx1 = get_closest_time(start_time,id_df)
        # y_val_idx2 = get_closest_time(end_time,id_df)        
        text = f"{str(start_time)} : {row['cause']} \n affected: {row['customers_affected']}"
        
        fig.add_trace(go.Scatter(
            x=[start_time,end_time],
            # y=[id_df.loc[y_val_idx1,'kwh'], id_df.loc[y_val_idx1,'kwh']],
            y=[0,top_coord],
            # mode="lines+markers+text",
            # name="Lines, Markers and Text",
            fillcolor='red',
            text=text,
            textposition="top center",
            showlegend=False,
            yaxis=yaxis
        ))

def plot_multitimeseries(table, name, df):
    plot_list = []
    for idx, elem_id in enumerate(df['element_id'].unique()):
        id_df = df.loc[df['element_id'] == elem_id]
        id_df = id_df.sort_values('start_time', ascending=True)
        energy_data = go.Scatter(x=id_df.start_time,
                                y=id_df.kwh,
                                yaxis=f'y{idx+1}')
        plot_list.append(energy_data)


    layout = go.Layout(height=600, width=800,
                   title='Energy Plot for Multiple IDs',
                   # Same x and first y
                   xaxis=dict(title='Date'),
                   yaxis=dict(title='(kWh)', color='red'),
                   # Add a second yaxis to the right of the plot
                   yaxis2=dict(title='(kWh)_2', color='blue',
                               overlaying='y', side='right')
                   )
    fig = go.Figure(data=plot_list, layout=layout)
    # iplot(fig)
    os.makedirs(f'figures/{name}/{table}', exist_ok=True)
    fig.write_html(f'figures/{name}/{table}/multi_elem.html')


def get_closest_time(t1,df):
    idx = np.argmin(np.abs(df.start_time.values - t1))
    return idx


def plot_timeseries_error_annotations(table, name, elem_id, id_df, oms_df, yaxis='y1'):
    id_df = id_df.sort_values('start_time', ascending=True)
    min_time, max_time = id_df['start_time'].min(), id_df['end_time'].max()
    energy_data = go.Scatter(x=id_df.start_time,
                            y=id_df.kwh,
                            yaxis=yaxis)

    layout = go.Layout(title=f'Energy Plot for {name} {table} elem id : {elem_id}', xaxis=dict(title='Date'),
                    yaxis=dict(title='(kWh)'))
    annotations = []
    fig = go.Figure(data=[energy_data], layout=layout)

    top_coord = id_df['kwh'].max()
    # TODO: later make this take in the fig so it can generically add OMS annotations
    # TODO: later make a function that makes annotations for you
    for row in oms_df.to_dict(orient="records"):
        start_time, end_time = row['start_time'], row['end_time']
        if start_time < min_time or end_time > max_time:
            continue
        y_val_idx1 = get_closest_time(start_time,id_df)
        y_val_idx2 = get_closest_time(end_time,id_df)        
        text = f"{str(start_time)} : {row['cause']} \n affected: {row['customers_affected']}"
        
        fig.add_trace(go.Scatter(
            x=[start_time,end_time],
            # y=[id_df.loc[y_val_idx1,'kwh'], id_df.loc[y_val_idx1,'kwh']],
            y=[top_coord,top_coord],
            # mode="lines+markers+text",
            # name="Lines, Markers and Text",
            fillcolor='red',
            text=text,
            textposition="top center",
            showlegend=False,
            yaxis=yaxis
        ))
        # annotations.append(end_dict)
    
    # iplot(fig)
    os.makedirs(f'figures/{name}/{table}', exist_ok=True)
    fig.write_html(f'figures/{name}/{table}/elem_{elem_id}_w_faults.html')


def plot_timeseries_error_annotations_pd(table, name, elem_id, id_df, yaxis='y1'):
    id_df = id_df.sort_values('start_time', ascending=True)
    id_df['kwh'] = id_df['kwh'].astype(float)
    min_time, max_time = id_df['start_time'].min(), id_df['end_time'].max()
    energy_data = go.Scatter(x=id_df.start_time,
                            y=id_df.kwh,
                            yaxis=yaxis)

    layout = go.Layout(title=f'Energy Plot for {name} {table} elem id : {elem_id}', xaxis=dict(title='Date'),
                    yaxis=dict(title='(kWh)'))
    annotations = []
    fig = go.Figure(data=[energy_data], layout=layout)

    top_coord = id_df['kwh'].max()
    # TODO: later make this take in the fig so it can generically add OMS annotations
    # TODO: later make a function that makes annotations for you
    seen = []
    for row in id_df.to_dict(orient="records"):
        start_time, end_time = row['start_time_oms'], row['end_time_oms']
        if start_time in seen:
            continue
        else:
            seen.append(start_time)

        if start_time < min_time or end_time > max_time:
            continue
        # y_val_idx1 = get_closest_time(start_time,id_df)
        # y_val_idx2 = get_closest_time(end_time,id_df)        
        text = f"{str(start_time)} : {row['cause_oms']} \n affected: {row['customers_affected_oms']}"
        
        fig.add_trace(go.Scatter(
            x=[start_time,end_time],
            # y=[id_df.loc[y_val_idx1,'kwh'], id_df.loc[y_val_idx1,'kwh']],
            y=[0,top_coord],
            # mode="lines+markers+text",
            # name="Lines, Markers and Text",
            fillcolor='red',
            text=text,
            textposition="top center",
            showlegend=False,
            yaxis=yaxis
        ))
        # annotations.append(end_dict)
    
    # iplot(fig)
    os.makedirs(f'figures/{name}/{table}', exist_ok=True)
    print(f'saved faults for {elem_id}')
    fig.write_html(f'figures/{name}/{table}/elem_{elem_id}_w_faults_from_sql.html')


def plot_timeseries_error_annotations_pd_joined(table, name, elem_id, id_df, yaxis='y1'):
    id_df = id_df.drop_duplicates(subset=['start_time_ami','kwh'])
    id_df = id_df.sort_values('start_time_ami', ascending=True)
    id_df['kwh'] = id_df['kwh'].astype(float)
    id_df = id_df[id_df['kwh'] != 0 ]
    # import pdb; pdb.set_trace()
    # np.unique(id_df['start_time_ami'].value_counts().values)

    min_time, max_time = id_df['start_time_ami'].min(), id_df['end_time_ami'].max()
    energy_data = go.Scatter(x=id_df.start_time,
                            y=id_df.kwh,
                            yaxis=yaxis)

    layout = go.Layout(title=f'Energy Plot for {name} {table} elem id : {elem_id}', xaxis=dict(title='Date'),
                    yaxis=dict(title='(kWh)'))
    annotations = []
    fig = go.Figure(data=[energy_data], layout=layout)

    top_coord = id_df['kwh'].max()
    # TODO: later make this take in the fig so it can generically add OMS annotations
    # TODO: later make a function that makes annotations for you
    seen = []
    for row in id_df.to_dict(orient="records"):
        start_time, end_time = row['start_time'], row['end_time']
        if start_time in seen:
            continue
        else:
            seen.append(start_time)

        if start_time < min_time or end_time > max_time:
            continue
        # y_val_idx1 = get_closest_time(start_time,id_df)
        # y_val_idx2 = get_closest_time(end_time,id_df)        
        text = f"{str(start_time)} : {row['cause_oms']} \n affected: {row['customers_affected_oms']}"
        
        fig.add_trace(go.Scatter(
            x=[start_time,end_time],
            # y=[id_df.loc[y_val_idx1,'kwh'], id_df.loc[y_val_idx1,'kwh']],
            y=[0,top_coord],
            # mode="lines+markers+text",
            # name="Lines, Markers and Text",
            fillcolor='red',
            text=text,
            textposition="top center",
            showlegend=False,
            yaxis=yaxis
        ))
        # annotations.append(end_dict)
    
    # iplot(fig)
    os.makedirs(f'figures/{name}/{table}', exist_ok=True)
    print(f'saved faults for {elem_id}')
    fig.write_html(f'figures/{name}/{table}/elem_{elem_id}_w_faults_from_sql.html')



def feeder_sub_dict_to_sankey_data(feeder_sub_dict):
    feeders = []
    substations = []
    meters = []
    lookup_dict = {}
    lookup_index = 0
    feeder_list = []
    for sub in feeder_sub_dict.keys():
        if str(sub) + " substation" not in lookup_dict:
            lookup_dict[str(sub) + " substation"] = lookup_index
            lookup_index += 1
        for feeder in feeder_sub_dict[sub].keys():
            # print('feeder: ', feeder)
            if str(feeder) + f'_{sub} feeder' not in lookup_dict:
                lookup_dict[str(feeder) + f'_{sub} feeder'] = lookup_index
                lookup_index += 1
            substations.append(str(sub) + " substation")
            feeder_list.append(str(feeder) + f'_{sub} feeder')
            for meter in feeder_sub_dict[sub][feeder]:
                if type(meter) == list and len(meter) == 1:
                    meter = meter[0]
                meter = str(meter).replace(" ", '')
                meters.append(meter)
                feeders.append(str(feeder) + f'_{sub} feeder')
                if meter not in lookup_dict:
                    lookup_dict[meter] = lookup_index
                    lookup_index += 1
    # labels should be the flat list of keys for lookup dict
    labels = list(lookup_dict.keys())#np.concatenate([feeder_list,substations,meters])
    target =  np.concatenate([substations,feeders])
    source = np.concatenate([feeder_list,meters])
    source_labels = [lookup_dict[src] for src in source]
    target_labels = [lookup_dict[str(trg)] for trg in target]
    return labels, source_labels, target_labels

def plot_sankey_diagram(feeder_sub_dict):
    labels, source_labels, target_labels = feeder_sub_dict_to_sankey_data(feeder_sub_dict)
    # override gray link colors with 'source' colors
    cmap = cm.get_cmap('seismic')
    rgba = cmap(0.5)
    colors, link_colors = [], []
    for c_val in np.arange(len(labels)) / len(labels):
        color = cmap(c_val)
        colors.append('rgb' + str(copy.deepcopy(color)))
        color = list(color)
        color[3] = .2
        link_colors.append('rgb' + str(tuple(color)) )


    fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = labels,
    #   color = "blue"
       color = colors
    ),
        link = dict(
        source = source_labels, # indices correspond to labels, eg A1, A2, A1, B1, ...
        target = target_labels,
        value = np.repeat(1, len(target_labels)),
        # color = link_colors,

    ))])

    fig.update_layout(title_text="Meter to Substation Sankey Diagram", font_size=25, font_family="Arial",
    font_color="black")
    fig.write_html('figures/sankey.html')
    # fig.show()
    

def plot_temporal_sankey_diagram(feeder_sub_dict, ami_fs_dict, time_intervals=10): # TODO: rather specify like a day or so
    os.makedirs('figures/sankey_folder', exist_ok=True)
    feeder =  list(ami_fs_dict.keys())[0]
    substation = list(ami_fs_dict[feeder].keys())[0]
    ami_df_ts = ami_fs_dict[feeder][substation][0]
    start_times = ami_df_ts['start_time'].values
    end_times = ami_df_ts['end_time'].values
    i = 0
    for start_time,end_time in zip(start_times, end_times):
        faulty_feeder_sub_dict = {}
        feeder_sub_dict_clone = copy.deepcopy(feeder_sub_dict)
        for feeder in ami_fs_dict.keys():
            for sub in ami_fs_dict[feeder].keys():
                for meter_df, curr_ami_id in zip(ami_fs_dict[feeder][sub], feeder_sub_dict[feeder][sub]):
                    curr_ami_df = meter_df.loc[(meter_df['start_time'] >= start_time) & (meter_df['start_time'] <= start_time)]
                    if (curr_ami_df['feeder_fault_present'] == True).sum() > 0:
                        faulty_feeder_sub_dict = utils.add_feeder_sub(faulty_feeder_sub_dict, feeder, substation, curr_ami_id)
                        feeder_sub_dict[feeder][sub].remove(curr_ami_id)
                    

   
        healthy_labels, healthy_source_labels, healthy_target_labels = feeder_sub_dict_to_sankey_data(feeder_sub_dict)
        faulty_labels, faulty_source_labels, faulty_target_labels = feeder_sub_dict_to_sankey_data(faulty_feeder_sub_dict)
        healthy_max_idx =  max(max(healthy_target_labels,healthy_source_labels))
        faulty_source_labels = np.array(faulty_source_labels) + healthy_max_idx + 1
        faulty_target_labels = np.array(faulty_target_labels) + healthy_max_idx + 1
        if len(faulty_labels) > 1:  
            print(f'fault {i}')
        else:
            continue

        labels = np.concatenate([healthy_labels, faulty_labels])
        link_colors = np.concatenate([np.repeat('black',len(healthy_source_labels)),  np.repeat('red',len(faulty_source_labels))])
        node_colors = np.concatenate([np.repeat('blue', len(healthy_labels)), np.repeat('red', len(faulty_labels))])
        source_labels = np.concatenate([healthy_source_labels, faulty_source_labels])
        target_labels = np.concatenate([healthy_target_labels, faulty_target_labels])
        # override gray link colors with 'source' colors
        fig = go.Figure(data=[go.Sankey(
        node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label = labels,
        color = node_colors#"blue"
        ),
            link = dict(
            source = source_labels, # indices correspond to labels, eg A1, A2, A1, B1, ...
            target = target_labels,
            value = np.repeat(1, len(target_labels)),
            color = link_colors
            
        ))])
        
        fig.update_layout(title_text="Meter to Substation Sankey Diagram", font_size=10)
        fig.write_html(f'figures/sankey_folder/{i}.html')
        i += 1

    # fig.show()

def plot_oms_outages(oms_df, save_path):
    df0 =  oms_df
    df_duration = df0[['element_id', 'start_time', 'end_time', 'cause_code','customers_affected', 'cause']]
    df_duration.loc[:,'customers_affected'] = df_duration.loc[:,'customers_affected'].astype(int)
    indices = {code:i for i, code in enumerate(df0['cause_code'].unique())}
    colors = cm.rainbow(np.linspace(0, 1, len(indices)))
    fig, ax = plt.subplots(1, 1, figsize=(20, 10)) # the 12 here controls figsize in what we prob. want to be a modular way -- maybe based on # of ticks
    for row in df_duration.iterrows():
        cause_code = row[1]['cause_code']
        cause_desc = row[1]['cause']
        label = str(cause_code) + " - " + cause_desc
        x = pd.date_range(row[1]['start_time'], row[1]['end_time'], freq='s')
        y = np.ones((x.size, )) * indices[cause_code] 
        ax.scatter(x, y, label=label, color=colors[indices[cause_code]], s=(row[1]['customers_affected']* 2) + 2)
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1.0), ncol=1)
    # ax.set_yticks([])
    # yticks = ax.get_yticks()
    # ax.set_yticks(np.arange(min(yticks), max(yticks)+1, .2))

    ax.set_xlabel('Event Duration')
    fig.savefig(save_path)
    plt.close(fig)


def plot_result(df, curr_data, output_path, legend=True, ax=None):
    res, site_data = curr_data['predicted'], curr_data['ground_truth']
    method = curr_data['method_name']
    # hack to get first dims
    site_data = site_data[:len(res),0]
    if len(res.shape) > 1:
        res = res[:,0]
    
    fig = plt.figure()
    plt.plot(site_data,label='ground truth', linewidth=2, color='black')
    plt.plot(res, label='predicted', color='red')
    plt.ylabel('KW', fontsize=14)
    plt.xlabel('Seconds (s)',fontsize=14)
    if legend:
        plt.legend(fontsize=14)
    if ax: # for compound plot of all methods
        # ax.plot(res, label=method, color=config.method_colors[method], marker=config.method_markers[method], markevery=5)
        ax.plot(res, label=method, color=config.method_colors[method], linestyle=config.method_styles[method], alpha=.8)

    plt.savefig(os.path.join(output_path, f'{method}_fit.png'))
    plt.title(method)
    plt.close()
    return fig


def plot_table(df, output_path, name='result',ci=False):
    # prepare df (could be a fxn)
    agg_df = df.groupby('method_name').agg([np.mean, np.std, 'count']).reset_index()
    agg_df = utils.drop_level_combine(agg_df)
    pm_cols = [col.split('_')[0] for col in agg_df.columns if 'mean' in col]
    for col in pm_cols:
        main = col + "_mean"
        agg_df = utils.plus_minus_cols(agg_df,  col + "_mean",  col + "_std", drop=True, ci=ci)
    agg_df.columns = [col.split('_')[0] for col in agg_df.columns]
    
    # plot df
    fig  = plt.figure()
    ax = fig.gca()
    fig.patch.set_visible(False)
    ax.axis('off')
    table = ax.table(cellText=agg_df.values, colLabels=agg_df.columns, loc='center')
    fig.savefig(os.path.join(os.path.dirname(output_path), f'{name}_table.png'), facecolor='white', bbox_inches='tight', dpi=600)
    plt.close(fig)

    # dt to tex
    col_fmt = "|".join(np.repeat('c', len(agg_df.columns)))
    col_fmt = "|" + col_fmt + "|"
    agg_df.to_latex(os.path.join(os.path.dirname(output_path), f'{name}_table.tex'), float_format="%.0f", index=False, column_format=col_fmt)


def methods_plot_result(ax, curr_data):
    method = curr_data['method_name']
    ax.plot(curr_data['predicted'][:,0], label=method,  color=config.method_colors[method], linestyle=config.method_styles[method], linewidth=2, zorder=-10)
    ax.plot(curr_data['ground_truth'][:,0], label='ground_truth', color='black', linewidth=2, zorder=-10)
        # We change the fontsize of minor ticks label 
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=30)
    ax.set_title(method,fontsize=30)

def indv_loss_plot(ax, curr_data, prefix='train'):
    if not len(curr_data[f'{prefix}_loss']):
        ax.set_visible(False)
        return curr_data[f'{prefix}_loss']

    ax.plot(curr_data[f'{prefix}_loss'], color='black', linewidth=2, zorder=-10)
    # We change the fontsize of minor ticks label 
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.set_title(curr_data['method_name'], fontsize=18)
    return curr_data[f'{prefix}_loss']

def final_loss_plot(df, output_path, name='result'):
    num_methods = len(df['method_name'].unique())
    loss_fig, loss_axes = plt.subplots(ncols=num_methods//2, nrows=(num_methods//2)+num_methods%2  , figsize=(5*num_methods,3*num_methods))
    ax_idx = 0
    for method in df['method_name'].unique():
        curr_ax = loss_axes.flatten()[ax_idx]
        curr_df = df[df['method_name'] ==  method]
        train_loss_arr = np.array([row['train_loss'] for row in curr_df.to_dict(orient="records")])
        if len(train_loss_arr.shape) < 2 or not train_loss_arr.shape[1]:
            curr_ax.set_visible(False)
            ax_idx += 1
            continue
        mean = np.mean(train_loss_arr, axis=0)
        sd =  np.std(train_loss_arr, axis=0)
        curr_ax.plot(np.arange(len(mean)), mean, color=config.method_colors[method], linestyle=config.method_styles[method])
        curr_ax.fill_between(np.arange(len(mean)), mean-sd, mean+sd, alpha=.2, color=config.method_colors[method])
        curr_ax.set_title(method)
        ax_idx += 1

    loss_fig.savefig(os.path.join(os.path.dirname(output_path), f'{name}_loss.png'), bbox_inches='tight')
    plt.close(loss_fig)


# visualize
def visualize_output(path):
    task_path = os.path.join(path, 'task.pkl')
    with open(task_path,'rb') as f:
        task =  pickle.load(f)
    if task['name'] == 'autoregression':
        visualize_output_autoreg(path, task)
    elif task['name'] == 'impute':
        visualize_output_interp(path, task)

def visualize_output_autoreg(path, task):
    
    runs = len(utils.listdir_only(path)) # could iterate through a set of runs and make a PDF of all the stuff
    aux_dfs = []
    for run_num in range(runs):
        base_path = os.path.join(path,str(run_num))
        df_path = os.path.join(base_path,'results.csv')
        exp_name = base_path.split('/')[-2]
        df = pd.read_csv(df_path)
        num_methods = len([elem for elem in os.listdir(base_path) if os.path.isdir(os.path.join(base_path,elem))])
        methods_fig, methods_axs = plt.subplots(nrows=num_methods, figsize=(9*num_methods,4*num_methods))
        loss_fig, loss_axes = plt.subplots(ncols=num_methods//2, nrows=(num_methods//2)+num_methods%2 , figsize=(5*num_methods,3*num_methods))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.8)
        figs, rows = [], []
        method_ix = 0
        for method in os.listdir(base_path):
            row = {}
            indv_fig = plt.figure(figsize=(14,5))
            indv_axs = indv_fig.gca()
            if not os.path.isdir(os.path.join(base_path,method)): continue
            method_dir = os.path.join(path,str(run_num),method)
            try:
                with open(os.path.join(method_dir, f'{method}.pkl'), 'rb') as f:
                    curr_data = pickle.load(f) 
            except:
                print(f'no data found for {method}, at run {run_num}, skipping ')
                continue
            
            fig = plot_result(df, curr_data, method_dir, ax=indv_axs)
            figs.append(indv_fig)
            plt.close(indv_fig)

            methods_ax = methods_axs.flatten()[method_ix]
            methods_plot_result(methods_ax, curr_data)

            # TODO: modularize this
            loss_ax = loss_axes.flatten()[method_ix]
            curr_loss = indv_loss_plot(loss_ax, curr_data)
            row["train_loss"] = curr_loss
            row['method_name'] = method
            rows.append(row)
            method_ix += 1

        # track auxillary data recorded post hoc for each run
        aux_dfs.append(pd.DataFrame(rows) )
        methods_fig.savefig(os.path.join(base_path, f'all_methods_{exp_name}.png'), bbox_inches='tight')
        loss_fig.savefig(os.path.join(base_path, f'{exp_name}_train_loss.png'), bbox_inches='tight')
        plt.close(loss_fig)
        plt.close(methods_fig)
    
    aux_df = pd.concat(aux_dfs).reset_index(drop=True)
    plot_table(df, base_path, name=task['name'])
    final_loss_plot(aux_df, base_path, name=task['name'])



    # TODO: low priority could save all figs to one PDF using figs.append()


def visualize_output_interp(path, task):
    runs = len(utils.listdir_only(path)) # 
    for run_num in range(runs):
        base_path = os.path.join(path,str(run_num))
        df_path = os.path.join(base_path,'results.csv')
        exp_name = base_path.split('/')[-2]
        df = pd.read_csv(df_path)
        methods_fig = plt.figure(figsize=(8,5))
        methods_ax = methods_fig.gca()
        figs = []
        for method in os.listdir(base_path):
            if not os.path.isdir(os.path.join(base_path,method)): continue
            method_dir = os.path.join(path,str(run_num),method)
            try:
                with open(os.path.join(method_dir, f'{method}.pkl'), 'rb') as f:
                    curr_data = pickle.load(f) 
            except:
                print(f'no data found for {method}, at run {run_num}, skipping ')
                continue
            fig = plot_result(df, curr_data, method_dir, ax=methods_ax)
            figs.append(fig)

            # methods_plot_result(methods_ax, curr_data)
        
    # methods_plot_result(methods_ax, curr_data)
    methods_ax.plot(curr_data['ground_truth'][:,0], label='ground_truth', color='black', linewidth=2, zorder=-10)
    methods_ax.legend()
    
    methods_ax.set_ylabel('KW')
    methods_ax.set_xlabel('Time (S)')
    methods_fig.savefig(os.path.join(base_path, f'all_methods_{exp_name}.png'))
    # TODO: could save all figs to one PDF
    plot_table(df, base_path, name=task['name'], ci=True)
