import argparse
import os
import sys
import csv
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_values
from pgcopy import CopyManager
import time
import pandas as pd
import numpy as np

ignore = ['Price','VEE_Change_Method','AMI_Vendor_Codes', 'element_id', 'Limit']

def parse_csv(csv_reader):
    csv_data = []
    line_count = 0
    for row in csv_reader:
        row = dict(row)
        csv_data.append(row)
        line_count += 1
    return csv_data


def get_rows(csv_path):
    with open(csv_path, mode='r', encoding='latin1') as csv_file:
        rows = parse_csv(csv.DictReader(csv_file, delimiter=','))
    return rows

# can add more dtypes?
sql_panda_type_converter = {
    'timestamp': pd.to_datetime, # datetime converter
    'datetime' : pd.to_datetime,
    'varchar' : str,
    'float' : float,
    'double': np.double,
    }

def descriptor_info(line):
    split_line = line.split(' ')
    split_line = [elem for elem in split_line if elem != '']
    name = split_line[0]
    dtype = None
    for field in split_line[:]:
        for key in sql_panda_type_converter.keys():
            if key in field:
                dtype = sql_panda_type_converter[key]
    if not dtype:
        assert False, f"FAILED TO INFER DTYPE FROM {line}"
    return (name, dtype)
    
def pull_spec(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    recording =  False
    names_and_types = []
    for line in lines:
        if recording and len(set(line)) > 3 and not '--' in line : # ensures line is not just '\n'
            name, dtype = descriptor_info(line)
            names_and_types.append([name, dtype])
        if "CREATE TABLE" in line: recording = True # start recording data descriptors
        if recording and ';' in line: break # ok we are done recording descriptors
    return names_and_types

def augment_spec(path, spec):
    header_df = pd.read_csv(path, nrows=1000)
    for col in header_df.columns:
        tp = header_df[col].dtype.type
        str_found = len([elem for elem in header_df[col].values if type(elem) == str or not np.isfinite(elem) ]) > 1
        if str_found: tp = str
        col = col.replace(" ", '_')
        if 'Unnamed' in col or col in ignore:
            continue
        seen = len([elem[0] for elem in spec if col.lower() in elem[0].lower()]) > 0
        if not seen:
            spec.append([col, tp])
    return spec

def reformat_headers(row):
    res = {}
    for key in row.keys():
        res[key.replace(' ', '_')] = row[key]
    return res

def spec_match_row(row, spec):
    res = []
    row = reformat_headers(row)
    # [res.append(dtype(row[name])) for name,dtype in spec]
    for name,dtype in spec:
        res.append(dtype(row[name])) 

    return res 
    

def insert_db(conn, curs, csv_path, field_permissive=True):
    """ Write rows of data to a timescale db table."""
    split_p = csv_path.split('/')
    # name table using the path of csv we read from (syemc_ami)
    table = split_p[-2] + '_'   + split_p[-3] 
    table = table.lower() # no caps in sql
    # pull spec from sql script
    spec =  pull_spec(os.path.join('sql',split_p[-3] + '.sql'))
    csv_data = get_rows(csv_path)
    if field_permissive:
        spec = augment_spec(csv_path, spec)
    col_str = ', '.join([elem[0] for elem in spec])
    sql_string = f"INSERT INTO {table} ({col_str}) VALUES %s"
    data = []
    for csv_row in csv_data:
        data.append(spec_match_row(csv_row,spec))

    cols = [elem[0].lower() for elem in spec] # no UPPER cols in sql?
    mgr = CopyManager(conn, table, cols)
    start = time.time()
    mgr.copy(data)
    end = time.time()
    print(f'ingesting {len(csv_data)} rows took :', end - start)

    conn.commit()

