import argparse
import os
import sys
import csv
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_values
from pgcopy import CopyManager
import pandas as pd
import time

def parse_csv(csv_reader):
    csv_data = []
    line_count = 0
    for row in csv_reader:
        row = dict(row)
        csv_data.append(row)
        line_count += 1
    return csv_data

def parse_df(filepath):
    dates = ['start_time', 'end_time']
    cols = ('start_time', 'end_time', 'element_id', \
                'map_location', 'feeder', 'KWH', 'co_op')
    dtypes = {col: 'str' for col in cols}
    dtypes['KWH']: 'float'
    df = pd.read_csv(filepath, dtype=dtypes, parse_dates=dates)
    return df, cols


def insert_db(conn, csv_data, table, cols):
    """ Write rows of data to a timescale db table."""

    data = []
    count = 0
    for csv_row in  csv_data.to_dict(orient="records"):
        
        data.append([
            csv_row['start_time'],
            csv_row['end_time'],
            csv_row['element_id'],
            csv_row['map_location'],
            str(int(csv_row['feeder'])),
            str(int(csv_row['substation'])),
            float(csv_row['KWH']),
            csv_row['co_op']
        ])

    
    cols = ('start_time', 'end_time', 'element_id', \
                'map_location', 'feeder', 'substation', 'kwh', 'co_op')
    mgr = CopyManager(conn, table, cols)
    start = time.time()
    mgr.copy(data)
    end = time.time()
    print(f'ingesting {len(csv_data)} rows took :', end - start)

    conn.commit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Parse AMI files in a directory.")
    parser.add_argument("root_data_path", help="root directory containing data files")

    args = parser.parse_args()

    conn = psycopg2.connect(host=private.host,
                        database=private.database,
                        user=private.user,
                        password=private.password)
    try:
        for root, dirs, files in os.walk(args.root_data_path):
            for f in files:
                if f.endswith('.csv'):
                    infile_path = os.path.join(root, f)

                    with open(infile_path, mode='r', encoding='latin1') as csv_file:
                        try:
                            df, cols = parse_df(infile_path)
                            import pdb; pdb.set_trace()
                            insert_db(conn, df, 'ami_grafana', cols)
                        except Exception as e:
                            print(f"Caught exception reading: {infile_path}\n{e}\n")
    finally:
        conn.close()
