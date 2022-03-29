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
    cols = ('time', 'substation', 'map_location', 'lat', \
                'lon')
    dtypes = {col: 'str' for col in cols}
    dtypes['lat']: 'float'
    dtypes['lon']: 'float'
    df = pd.read_csv(filepath, dtype=dtypes)
    df['time'] = pd.to_datetime(df['time'])
    return df, cols


def insert_db(conn, csv_data, table, cols):
    """ Write rows of data to a timescale db table."""

    data = []
    count = 0
    for csv_row in  csv_data.to_dict(orient="records"):
        
        data.append([
            csv_row['time'],
            csv_row['substation'],
            csv_row['map_location'],
            float(csv_row['lat']),
            float(csv_row['lon']),
            csv_row['co_op']
        ])

    
    cols = ('datetime','substation', 'map_location', 'lat', \
                'lon', 'co_op')
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
                            insert_db(conn, df, 'gis_grafana', cols)
                        except Exception as e:
                            print(f"Caught exception reading: {infile_path}\n{e}\n")
    finally:
        conn.close()
