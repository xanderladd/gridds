import argparse
import os
import sys
import csv
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_values


def parse_csv(csv_reader):
    csv_data = []
    line_count = 0
    for row in csv_reader:
        row = dict(row)
        csv_data.append(row)
        line_count += 1
    return csv_data


def insert_db(conn, csv_data, table):
    """ Write rows of data to a timescale db table."""
    with conn.cursor() as curs:
        sql_string = f"INSERT INTO {table} (datetime, element_id," + \
                    f"point_description, point_name, message, value) VALUES %s"
        data = []
        count = 0
        for csv_row in csv_data:

            data.append([
                csv_row['date'],
                csv_row['element_id'],
                csv_row['point_description'],
                csv_row['point_name'],
                csv_row['message'],
                csv_row['value']

            ])

         

            count = count + 1
            if count % 100 == 0:
                print(f"inserting row {count}")
                execute_values(curs, sql_string, data)
                data = []

        if len(data) > 0:
            # If number of rows is not exactly divisible by 100 then there will be some left over data that needs to
            # be inserted.
            execute_values(curs, sql_string, data)

        conn.commit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Parse SCADA event csv files in a directory.")
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
                            rows = parse_csv(csv.DictReader(csv_file, delimiter=','))
                            insert_db(conn, rows, 'scada_event')
                        except Exception as e:
                            print(f"Caught exception reading: {infile_path}\n{e}\n")
    finally:
        conn.close()
