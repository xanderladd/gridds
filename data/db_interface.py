import psycopg2
import pandas.io.sql as sqlio
import pandas as pd
import gridds.tools.utils as utils
import os 
from gridds.sql.ingest import insert_db
import numpy as np

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

pd_to_sql = {

    np.object_ :'varchar(32)' ,
    np.object :'varchar(32)',
    np.int64 : 'float(32)',
    np.float64 : 'float(32)'

    }

ignore = ['Price','VEE_Change_Method','AMI_Vendor_Codes', 'Limit']


class DbObject:

    def __init__(self):
        """."""

    def connect(self):
        CONNECTION =  private.CONNECTION 
        self.conn = psycopg2.connect(CONNECTION)
        self.cur = self.conn.cursor()
        print(f"Connecting to DB:  {bcolors.OKGREEN} Success  {bcolors.ENDC}")
        return 0

    def add_auxilliary_cols(self,command,path):
        files =  [os.path.join(path,elem) for elem in os.listdir(path) if os.path.isfile(os.path.join(path,elem))]
        command = command[:-2]
        for fl in files:
            df = pd.read_csv(fl, nrows=4)
            for col in df.columns:
                sql_type =  pd_to_sql[df[col].dtype.type]
                col = col.replace(" ", '_')
                if 'Unnamed' in col or col.lower() in command.lower() or col in ignore:
                    continue
                command += f',    {col}  {sql_type}\n'
        command += ');'
        return command


    def create(self,data_type,path,prefix,replace=True, field_permissive=True):
        tbl_name = prefix + '_' + data_type
        if replace:
            #Dropping EMPLOYEE table if already exists.
            self.cur.execute(f"DROP TABLE IF EXISTS {tbl_name}")
        fd = open(os.path.join('sql',data_type+'.sql'), 'r')
        sql_create = fd.read()
        fd.close()
        # all SQL commands (split on ';')
        sqlCommands = sql_create.split(';')
        retcode = 0
        try:
            for command in sqlCommands:
                if len(set(command)) < 5: # command needs at least 5 unique characters
                    continue
                command = command.replace('table_name',prefix + '_' + data_type)
                if field_permissive:
                    command = self.add_auxilliary_cols(command, path)
                self.cur.execute(command)
            self.conn.commit()
        except Exception as e:
            print(e, "exception while creating table")
            retcode = 1
        return retcode
            
    def ingest(self,data_type,path,prefix,replace=True,field_permissive=True):
        for root, dirs, files in os.walk(path):
            for f in files:
                if not f.endswith('.csv'):
                    continue
                infile_path = os.path.join(root, f)
                #try:
                insert_db(self.conn, self.cur, infile_path, field_permissive)
                #except Exception as e:
                 #   print(f"Caught exception reading: {infile_path}\n{e}\n")
    


    # convienence function to deal with OOM memory issues with hypertables
    # def drop_chunks(table):
    #     # TODO:
    #         # base case to check \d+ output
    #     cmd = f"SELECT drop_chunks('{table}','2020-08-01');"
    #     self.cur.execute(cmd)
    #     output = self.cur.fetchall()
    #     if output:
    #         return output
    #     else:
    #         drop_chunks(table, date="NEW DATE") # TODO: add new date here to incrementally delete chunks



    def construct_query(self, sql_command_dict):
        """
         Format custom SQL query like:
        sql_command_dict = {
        'SELECT': '*',
        'FROM': 'ami',
        'WHERE': "element_id='81592199'",#"start_time > '2019'",
        'ORDER BY': 'end_time',
        'INNER JOIN': '',
        'GROUP BY': ''}
        """
        res_string = ""
        for key in sql_command_dict:
            if len(sql_command_dict[key]) > 0:
                res_string += f" {key} {sql_command_dict[key]}"
        res_string = res_string #+ " ;"
        return res_string

    def check_table(self, table_name):
        self.cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE"\
        + f" table_name = '{table_name}'")
        if self.cur.fetchone()[0] == 1:
            return True

        return False

    def execute_query(self, query):
        self.cur.execute(query)
        return self.cur.fetchall()


    def query_to_pandas(self,query):
        # print(query)
        # dat, dat2 = sqlio.read_sql(query, self.conn)
        self.cur.execute(query)
        rows = self.cur.fetchall()
        columns = [column[0] for column in self.cur.description]
        # print(columns, "COLS")
        res = pd.DataFrame.from_records(rows, columns=columns)
        return res

    def check_col(self, table, colname):
        query = f"SELECT '{colname}'  FROM information_schema.columns WHERE table_name='{table}' and column_name='{colname}';"
        self.cur.execute(query)
        if self.cur.fetchone(): # found column 
            return True 
        else: # did not find
            return False

    
    def update_col(self, table, col, value, where_clause=''):
        query = f"UPDATE {table} SET {col} = {value}" + where_clause + ";"
        self.cur.execute(query)
        self.conn.commit()


    def add_col(self, table, colname, dtype, default_val):
        """
        TODO: add retcodes
        """
        if self.check_col(table, colname):
            print(f"Column : '{colname}' exists")
            return 0

        query = f"ALTER TABLE {table} ADD COLUMN {colname} {dtype}" \
                + f" DEFAULT ('{default_val}');"
        self.cur.execute(query)
        self.conn.commit()
        return 0

    def drop_col(self, table, colname):
        """
        TODO: check success
        """
        if not self.check_col(table, colname):
            print(f"Column : '{colname}' does not exist!")
            return 0
        query =  f"ALTER TABLE {table} DROP COLUMN {colname};"
        self.cur.execute(query)
        self.conn.commit()
        return 1
        
    def query_by_id(self, table, id, start_time=None, id_col='element_id', target="*", distinct=True):
        query_string = f"SELECT {target} FROM {table} WHERE {id_col}='{id}';"
        if distinct:
            query_string = f"SELECT DISTINCT({target}) FROM {table} WHERE {id_col}='{id}';"
        dat = sqlio.read_sql_query(query_string, self.conn)
        return dat

    def query_by_time(self, table, start_time=None, end_time=None):
        if start_time and end_time:
            query_string = f"SELECT * FROM {table} WHERE " + \
            f"start_time > '{start_time}'  AND end_time < '{end_time}'"
        elif start_time and not end_time:
            query_string = f"SELECT * FROM {table} WHERE " + \
            f"start_time > '{start_time}' "
        elif end_time and not start_time:
            query_string = f"SELECT * FROM {table} WHERE " + \
            f"end_time > '{start_time}' "
        dat = sqlio.read_sql_query(query_string, self.conn)
        return dat

    def query_multiple_ids(self, table, ids, start_time=None):
        assert len(ids) > 0
        query_string = f"SELECT * FROM {table} WHERE "
        for idx,elem_id in enumerate(ids):
            if idx == len(ids) - 1:
                query_string += f"element_id='{elem_id}'"
            else:
                query_string += f"element_id='{elem_id}' OR "
        dat = sqlio.read_sql_query(query_string, self.conn)
        return dat

    def query_table(self, table):
        query_string = f"SELECT * FROM {table}"
        dat = sqlio.read_sql_query(query_string, self.conn)
        return dat

    def query_by_feeder(self, table, feeder, sub=None, start_time=None, end_time=None):
        query_string = "SELECT * FROM {table} WHERE feeder='{meter_id}' and sub={sub}"
        dat = sqlio.read_sql_query(query_string, self.conn)
        return dat
        
        
    def get_substations(self):
        query = "SELECT DISTINCT(substation) from ami_oms_fault_times;"
        self.cur.execute(query)
        return utils.flatten_list(self.cur.fetchall())

    def get_feeders(self):
        query = "SELECT DISTINCT(feeder) from ami_oms_fault_times;"
        self.cur.execute(query)
        return utils.flatten_list(self.cur.fetchall())
    
    def get_subs_and_feeders(self):
        query = "SELECT DISTINCT(substation, feeder, co_op) from ami;"
        self.cur.execute(query)
        subs_and_feeders =  utils.flatten_list(self.cur.fetchall())
        subs_and_feeders = utils.format_sub_feeder_tuples(subs_and_feeders)
        return subs_and_feeders
        

    def list_tables(self):
        self.cur.execute("""SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'""")
        for table in self.cur.fetchall():
            print(table)

    def list_column_names(self, db):
        self.cur.execute(f"SELECT * FROM {db} LIMIT 0")
        colnames = [desc[0] for desc in self.cur.description]
        return colnames


    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.conn.close()
        return 0
            

