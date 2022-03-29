import pandera as pa

from pandera import Column, DataFrameSchema, Check, Index
import datetime
import numpy as np

"""
Attributes
Element ID (str)
Feeder (str)
Map Location (str) 
Blinks (int)
Phase (str)
KWH (float)
Date (date)
Time (date)


"""
ami_schema = DataFrameSchema(
    {   'element_id' : Column(str),
        'feeder' : Column(str),
        # 'substation' : Column(str),
        'map_location' : Column(str), 
        'KWH' : Column(pa.Float),
        'start_time' : Column(),
        'end_time' : Column()#Column(pa.DateTime),

      
    },
    index=Index(int),
    strict=False,
    coerce=True,
)

ami_blink_schema = DataFrameSchema(
    {   'element_id' : Column(str),
        # 'feeder' : Column(str),
        'map_location' : Column(str), 
        'blinks' : Column(int),
        # 'phase' : Column(str),
        # 'KWH' : Column(pa.Float),
        'date' : Column(),#Column(pa.DateTime),
        'time' : Column()#Column(pa.DateTime),
      
    },
    index=Index(int),
    strict=False,
    coerce=True,
)

"""
Attributes
Element ID (str)
Element (str)
Outage Start (time)
Outage End (time)
Customer hours (int)
Duration (time)
Customers affected (int)
customers_impact (int) : total minutes of outage * number customers affected
Cause (str)
Cause Code (int)
Map location (str / int)
Substation
Phase
Feeder

Equipment
Equipment code
Weather
Weather code


"""

oms_schema = DataFrameSchema(
    {
        'element_id': Column(str),
        'element_name': Column(str),
        'outage_start': Column(), #Column(pa.DateTime),
        'outage_end': Column(), #Column(pa.DateTime),
        'duration': Column(), #Column(pa.DateTime),
        'customers_affected': Column(int),
        # 'customers_impact': Column(int),
        'cause': Column(str, nullable=True),
        # should be three digit code
        'cause_code': Column(str, Check(lambda s: len(s) <= 3, element_wise=True)),
        'map_location': Column(str),
        'co_op': Column(str),
        'equipment': Column(str),
        'equipment_code': Column(str, Check(lambda s: len(s) <= 3, element_wise=True)),
        # 'weather': Column(str),
        # 'weather_code': Column(str)

       
    },
    index=Index(int),
    strict=False,
    coerce=True,
)

"""
Element ID (str)
Primary key
Date (pa.DateTime)
Point description (str)
Point name (str)
Message (str)
Value (int)
"""
scada_event_schema = DataFrameSchema(
    {   
        'element_id' : Column(str),
        'date' :  Column(pa.DateTime),
        'point_description' : Column(str),
        'point_name': Column(str),
        'message' : Column(str),
        'value' : Column(), # unspecified data type?
        'co_op': Column(str)

    },
    index=Index(int),
    strict=False,
    coerce=True,
)

"""
Attributes
Date (pa.DateTime)
Device type
KW

"""

scada_continuous_schema = DataFrameSchema(
    {
        'time' :  Column(pa.DateTime),
        'point_description': Column(str),
        'substation': Column(str),
        'units': Column(str),
        'channel': Column(str),
        'value' : Column(pa.Float),
        'co_op': Column(str),

    },
    index=Index(int),
    strict=False,
    coerce=True,
)


gis_schema = DataFrameSchema(
    {
        'map_loc': Column(str),
        'lat' : Column(pa.Float),
        'lon' : Column(pa.Float)

    },
    index=Index(int),
    strict=False,
    coerce=True,
)
