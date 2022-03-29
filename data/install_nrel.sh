#!/bin/bash


mkdir -p nrel_smart_ds/AUS
# aws s3 sync --no-sign-request s3://oedi-data-lake/SMART-DS/v1.0/2018/AUS  nrel_smart_ds/AUS
mkdir -p nrel_smart_ds/AUS/GIS
aws s3 sync --no-sign-request s3://oedi-data-lake/SMART-DS/v1.0/GIS  nrel_smart_ds/AUS/GIS/

aws s3 sync --no-sign-request s3://oedi-data-lake/SMART-DS/v1.0/2018/AUS/full_dataset_analysis/base_timeseries  nrel_smart_ds/AUS/analysis
