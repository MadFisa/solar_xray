#!/usr/bin/env python3
"""
File: plot_flares.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: Code to overplot flares obatined from HEK on top of DAXSS data.
"""

# %% Read the data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from netCDF4 import Dataset
# import xarray as xr
from data_utils import read_daxss_data

DAXSS_file = "./data/daxss_solarSXR_level1_2022-02-14-mission_v2.0.0.ncdf"
daxss_data = read_daxss_data(DAXSS_file)
net_counts = daxss_data['cps'].sum(dim='energy')
# time_csv_file = "DAXSS_time_intervals.csv"
# DAXSS_time_intervals = np.loadtxt(time_csv_file,  delimiter=',', dtype='np.datetime64[ns]')
# columns = ['Unnamed: 0', 'event_starttime', 'event_peaktime', 'event_endtime',
# 'fl_goescls', 'ar_noaanum', 'hpc_bbox', 'daxss_interval_beg',
# 'daxss_interval_end']
# col_dtypes = (np.int, np.datetime64, np.datetime64,
# np.datetime64, str, np.int, str)
# dtype_dict = {column_i: dtype_i for column_i,
# dtype_i in zip(columns, col_dtypes)}

# flare_table = pd.read_csv("./data/daxss_flares_bak.csv", dtype=dtype_dict)
flare_table = pd.read_csv("./data/daxss_flares_bak.csv",
                          parse_dates=['event_starttime', 'event_peaktime', 'event_endtime'])
flare_peak_time = flare_table['event_peaktime']
flare_class_data = flare_table['fl_goescls']
flare_class = ['C', 'M', 'X']

# %% Plot the data
plt.figure()
net_counts.plot.line("--o", label='data')
for flare_class_i in flare_class:
    class_mask = flare_class_data.str.startswith(flare_class_i)
    flare_interp = net_counts.interp(time=flare_peak_time[class_mask])
    flare_interp.plot.line('x', label=flare_class_i)
plt.legend()
