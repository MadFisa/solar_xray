#!/usr/bin/env python3
"""
File: find_intervals_daxss.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: A script to extract all the time intervals in DAXSS data.
"""

# %% Read the data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_utils import read_daxss_data

DAXSS_file = "./data/daxss_solarSXR_level1_2022-02-14-mission_v2.0.0.ncdf"
daxss_data = read_daxss_data(DAXSS_file)
t_unit = pd.Timedelta(minutes=1)  # Time unit we will be working on
# %% Visualise
# df_time = daxss_data.time.to_dataframe()
# df_dt = df_time.diff()
# df_dt.rename(columns={'time': 'delta_t'}, inplace=True)
# # Converting to seconds because matplotlub cant deal with timedeltas for some reason
# # Lets convert it to seconds
# # t_unit = pd.Timedelta(seconds=10)
# delta_ts = df_dt['delta_t'] / t_unit
# n_bins = 15
# logbins = np.geomspace(delta_ts.min(), delta_ts.max(), n_bins)
# hist_range = (pd.Timedelta(seconds=20)/t_unit, delta_ts.max())
# no_counts, bins, _ = plt.hist(
# delta_ts, bins=logbins, rwidth=0.8, range=hist_range)
# plt.ylim(top=50)
# %% Lets figure out the intervals

net_counts = daxss_data['cps'].sum(dim='energy')
dt = net_counts.time.diff(dim='time')/t_unit
# Maximum gap (in t_unit) between two nearby datapoints  to be considered as a break
max_gap = 5
begin_point_mask = (dt > max_gap).data
begin_point_mask = np.insert(begin_point_mask, 0, True)
begin_points = net_counts.time[begin_point_mask]

end_point_mask = np.roll(begin_point_mask, -1)
end_points = net_counts.time[end_point_mask]
# %% Visualise
net_counts.plot.line("--o", label='data')
net_counts[begin_point_mask].plot.line("rx", label='start_points')
net_counts[end_point_mask].plot.line("kx", label='end_points')
# y_hline = net_counts[begin_point_mask]
y_hline = net_counts.max()
y_hline = y_hline.data.tolist()
plt.hlines([y_hline]*len(begin_points), begin_points.data,
           end_points.data, color='g', label='intervals', lw=15)
plt.legend()

# %% write it out

time_intervals = np.vstack((begin_points, end_points)).T
time_csv_file = "DAXSS_time_intervals.csv"
np.savetxt(time_csv_file, time_intervals, delimiter=',', fmt='%s', header='start_points,end_points')
