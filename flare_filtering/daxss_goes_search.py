#!/usr/bin/env python3
"""
File: daxss_goes_search.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: find goes light curve daxss time intervals.
"""
from sunpy.net import Fido
from sunpy.net import attrs as a
import numpy as np
import pandas as pd
from tqdm import tqdm

daxss_time_csv_file = "./data/DAXSS_time_intervals.csv"
daxss_time_intervals = np.loadtxt(
    daxss_time_csv_file, delimiter=',', dtype='datetime64[ns]')
out_dir = "./data/daxss_flares"

# %% Search
goes16_file_names = []
goes17_file_names = []

for times_i in tqdm(daxss_time_intervals):
    times_query = a.Time(times_i[0], times_i[1])
    results = Fido.search(times_query,
         a.Instrument("XRS"), a.goes.SatelliteNumber(16))
    goes16_result = results['xrs']
    files_goes16 = Fido.fetch(goes16_result)
    results = Fido.search(times_query,
         a.Instrument("XRS"), a.goes.SatelliteNumber(17))
    goes17_result = results['xrs']
    files_goes17 = Fido.fetch(goes17_result)
    goes16_file_names.append(files_goes16.data)
    goes17_file_names.append(files_goes17.data)

#%% Make a panda dataframe to store
df_dict = {'daxss_interval_beg' : daxss_time_intervals[:,0],
           'daxss_interval_end' : daxss_time_intervals[:,1],
           'goes16_file' : goes16_file_names,
           'goes17_file' : goes17_file_names,
           }
df = pd.DataFrame(df_dict)
df.to_hdf("./data/goes_flares_info.h5", key='df')
df.to_csv("./data/goes_flares_info.csv", index=False)
