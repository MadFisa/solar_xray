#!/usr/bin/env python3
"""
File: goes_flare.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: A sample script to try sifting through Goes flare data.
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
my_df = []

# for times_i in daxss_time_intervals:
for times_i in tqdm(daxss_time_intervals):
    times_query = a.Time(times_i[0], times_i[1])
    event_type = "FL"
    result = Fido.search(times_query,
                         a.hek.EventType(event_type),
                         a.hek.FL.GOESCls > "B1.0",
                         a.hek.OBS.Observatory == 'GOES')

    # %% Show the results
    hek_results = result["hek"]
    if len(hek_results) > 0:
        # new_table = hek_results["event_starttime", "event_peaktime",
        # "event_endtime", "fl_goescls","fl_peakflux","fl_peakfluxunit", "ar_noaanum",'hpc_bbox']
        new_table = hek_results["event_starttime", "event_peaktime",
                                "event_endtime", "fl_goescls", "ar_noaanum", 'hpc_bbox']
        temp_t = pd.to_datetime(times_i[0])
        date_string_beg = temp_t.strftime('%Y_%m_%dT%H:%M:%S')
        temp_t = pd.to_datetime(times_i[1])
        date_string_end = temp_t.strftime('%Y_%m_%dT%H:%M:%S')
        # %% Save results
        out_put_filename = f'{out_dir}/result_{date_string_beg}_to_{date_string_end}'
        new_table.write(f"{out_put_filename}.h5", format="hdf5")
        new_table.write(f"{out_put_filename}.csv", format="csv")
        temp_df = new_table.to_pandas()
        temp_df["daxss_interval_beg"] = times_i[0]
        temp_df["daxss_interval_end"] = times_i[1]
        my_df.append(temp_df)

# %% Combine dataframe

df = pd.concat(my_df)
df.reset_index(inplace=True)
df.drop(columns='index', inplace=True)
df.to_csv(f"{out_dir}/daxss_flares.csv", index=False)
df.to_csv("./data/daxss_flares.csv", index=False)
df.to_hdf(f"{out_dir}/daxss_flares.h5", key='df')
df.to_hdf("./data/daxss_flares.h5", key='df')
