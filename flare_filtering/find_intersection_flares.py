#!/usr/bin/env python3
"""
File: find_intersection_flares.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: Code to find flares that is observed by both XSM and DAXSS.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from data_utils import read_daxss_data

DAXSS_file = "./data/daxss_solarSXR_level1_2022-02-14-mission_v2.0.0.ncdf"
xsm_lc_files = "./data/xsm_2022_lc.nc"
sunpy_path = "/home/sac/sunpy/data"
flares_file = "./data/daxss_flares.h5"
daxss_intervals_file = "./data/DAXSS_time_intervals.csv"

# load all the data
goes16_data = xr.open_mfdataset(f"{sunpy_path}/*g16*.nc")
goes17_data = xr.open_mfdataset(f"{sunpy_path}/*g17*.nc")
daxss_data = read_daxss_data(DAXSS_file)
xsm_lc = xr.load_dataset(xsm_lc_files)
flare_table_full = pd.read_hdf(flares_file)

daxss_intervals = pd.read_csv(
    daxss_intervals_file, names=["start_points", "end_point"], parse_dates=[0, 1]
)
daxss_intervals_np = daxss_intervals.to_numpy()

# Calculate the flux for DAXSS
flux_kev = daxss_data["irradiance"] * daxss_data.energy  # in kev/s
flux_W = flux_kev * 1.6e-16
daxss_flux = flux_W.isel(energy=slice(2, -1)).sum(dim="energy")

#%% Look for data in flare intervals


# Take care of duplicates
flare_table = flare_table_full.drop_duplicates(subset=["event_peaktime"])
flare_interval_array = pd.arrays.IntervalArray.from_arrays(
    flare_table["event_starttime"], flare_table["event_endtime"]
)
overlap = np.array([flare_interval_array.overlaps(x) for x in flare_interval_array])

# Find intervals that overlap with others
no_overlap = np.sum(overlap, axis=1)
overlap_mask = np.where(no_overlap > 1, True, False)
overlap_index = np.where(overlap_mask)[0]

# Lets make new non overlapping index with new data
# Lets remove the one with max ovrelap
flare_table = flare_table.drop(15)

# flare_interval_array = pd.arrays.IntervalArray.from_arrays(
# flare_table["event_starttime"], flare_table["event_endtime"]
# )
#%% Mark the data
clasifier = lambda df: [
    df[(event_start < df.time) & (df.time < event_end)]
    for event_start, event_end in zip(
        flare_table["event_starttime"], flare_table["event_endtime"]
    )
]
flare_table["daxss_lc"] = clasifier(daxss_flux)
flare_table["xsm_lc"] = clasifier(xsm_lc["flux (1-8A)"])
flare_table["goes16_lc"] = clasifier(goes16_data["xrsa_flux"])
flare_table["goes17_lc"] = clasifier(goes17_data["xrsa_flux"])


#%% Save the table
observations = flare_table.drop(columns=["daxss_interval_beg", "daxss_interval_end"])
observation_file = "./data/flare_observation.h5"
observations.to_hdf(observation_file, "obs")

# fig, ax = plt.subplots(1, 1)
# for interval in daxss_intervals_np:
# interval_beg = interval[0]
# interval_end = interval[1]
# t_slice = slice(interval_beg, interval_end)
# daxss_flux_slice = daxss_flux.sel(time=t_slice)
# goes16_flux_slice = goes16_data["xrsb_flux"].sel(time=t_slice)
# goes17_flux_slice = goes17_data["xrsb_flux"].sel(time=t_slice)
# try:
# xsm_flux_slice = xsm_lc["flux (1-8A)"].sel(time=t_slice)
# except:
# xsm_flux_slice = None
