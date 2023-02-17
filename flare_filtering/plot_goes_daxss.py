#!/usr/bin/env python3
"""
File: plot_goes_daxss.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: Code to overplot goes data on top of DAXSS data.
"""

# %% Read the data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from data_utils import read_daxss_data

DAXSS_file = "./data/daxss_solarSXR_level1_2022-02-14-mission_v2.0.0.ncdf"
daxss_data = read_daxss_data(DAXSS_file)

# These are just for visualisation, not exact calculations
flux_kev = daxss_data["irradiance"] * daxss_data.energy  # in kev/s
flux_W = flux_kev * 1.6e-16
daxss_flux = flux_W.sum(dim="energy")
sunpy_path = "/home/sac/sunpy/data"
# flare_table = pd.read_csv("./data/goes_flares_info.csv",
# parse_dates=['event_starttime', 'event_peaktime', 'event_endtime'])

goes16_data = xr.open_mfdataset(f"{sunpy_path}/*g16*.nc")
goes17_data = xr.open_mfdataset(f"{sunpy_path}/*g17*.nc")
#%% Read GOES data
fig, ax = plt.subplots(1, 1)
flare_table = pd.read_hdf(
    "./data/goes_flares_info.h5",
)
for row in flare_table:
    row = flare_table.loc[0]
    daxss_t_beg = row["daxss_interval_beg"]
    daxss_t_end = row["daxss_interval_end"]
    goes16_file_list = row["goes16_file"]
    goes17_file_list = row["goes17_file"]
    # goes16_file = goes16_file_list[0]

    #%% Plot the data
    time_interval = slice(daxss_t_beg, daxss_t_end)
    daxss_sliced = daxss_flux.sel(time=time_interval)
    daxss_sliced.plot.line("b--o", label="DAXSS", ax=ax)
    goes16_xrsa_sliced = goes16_data["xrsa_flux"].sel(time=time_interval)
    goes16_xrsb_sliced = goes16_data["xrsb_flux"].sel(time=time_interval)
    goes16_xrsa_sliced.plot.line("r--o", label="16 A", ax=ax)
    goes16_xrsb_sliced.plot.line("g--o", label="16 B", ax=ax)

    goes17_xrsa_sliced = goes17_data["xrsa_flux"].sel(time=time_interval)
    goes17_xrsb_sliced = goes17_data["xrsb_flux"].sel(time=time_interval)
    goes17_xrsa_sliced.plot.line("y--o", label="17 A", ax=ax)
    goes17_xrsb_sliced.plot.line("k--o", label="17 B", ax=ax)

plt.legend()
