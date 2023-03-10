#!/usr/bin/env python3
"""
File: xsm_files_list_creator.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: A script to create file names for downloading from pradan download
scirpt
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def generate_xsm_filename(interval_start_array, interval_end_array):
    """
    Creates a list of filenames of xsm files. Made for use with wget script for
    pradhan

    Parameters
    ----------
    interval_start_array : array of initial times
    interval_end_array : array of end times

    Returns
    -------
    list, with filenames in it

    """
    file_list = []
    dates_list = [ pd.date_range(start=st,end=se,freq='D') for st,se in zip(interval_start_array,interval_end_array)]

    url_list = []
    for ev in dates_list:
        for date in ev:
            file_name = f'ch2_xsm_{date.strftime("%Y%m%d")}_v1.zip'
            file_list.append(file_name)
    return file_list


observation_file = "./data/flare_observation.h5"
observations_table = pd.read_hdf(observation_file, "obs")

#%% Generate urls 
urlPrefix="https://pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/xsm_collection"
payload = "xsm"

interval_start_array = []
interval_end_array = []
observations_table["xsm_len"] = observations_table["xsm_lc"].apply(len)
xsm_obs_mask = observations_table["xsm_len"] != 0
interval_start_array = observations_table[xsm_obs_mask]["event_starttime"]
interval_end_array = observations_table[xsm_obs_mask]["event_endtime"]

file_list = generate_xsm_filename(interval_start_array,interval_end_array)

np.savetxt("./data/xsm_files.txt",file_list,fmt="%s")



