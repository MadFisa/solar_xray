#!/usr/bin/env python3
"""
File: DAXSS_pyspec.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: Code to analyse DAXSS flares.
"""
import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xspec as xp

from data_utils import create_daxss_pha, read_daxss_data
from fit_utils import chisoth_2T, do_grppha

plt.style.use("fivethirtyeight")
flare_num = 7
flare_dir = f"./data/pha/flare_num_{flare_num}/daxss"
DAXSS_file = "./data/daxss_solarSXR_level1_2022-02-14-mission_v2.0.0.ncdf"
observation_file = "./flare_filtering/data/flare_observation.h5"


def create_pha_files(flare_num, bin_size=None):
    """
    creates pha files from data and flare_number
    Parameters
    ----------
    flare_num : int, flare_number(index) in the observation_table.

    Returns
    -------
    TODO

    """
    daxss_data = read_daxss_data(DAXSS_file)
    observation_table = pd.read_hdf(observation_file, "obs")

    #%% Chosse flare
    flare = observation_table.loc[flare_num]
    time_beg = flare["event_starttime"]
    time_end = flare["event_endtime"]
    daxss_flare = daxss_data.sel(time=slice(time_beg, time_end))

    #%% Create PHA files
    arf_path = "./data/minxss_fm3_ARF.fits"
    rmf_path = "./data/minxss_fm3_RMF.fits"
    out_dir = f"{flare_dir}/orig_pha/"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    pp = create_daxss_pha(
        daxss_flare,
        out_dir=out_dir,
        arf_path=arf_path,
        rmf_path=rmf_path,
        bin_size=bin_size,
    )
    return pp


#%% Create files
CREATE = True
BIN = True
# Create initial PHA files
if not os.path.isdir(f"{flare_dir}/orig_pha/"):
    CREATE = True
if CREATE:
    shutil.rmtree(f"{flare_dir}/orig_pha/", ignore_errors=True)
    resamp = create_pha_files(flare_num, bin_size="27S")
if CREATE:
    shutil.rmtree(f"{flare_dir}/orig_pha/", ignore_errors=True)
    resamp = create_pha_files(flare_num,bin_size='27S')

# Run grppha on files to group
orig_PHA_file_list = glob.glob(f"{flare_dir}/orig_pha/*.pha")
orig_PHA_file_list.sort()
PHA_file_list = [
orig_i.replace("/orig_pha/", "/grpd_pha/") for orig_i in orig_PHA_file_list
]

if not os.path.isdir(f"{flare_dir}/grpd_pha/"):
    BIN = True
cutoff_cps = 1
if BIN:
    shutil.rmtree(f"{flare_dir}/grpd_pha/", ignore_errors=True)
    os.makedirs(f"{flare_dir}/grpd_pha")
    do_grppha(orig_PHA_file_list, PHA_file_list, cutoff_cps)

# PHA_file_list = orig_PHA_file_list
# #%% Initialise

chiso = chisoth_2T(PHA_file_list, flare_dir)
# FIP_elements = ["Mg", "Si", "S", "Ar", "Ca", "Fe"]
FIP_elements = ["Mg", "Si", "S", "Ar", "Fe"]
chiso.init_chisoth(FIP_elements)

#%%Fit

df = chiso.fit(min_E=1.0)
