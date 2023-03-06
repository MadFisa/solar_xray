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
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xspec as xp

from data_utils import create_daxss_pha, read_daxss_data
from fit_utils import chisoth_2T, do_grppha

plt.style.use("fivethirtyeight")
flare_num = 7
flare_dir = f"./data/pha/flare_num_{flare_num}/xsm"
observation_file = "./flare_filtering/data/flare_observation.h5"

observation_table = pd.read_hdf(observation_file, "obs")

xsm_origin_time = datetime(2017, 1, 1)

utc2met = lambda time: (time - xsm_origin_time).total_seconds()


#%% Chosse flare
def create_pha_files(flare_num, bin_size):
    flare = observation_table.loc[flare_num]
    time_beg = flare["event_starttime"]
    time_end = flare["event_endtime"]
    # xsm_lc = flare["xsm_lc"]
    # daxss_lc = flare["daxss_lc"]
    times_list = pd.date_range(time_beg, time_end, freq=bin_size)
    times_met = utc2met(times_list)
    #%% Create PHA files
    out_dir = f"{flare_dir}/orig_pha/"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for i, time_i in enumerate(times_list[:-1]):
        met_i_beg = times_met[i]
        met_i_end = times_met[i + 1]
        year = time_i.year
        month = time_i.month
        day = time_i.day

        root_dir = f"./data/xsm/data/{year}/{str(month).zfill(2)}/{str(day).zfill(2)}"
        file_basename = f'ch2_xsm_{time_i.strftime("%Y%m%d")}_v1_level'
        l1file = f"{root_dir}/raw/{file_basename}1.fits"
        hkfile = f"{root_dir}/raw/{file_basename}1.hk"
        safile = f"{root_dir}/raw/{file_basename}1.sa"
        gtifile = f"{root_dir}/calibrated/{file_basename}2.gti"
        outfile = (
            f"{flare_dir}/orig_pha/XSM_{np.datetime_as_string(time_i.to_numpy())}.pha"
        )

        command = f"xsmgenspec l1file={l1file} specfile={outfile} spectype='time-integrated' hkfile={hkfile} safile={safile} gtifile={gtifile} tstart={met_i_beg} tstop={met_i_end} "
        os.system(command)


#%% Create files
CREATE = False
BIN = False
# Create initial PHA files
if not os.path.isdir(f"{flare_dir}/orig_pha/"):
    CREATE = True
if CREATE:
    shutil.rmtree(f"{flare_dir}/orig_pha/", ignore_errors=True)
    resamp = create_pha_files(flare_num, bin_size="27S")

# Run grppha on files to group
orig_PHA_file_list = glob.glob(f"{flare_dir}/orig_pha/*.pha")
orig_PHA_file_list.sort()
PHA_file_list = [
    orig_i.replace("/orig_pha/", "/grpd_pha/") for orig_i in orig_PHA_file_list
]

cutoff_cps = 1
if not os.path.isdir(f"{flare_dir}/grpd_pha/"):
    BIN = True
if BIN:
    shutil.rmtree(f"{flare_dir}/grpd_pha/", ignore_errors=True)
    os.makedirs(f"{flare_dir}/grpd_pha")
    do_grppha(orig_PHA_file_list, PHA_file_list, cutoff_cps)
    os.system(f"cp {flare_dir}/orig_pha/*.arf {flare_dir}/grpd_pha/")

# #%% Initialise

chiso = chisoth_2T(PHA_file_list, flare_dir)
# FIP_elements = ["Mg", "Si", "S", "Ar", "Ca", "Fe"]
FIP_elements = ["Mg", "Si", "S", "Ar", "Fe"]
chiso.init_chisoth(FIP_elements)

#%%Fit
chiso.arf_files_list = [
    orig_i.removesuffix(".pha") + (".arf") for orig_i in orig_PHA_file_list
]

df = chiso.fit(min_E=1.3)
