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
import pandas as pd
import xspec as xp

import instruments
from models import chisoth_2T
from plot_fits import plot_individual

plt.style.use("fivethirtyeight")

instrument = "daxss"
#%% Loading flare table
observation_file = "./flare_filtering/data/flare_observation.h5"
observation_table = pd.read_hdf(observation_file, "obs")

CREATE = True
BIN = True
min_count = 20  # Minimum count for grppha
#%% DAXSS Stuff
DAXSS_file = "./data/daxss_solarSXR_level1_2022-02-14-mission_v2.0.0.ncdf"
daxss_arf_path = "./data/minxss_fm3_ARF.fits"
daxss_rmf_path = "./data/minxss_fm3_RMF.fits"
bin_size = "27S"

#%% For fitting
FIP_elements = ["Mg", "Si"]
min_E = 1.3
max_E = 10.0
bin_size = "27S"

do_dynamic_elements = True

#%% Load instrument
daxss = instruments.daxss()
daxss.load_data(DAXSS_file, daxss_rmf_path, daxss_arf_path)

#%% Flare specific stuff
flare_num = 7
flare_dir = f"./data/pha_class/flare_num_{flare_num}/{instrument}"
daxss.set_output_dir(flare_dir)
flare = observation_table.loc[flare_num]
time_beg = flare["event_starttime"]
time_end = flare["event_endtime"]
#%% Create files if it does not exist
if not os.path.isdir(f"{flare_dir}/orig_pha/"):
    CREATE = True
else:
    PHA_file_list = glob.glob(f"{flare_dir}/orig_pha/*.pha")
    daxss.set_pha_files(PHA_file_list, ["USE_DEFAULT"] * len(PHA_file_list))
if CREATE:
    shutil.rmtree(f"{flare_dir}/orig_pha/", ignore_errors=True)
    daxss.create_pha_files(time_beg, time_end, bin_size=bin_size)

if not os.path.isdir(f"{flare_dir}/grpd_pha/"):
    BIN = True
if BIN:
    shutil.rmtree(f"{flare_dir}/grpd_pha/", ignore_errors=True)
    daxss.do_grouping(min_count)
else:
    PHA_file_list = glob.glob(f"{flare_dir}/grpd_pha/*.pha")
    daxss.set_pha_files(PHA_file_list, ["USE_DEFAULT"] * len(PHA_file_list))

#%% Fit
# Create model arguments
model_args = {"FIP_elements": FIP_elements}
fit_args = {"min_E": min_E, "max_E": max_E, "do_dynamic_elements": do_dynamic_elements}
daxss.fit(chisoth_2T, model_args, fit_args)
