#!/usr/bin/env python3
"""
File: DAXSS_pyspec.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: Code to analyse DAXSS flares.
"""


import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xspec as xp
import glob
from plot_fits import plot_individual, plot_simult
import instruments
from models import chisoth_2T

best_flares = [7, 23, 33, 39, 42, 50, 64, 66, 85, 88, 89, 93, 97, 98, 99]
good_flares = [2, 4, 6, 9, 32, 67, 69, 78, 83, 84, 86, 95]
meh_flares = [10, 11, 28, 37, 41, 45, 46, 63, 79, 91, 96]

flares = best_flares + good_flares + meh_flares

# instruments = ['xsm', 'daxss', 'simult']
xsm = instruments.xsm()
daxss = instruments.daxss()
instruments_list = [xsm,daxss]
instrument_names = ["xsm","daxss"]
#%% Loading flare table
observation_file = "./flare_filtering/data/flare_observation.h5"
observation_table = pd.read_hdf(observation_file, "obs")

CREATE = True
BIN = True
min_count = 10 # Minimum count for grppha
#%% xsm Stuff
xsm_folder = "./data/xsm/data/"
bin_size = "27S"

#%% For fitting
FIP_elements = ["Mg", "Si"]
min_E = 1.3
max_E = 10.0
bin_size = "27S"
xsm_cutoff_cps = 1.
daxss_cutoff_cps = 10.

do_dynamic_elements = True
xsm_fit_args = {"min_E": min_E, "max_E": max_E, "do_dynamic_elements": do_dynamic_elements,"cutoff_cps":xsm_cutoff_cps}
daxss_fit_args = {"min_E": min_E, "max_E": max_E, "do_dynamic_elements": do_dynamic_elements,"cutoff_cps":daxss_cutoff_cps}
model_args = {"FIP_elements": FIP_elements}
#%% Instrument specifics
# xsm Stuff
xsm_folder = "./data/xsm/data/"
xsm.load_data(xsm_folder)

DAXSS_file = "./data/daxss_solarSXR_level1_2022-02-14-mission_v2.0.0.ncdf"
daxss_arf_path = "./data/minxss_fm3_ARF.fits"
daxss_rmf_path = "./data/minxss_fm3_RMF.fits"
daxss.load_data(DAXSS_file, daxss_rmf_path, daxss_arf_path)

for flare_num in flares:
    for instrument_i in instruments_list:
        flare_dir = f"./data/pha_class/flare_num_{flare_num}/{instrument_i.name}"
        fit_file = f"{flare_dir}/fit/results.csv"
        instrument_i.set_output_dir(flare_dir)
        flare = observation_table.loc[flare_num]
        time_beg = flare["event_starttime"]
        time_end = flare["event_endtime"]
        #%% Create files if it does not exist
        if not os.path.isfile(fit_file):
            if not os.path.isdir(f"{flare_dir}/orig_pha/"):
                CREATE = True
            # else:
                # PHA_file_list = glob.glob(f"{flare_dir}/orig_pha/*.pha")
                # instrument_i.set_pha_files(PHA_file_list, ["USE_DEFAULT"] * len(PHA_file_list))
            if CREATE:
                shutil.rmtree(f"{flare_dir}/orig_pha/", ignore_errors=True)
                instrument_i.create_pha_files(time_beg, time_end, bin_size=bin_size)

            if not os.path.isdir(f"{flare_dir}/grpd_pha/"):
                BIN = True
            if BIN:
                shutil.rmtree(f"{flare_dir}/grpd_pha/", ignore_errors=True)
                instrument_i.do_grouping(min_count)
            else:
                instrument_i.PHA_file_list = glob.glob(f"{flare_dir}/grpd_pha/*.pha")
            #%% Fit
            if instrument_i.name == "daxss":
                instrument_i.set_pha_files(instrument_i.PHA_file_list, ["USE_DEFAULT"] * len(instrument_i.PHA_file_list))
                instrument_i.fit(chisoth_2T, model_args, daxss_fit_args)
                plot_individual(instrument_i.name,flare_dir,)
            if instrument_i.name == "xsm":
                instrument_i.set_pha_files(instrument_i.PHA_file_list, instrument_i.arf_files)
                instrument_i.fit(chisoth_2T, model_args, xsm_fit_args)
                plot_individual(instrument_i.name,flare_dir,)
            # Create model arguments
    plot_simult("./data/pha_class",flare_num,instrument_names)

