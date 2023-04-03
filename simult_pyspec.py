"""
File: simult_pyspec.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: Code to do a simultaneous fit to both xsm daxss.
"""

import glob
import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import xspec as xp

import instruments
from models import chisoth_2T_multi
from plot_fits import plot_individual

plt.style.use("fivethirtyeight")

best_flares = [7, 23, 33, 39, 42, 50, 64, 66, 85, 88, 89, 93, 97, 98, 99]
good_flares = [2, 4, 6, 9, 32, 67, 69, 78, 83, 84, 86, 95]
meh_flares = [10, 11, 28, 37, 41, 45, 46, 63, 79, 91, 96]

flares = best_flares + good_flares + meh_flares

# instruments = ['xsm', 'daxss', 'simult']
xsm = instruments.xsm()
daxss = instruments.daxss()
instruments_list = [xsm, daxss]
instrument_names = ["xsm", "daxss"]
# %% Loading flare table
observation_file = "./flare_filtering/data/flare_observation.h5"
observation_table = pd.read_hdf(observation_file, key="obs")


#%% Prepare arguments for fitting and creating files
FIP_elements = ["Mg", "Si"]
model_args = {"FIP_elements": FIP_elements}

xsm_min_E = 1.3
daxss_min_E = 1.0

min_E = [daxss_min_E, xsm_min_E]
max_E = [10.0] * 2

xsm_min_count = 10.0
daxss_min_count = 10.0

xsm_cutoff_cps = 1.0
daxss_cutoff_cps = 10.0

cutoff_cps = [daxss_cutoff_cps, xsm_cutoff_cps]
do_dynamic_elements = True
bin_size = "27S"
min_count = 10  # Minimum count for grppha
labels = ["DAXSS", "XSM"]

# %% Instrument specifics
# xsm Stuff
xsm_folder = "./data/xsm/data/"
data_dir = "./data/pha_multi"
xsm.load_data(xsm_folder)

DAXSS_file = "./data/daxss_solarSXR_level1_2022-02-14-mission_v2.0.0.ncdf"
daxss_arf_path = "./data/minxss_fm3_ARF.fits"
daxss_rmf_path = "./data/minxss_fm3_RMF.fits"
daxss.load_data(DAXSS_file, daxss_rmf_path, daxss_arf_path)

# %% Funtion to Create pha files
def create_pha_files(flare_num):
    flare_dir = f"{data_dir}/flare_num_{flare_num}/simult"
    pha_dir = f"{flare_dir}/orig_pha"
    fit_file = f"{flare_dir}/fit/results.csv"
    daxss.set_output_dir(flare_dir)
    xsm.set_output_dir(flare_dir)
    flare = observation_table.loc[flare_num]
    time_beg = flare["event_starttime"]
    time_end = flare["event_endtime"]
    daxss.create_pha_files(time_beg, time_end, bin_size=bin_size)
    daxss.do_grouping(daxss_min_count)
    data_array = daxss.flare_data
    daxss_data_selected = data_array.isel(energy=slice(6, 1006))
    cps = daxss_data_selected["cps"]
    resampled = cps.resample(time=bin_size, origin="start", closed="left", label="left")
    timebin_beg_list = []
    timebin_end_list = []
    ls_resamp = list(resampled)
    for i, resampled_i in enumerate(ls_resamp[:-1]):
        timebin_beg = pd.Timestamp(ls_resamp[i][0])
        timebin_end = pd.Timestamp(ls_resamp[i + 1][0])
        # timebin_beg = resampled_i[1].time.isel(time=0).data
        # timebin_end = resampled_i[1].time.isel(time=-1).data
        timebin_beg_list.append(timebin_beg)
        timebin_end_list.append(timebin_end)
        xsm.do_xsmgenspec(timebin_beg, timebin_end)
    timebin_beg = timebin_end
    timebin_end = timebin_beg + pd.Timedelta(bin_size)
    timebin_beg_list.append(timebin_beg)
    timebin_end_list.append(timebin_end)
    xsm.do_xsmgenspec(timebin_beg, timebin_end)
    xsm_PHA_Files = glob.glob(
        f"{pha_dir}/XSM_*.pha"
    )  # Had to do this way because command can fail some times due to no GTI
    xsm_PHA_Files.sort()
    xsm_arf_files = [pha_i.replace(".pha", ".arf") for pha_i in xsm_PHA_Files]
    xsm.set_pha_files(xsm_PHA_Files, xsm_arf_files)
    xsm.do_grouping(xsm_min_count)
    daxss_PHA_files = daxss.PHA_file_list
    xsm_PHA_Files = xsm.PHA_file_list
    #%% Create a list of files
    PHA_files_list = []
    arf_files_list = []
    for daxss_file_i in daxss_PHA_files:
        xsm_temp = daxss_file_i.replace("DAXSS", "XSM")
        if xsm_temp in xsm_PHA_Files:
            xsm_arf = xsm_temp.replace("/grpd_pha/", "/orig_pha/")
            xsm_arf = xsm_arf.replace(".pha", ".arf")
            PHA_files_list.append([daxss_file_i, xsm_temp])
            arf_files_list.append(["USE_DEFAULT", xsm_arf])
    return PHA_files_list, arf_files_list


#%% Do fit
for flare_num in flares:
    flare_dir = f"{data_dir}/flare_num_{flare_num}/simult"
    pha_dir = f"{flare_dir}/orig_pha"
    fit_file = f"{flare_dir}/fit/results.csv"
    #Only do the fit if the file does not exists
    if not os.path.isfile(fit_file):
        PHA_files_list, arf_files_list = create_pha_files(flare_num)
        chiso = chisoth_2T_multi(
            PHA_files_list,
            arf_files_list,
            output_dir=flare_dir,
            FIP_elements=FIP_elements,
        )
        chiso.fit(
            min_E=min_E,
            max_E=max_E,
            cutoff_cps=cutoff_cps,
            do_dynamic_elements=True,
            labels=labels,
        )
    #%% Plot
    plot_individual("simult", data_dir, flare_num, PLOT_LIGHTCURVE=False)
    plot_individual("simult", data_dir, flare_num,out_dir=f"{flare_dir}/figures_masked", PLOT_LIGHTCURVE=False,MASK=True)
