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
from plot_fits import plot_individual

plt.style.use("fivethirtyeight")

observation_file = "./flare_filtering/data/flare_observation.h5"
DAXSS_file = "./data/daxss_solarSXR_level1_2022-02-14-mission_v2.0.0.ncdf"
daxss_data = read_daxss_data(DAXSS_file)


#%% Create files
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
    observation_table = pd.read_hdf(observation_file, "obs")
    flare_dir = f"./data/pha/flare_num_{flare_num}/xsm"
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


def fit_daxss(
    flare_num,
    FIP_elements=["Mg", "Si", "S", "Ar", "Ca", "Fe"],
    CREATE=False,
    BIN=False,
    cutoff_cps=1,
    min_E=1.3,
    max_E=10.0,
):
    """
    function to daxss fitting

    Parameters
    ----------
    flare_num : int, index of flare in the list.
    FIP_elements : list of elements to be considered as FIP.
    CREATE : bool, Whether to create_pha_files.
    BIN : bool, Whetther to bin the pha files using grbpha
    cutoff_cps: float, threshold cps to bin i.e the grppha
    min_E : float, minimum energy of spectra to start fit from
    max_E : float, optional, maximum energy to fit from

    Returns
    -------
    Datframe with results

    """
    # Create initial PHA files
    flare_dir = f"./data/pha/flare_num_{flare_num}/daxss"
    if not os.path.isdir(f"{flare_dir}/orig_pha/"):
        CREATE = True
    if CREATE:
        shutil.rmtree(f"{flare_dir}/orig_pha/", ignore_errors=True)
        resamp = create_pha_files(flare_num, bin_size="27S")
    if CREATE:
        shutil.rmtree(f"{flare_dir}/orig_pha/", ignore_errors=True)
        resamp = create_pha_files(flare_num, bin_size="27S")

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
    chiso.init_chisoth(FIP_elements, error_sigma=4.00)

    #%%Fit

    df = chiso.fit(min_E, max_E)
    return df


flare_num = 42
FIP_elements = ["Mg", "Si", "S", "Ar", "Ca", "Fe"]
df = fit_daxss(flare_num=flare_num, FIP_elements=FIP_elements)

#%% plot
plot_individual(instrument="daxss", flare_num=flare_num)
