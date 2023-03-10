#!/usr/bin/env python3
"""
File: DAXSS_pyspec.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: Code to analyse XSM flares.
"""

import glob
import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_utils import chisoth_2T, do_grppha
from plot_fits import plot_individual

plt.style.use("fivethirtyeight")

xsm_origin_time = datetime(2017, 1, 1)
observation_file = "./flare_filtering/data/flare_observation.h5"

utc2met = lambda time: (time - xsm_origin_time).total_seconds()


def create_pha_files(flare_num, bin_size):
    flare_dir = f"./data/pha/flare_num_{flare_num}/xsm"
    observation_table = pd.read_hdf(observation_file, "obs")
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
        arffile = (
            f"{flare_dir}/orig_pha/XSM_{np.datetime_as_string(time_i.to_numpy())}.arf"
        )

        command = f"xsmgenspec l1file={l1file} specfile={outfile} spectype='time-integrated' hkfile={hkfile} safile={safile} gtifile={gtifile} arffile={arffile} tstart={met_i_beg} tstop={met_i_end} "
        os.system(command)


#%% Create files
def fit_xsm(
    flare_num,
    FIP_elements=["Mg", "Si", "S", "Ar", "Ca", "Fe"],
    CREATE=False,
    BIN=False,
    cutoff_cps=1,
    min_E=1.3,
    max_E=10.0,
):
    """

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
    flare_dir = f"./data/pha/flare_num_{flare_num}/xsm"
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

    if not os.path.isdir(f"{flare_dir}/grpd_pha/"):
        BIN = True
    if BIN:
        shutil.rmtree(f"{flare_dir}/grpd_pha/", ignore_errors=True)
        os.makedirs(f"{flare_dir}/grpd_pha")
        do_grppha(orig_PHA_file_list, PHA_file_list, cutoff_cps)
        os.system(f"cp {flare_dir}/orig_pha/*.arf {flare_dir}/grpd_pha/")

    # #%% Initialise

    arf_file_list = [
        orig_i.removesuffix(".pha") + (".arf") for orig_i in orig_PHA_file_list
    ]
    chiso = chisoth_2T(PHA_file_list, arf_file_list, flare_dir)
    chiso.init_chisoth(FIP_elements, error_sigma=12.00)

    #%%Fit
    df = chiso.fit(min_E, max_E)
    return df
