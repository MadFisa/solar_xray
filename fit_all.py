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

from DAXSS_pyspec import fit_daxss
from plot_fits import plot_individual, plot_simult
from simult_pyspec import fit_simult
from XSM_pyspec import fit_xsm

best_flares = [7, 23, 33, 39, 42, 50, 64, 66, 85, 88, 89, 93, 97, 98, 99]
good_flares = [2, 4, 6, 9, 32, 67, 69, 78, 83, 84, 86, 95]
meh_flares = [10, 11, 28, 37, 41, 45, 46, 63, 79, 91, 96]

flares = best_flares + good_flares + meh_flares

# instruments = ['xsm', 'daxss', 'simult']
instruments = ["xsm", "daxss"]
FIP_elements = ["Mg", "Si"]
for flare_num in flares:
    flare_dir = f"./data/pha/flare_num_{flare_num}"
    daxss_out_dir = f"{flare_dir}/daxss"
    xsm_out_dir = f"{flare_dir}/xsm"
    simult_out_dir = f"{flare_dir}/simult"
    if not os.path.isfile(f"{daxss_out_dir}/fit/results.csv"):
        print(f"----------Fitting for flare {flare_num} for daxss----------")
        shutil.rmtree(f"{daxss_out_dir}/fit", ignore_errors=True)
        df_daxss = fit_daxss(flare_num=flare_num, FIP_elements=FIP_elements)
    plot_individual(instrument="daxss", flare_num=flare_num)
    if not os.path.isfile(f"{xsm_out_dir}/fit/results.csv"):
        print(f"----------Fitting for flare {flare_num} for xsm--------------")
        shutil.rmtree(f"{xsm_out_dir}/fit", ignore_errors=True)
        df_xsm = fit_xsm(
            flare_num=flare_num,
            FIP_elements=FIP_elements,
        )
    plot_individual(instrument="xsm", flare_num=flare_num)
    # if not os.path.isdir(simult_out_dir):
    # print(f"Fitting for flare {flare_num} for daxss")
    # df_simult = fit_simult(
    # flare_num=flare_num, FIP_elements=FIP_elements, max_E=8.0
    # )
    # plot_individual(instrument="simult", flare_num=flare_num)
    plot_simult(flare_num, instruments)
