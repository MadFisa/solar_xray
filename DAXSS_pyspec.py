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

plt.style.use("fivethirtyeight")
flare_num = 7
flare_dir = f"./data/pha/flare_num_{flare_num}/daxss"
DAXSS_file = "./data/daxss_solarSXR_level1_2022-02-14-mission_v2.0.0.ncdf"
observation_file = "./flare_filtering/data/flare_observation.h5"


def create_pha_files(flare_num, bin_size="27S"):
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


def do_grouping(
    file_list,
    out_put_file_list,
    cutoff_cps,
):
    """
    The function will do gpb pha on files based on cutoff cps.

    Parameters
    ----------
    file_list : list of files
    cutoff_cps : cutoff cps
    out_put_file_list : list of output file names.

    Returns
    -------
    TODO

    """
    for in_file, out_file in zip(file_list, out_put_file_list):
        command = f"grppha infile='{in_file}' outfile='!{out_file}' comm='GROUP MIN {cutoff_cps}&exit' "
        os.system(command)


#%% Create files
CREATE = True
# Create initial PHA files
if CREATE:
    shutil.rmtree(f"{flare_dir}/orig_pha/", ignore_errors=True)
    resamp = create_pha_files(flare_num)

# Run grppha on files to group
orig_PHA_file_list = glob.glob(f"{flare_dir}/orig_pha/*.pha")
PHA_file_list = [
    orig_i.replace("/orig_pha/", "/grpd_pha/") for orig_i in orig_PHA_file_list
]

BIN = True
cutoff_cps = 1
if BIN:
    shutil.rmtree(f"{flare_dir}/grpd_pha/", ignore_errors=True)
    os.makedirs(f"{flare_dir}/grpd_pha")
    do_grouping(orig_PHA_file_list, PHA_file_list, cutoff_cps)
#%% Initialise
xp.AllModels.lmod("chspec", dirPath="/home/sac/chspec/")
xp.AllData.clear()
xp.AllModels.clear()
xp.Fit.query = "no"  # No asking for uesr confirmation while fitting
xp.Plot.xAxis = "keV"
xp.Plot.yLog = True
xp.Plot.xLog = False
xp.Xset.parallel.leven = 6

#%% Setup Model
FIP_elements = ["Fe", "Ar", "Ca", "Si", "S", "Mg"]
other_pars = ["logT", "norm"]
fit_pars = other_pars + FIP_elements
suffix = ["", "_2"]

# Creating column names for pandas later
colum_names = []
for fit_pars_i in fit_pars:
    for suffix_i in suffix:
        colum_names.append(fit_pars_i + suffix_i + "_values")
        colum_names.append(fit_pars_i + suffix_i + "_UB")
        colum_names.append(fit_pars_i + suffix_i + "_LB")
        colum_names.append(fit_pars_i + suffix_i + "_err_code")
colum_names.append("Chi")

m = xp.Model("chisoth + chisoth", "flare")
# Dictionary that will be used to unfreeze parameters
FIP_unfreeze_dict = {}
temperature_unfreeze_dict = {
    eval("m.chisoth.logT.index"): ",0.001,,,,,",
    eval("m.chisoth_2.logT.index"): ",0.001,,,,,",
}


other_par_index = []
model_components_list = m.componentNames
for component_i in model_components_list:
    for other_pars_i in other_pars:
        idx_temp = eval(f"m.{component_i}.{other_pars_i}.index")
        other_par_index.append(idx_temp)

FIP_par_index = []
for FIP_el in FIP_elements:
    idx_temp = eval(f"m.chisoth.{FIP_el}.index")
    par_1 = eval(f"m.chisoth.{FIP_el}")
    par_2 = eval(f"m.chisoth_2.{FIP_el}")
    par_2.link = par_1
    idx_temp = eval(f"m.chisoth.{FIP_el}.index")
    FIP_par_index.append(idx_temp)
    FIP_unfreeze_dict[idx_temp] = ",0.1,,,,,"

all_par_index = other_par_index + FIP_par_index
err_string = "flare:" + "".join(
    [str(i) + "," for i in all_par_index]
)  # error string to be used later with xspec err command


# We will fit with just temperatures unfreezed to reach solution fast then -
# unfreeze rest.
xp.AllData.clear()
s = xp.Spectrum(PHA_file_list[0])
xp.Fit.renorm()
m.setPars(temperature_unfreeze_dict)
xp.Fit.perform()
xp.Fit.renorm()
m.setPars(FIP_unfreeze_dict)

out_dir = f"{flare_dir}/fit"
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
par_vals = []
#%% Fit
for PHA_file in PHA_file_list:
    f_name = os.path.basename(PHA_file).removesuffix(".pha")
    xp.AllData.clear()
    logFile = xp.Xset.openLog(f"{out_dir}/{f_name}.log")
    s = xp.Spectrum(PHA_file)
    xp.Fit.statMethod = "chi"
    s.ignore("**-1.0 10.0-**")
    spectra = np.array(s.values)
    xp.Fit.renorm()
    xp.Fit.perform()
    n = 0
    while xp.Fit.testStatistic > 350 and n < 5:
        n += 1
        xp.Fit.perform()
    # Finding errors
    xp.Fit.error(err_string)
    xp.Xset.save(f"{out_dir}/{f_name}.xcm")
    xp.Xset.closeLog()
    temp_col = []
    for fit_pars_i in fit_pars:
        for suffix_i in suffix:
            m_par_i = eval(f"m.chisoth{suffix_i}.{fit_pars_i}")
            temp_col.append(m_par_i.values[0])
            temp_col.append(m_par_i.error[0])
            temp_col.append(m_par_i.error[1])
            temp_col.append(m_par_i.error[2])
    temp_col.append(xp.Fit.testStatistic)
    par_vals.append(temp_col)
    ##% Plot
    # Stuff required for plotting
    xp.Plot.device = "/xs"
    # xp.Plot("data", "resid")
    xp.Plot("data", "delchi")
    x = xp.Plot.x()
    x_err = xp.Plot.xErr()
    y = xp.Plot.y()
    y_err = xp.Plot.yErr()
    model = xp.Plot.model()
    xp.Plot("delchi")
    chix = xp.Plot.x()
    chix_err = xp.Plot.xErr()
    chi = xp.Plot.y()
    chi_err = xp.Plot.yErr()

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(16, 9))
    ax[0].plot(x, model, drawstyle="steps-mid")
    ax[0].errorbar(x, y, xerr=x_err, yerr=y_err, linestyle="None", fmt="k", alpha=0.6)
    ax[0].set_yscale("log")
    ax[0].set_ylim(bottom=1)
    ax[0].set_ylabel("counts/s/keV")
    ax[1].hlines(
        0,
        min(chix),
        max(chix),
    )
    ax[1].errorbar(
        chix, chi, xerr=chix_err, yerr=chi_err, linestyle="None", fmt="k", alpha=0.6
    )
    ax[1].set_ylabel("(data-model)/error")
    fig.supxlabel("Energy (keV)")
    plt.savefig(f"{out_dir}/{f_name}.png")
    plt.close()
#%% Make a data frame
times = [
    os.path.basename(PHA_file_i).removesuffix(".pha")[6:]
    for PHA_file_i in PHA_file_list
]
times = pd.to_datetime(times)

df = pd.DataFrame(par_vals, columns=colum_names, index=times)
df.to_csv(f"{out_dir}/results.csv")
df.to_hdf(f"{out_dir}/results.h5", "results")
