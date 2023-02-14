"""
File: xspec_fit_data_plotting.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: Plotting for fitted data.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

from data_utils import create_daxss_pha, read_daxss_data

plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.style.use("fivethirtyeight")
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["font.size"] = 24
date_formatter = DateFormatter("%H:%M")
keV_to_MK = 11604562.9141 / 1e6
dates_init = [
    "2022-03-15T23:20:00",
    "2022-03-31T18:00:00",
    "2022-05-05T13:32:00",
    "2022-03-22T20:20:00",
    "2022-05-03T08:00:00",
    "2022-03-22T11:00:00",
    "2022-03-07T22:40:00",
]
dates_stop = [
    "2022-03-15T23:30:00",
    "2022-03-31T23:00:00",
    "2022-05-05T13:50:00",
    "2022-03-22T20:30:00",
    "2022-05-03T08:38:00",
    "2022-03-22T11:13:00",
    "2022-03-07T22:55:00",
]

spec_num = 1
date_init = dates_init[1]
date_stop = dates_stop[1]
DAXSS_file = "/home/sac/Asif/Chianti_codes/main/daxss_solarSXR_level1_2022-02-14-mission_v2.0.0.ncdf"
daxss_data = read_daxss_data(DAXSS_file)
time_sel = slice(date_init, date_stop)

daxss_data = read_daxss_data(DAXSS_file)
flare_data = daxss_data.sel(time=slice(date_init, date_stop))

#%% Create PHA files
PHA_dir = f"./PHA_Files/prl_model/flare_{date_init}"
arf_path = "/home/sac/Asif/Chianti_codes/main/minxss_fm3_ARF.fits"
rmf_path = "/home/sac/Asif/Chianti_codes/main/minxss_fm3_RMF.fits"
preflare_dir = f"{PHA_dir}/preflare"


##% setup
# PHA_file = "minxss_fm3_PHA_2022-03-15T23-20-41Z.pha"
PHA_file_list = [
    f"{PHA_dir}/DAXSS_{np.datetime_as_string(time)}.pha"
    for time in flare_data.time.data
]
#%% read csv

out_dir = f"{PHA_dir}/2T"
fig_dir = f"{out_dir}/figures"
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)
results_file = f"{out_dir}/results.csv"

df = pd.read_csv(results_file, index_col=0, parse_dates=True)

net_counts = flare_data["cps"].sum(dim="energy")
times = df.index
#%% Plot Individual
pars = df.columns[0:-1:4]
plt.figure(figsize=(16, 9))
for par_i in pars[:2]:
    par_i_UB = par_i.replace("values", "UB")
    par_i_LB = par_i.replace("values", "LB")
    UB_err = df[par_i_UB] - df[par_i]
    LB_err = df[par_i] - df[par_i_LB]
    plt.errorbar(
        times, df[par_i], yerr=(LB_err, UB_err), elinewidth=1, linestyle="", label=par_i
    )
    # plt.errorbar(times, df[par_i], elinewidth=1, fmt="o", linestyle="", label=par_i)
    # plt.ylabel(par_i)
plt.legend()
plt.xlabel("Time")
plt.ylabel("log(T) (MK)")
plt.title(f"flare on {date_init}")
ax2 = plt.twinx()
ax2.grid(visible=False)
net_counts.plot.line("o--", alpha=0.6, color="black", ax=ax2)
ax2.set_ylabel("Photon flux (Counts/second)")
plt.gca().xaxis.set_major_formatter(date_formatter)
plt.tight_layout()
plt.savefig(f"{fig_dir}/Temperature_evolution.png")


plt.figure(figsize=(16, 9))
for par_i in pars[2:4]:
    par_i_UB = par_i.replace("values", "UB")
    par_i_LB = par_i.replace("values", "LB")
    UB_err = df[par_i_UB] - df[par_i]
    LB_err = df[par_i] - df[par_i_LB]
    plt.errorbar(
        times, df[par_i], yerr=(LB_err, UB_err), elinewidth=1, linestyle="", label=par_i
    )
    # plt.errorbar(times, df[par_i], elinewidth=1, fmt="o", linestyle="", label=par_i)
    # plt.ylabel(par_i)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Emission Measure $(10^{46} cm^{-3})$")
plt.title(f"flare on {date_init}")
ax2 = plt.twinx()
ax2.grid(visible=False)
net_counts.plot.line("o--", alpha=0.6, color="black", ax=ax2)
ax2.set_ylabel("Photon flux (Counts/second)")
plt.gca().xaxis.set_major_formatter(date_formatter)
plt.tight_layout()
plt.savefig(f"{fig_dir}/emission_measure_evolution.png")

#%%Plot individual elements

for par_i in pars[4::2]:
    plt.figure(figsize=(16, 9))
    par_i_UB = par_i.replace("values", "UB")
    par_i_LB = par_i.replace("values", "LB")
    UB_err = df[par_i_UB] - df[par_i]
    LB_err = df[par_i] - df[par_i_LB]
    plt.errorbar(
        times, df[par_i], yerr=(LB_err, UB_err), elinewidth=1, linestyle="", label=par_i
    )
    # plt.errorbar(times, df[par_i], elinewidth=1, fmt="o", linestyle="", label=par_i)
    plt.ylabel(par_i)
    plt.xlabel("Time")
    plt.title(f"flare on {date_init}")
    ax2 = plt.twinx()
    ax2.grid(visible=False)
    net_counts.plot.line("o--", alpha=0.6, color="black", ax=ax2)
    ax2.set_ylabel("Photon flux (Counts/second)")
    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{par_i}.png")
#%% Plot for individual temps and norms
for par_i in pars[:4]:
    plt.figure(figsize=(16, 9))
    par_i_UB = par_i.replace("values", "UB")
    par_i_LB = par_i.replace("values", "LB")
    UB_err = df[par_i_UB] - df[par_i]
    LB_err = df[par_i] - df[par_i_LB]
    plt.errorbar(
        times, df[par_i], yerr=(LB_err, UB_err), elinewidth=1, linestyle="", label=par_i
    )
    # plt.errorbar(times, df[par_i], elinewidth=1, fmt="o", linestyle="", label=par_i)
    plt.ylabel(par_i)
    plt.xlabel("Time")
    plt.title(f"flare on {date_init}")
    ax2 = plt.twinx()
    ax2.grid(visible=False)
    net_counts.plot.line("o--", alpha=0.6, color="black", ax=ax2)
    ax2.set_ylabel("Photon flux (Counts/second)")
    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{par_i}.png")
