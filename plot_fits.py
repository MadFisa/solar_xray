#!/usr/bin/env python3
"""
File: plot_fits.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: A code to plot fits made with the other codes
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os

# Set up plotting stuff
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.style.use("fivethirtyeight")
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["font.size"] = 24
date_formatter = DateFormatter("%H:%M")

# define data
flare_num = 7
flare_dir = f"./data/pha/flare_num_{flare_num}/daxss"
out_dir = f"{flare_dir}/fit"
results = f'{out_dir}/results.csv'
# results = f'{out_dir}/results.h5'
observation_file = "./flare_filtering/data/flare_observation.h5"
fig_dir=f'{out_dir}/figures'
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

#Load data
observation_table = pd.read_hdf(observation_file, "obs")
df = pd.read_csv(results,parse_dates=True,index_col=0)
# df = pd.read_hdf(results,parse_dates=True)

net_counts = observation_table.loc[7]['daxss_lc']
times = df.index
flare = observation_table.loc[flare_num]
time_beg = flare["event_starttime"]
time_end = flare["event_endtime"]
peak_time = flare["event_peaktime"]
flare_class = flare["fl_goescls"]
daxss_lc = flare["daxss_lc"]

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
plt.title(f"flare on {peak_time}")
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
plt.title(f"flare on {peak_time}")
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
    plt.title(f"flare on {peak_time}")
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
    plt.title(f"flare on {peak_time}")
    ax2 = plt.twinx()
    ax2.grid(visible=False)
    net_counts.plot.line("o--", alpha=0.6, color="black", ax=ax2)
    ax2.set_ylabel("Photon flux (Counts/second)")
    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{par_i}.png")
