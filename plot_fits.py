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
observation_file = "./flare_filtering/data/flare_observation.h5"

def plot_individual(instrument, flare_num):
    """
    Function to plot individual components of the fit.

    Parameters
    ----------
    instrument :str, name of the intrumner i.e 'xsm' or 'daxss'
    flare_num : int, flare's number in the observation_table

    Returns
    -------

    """
    flare_dir = f"./data/pha/flare_num_{flare_num}/{instrument}"
    out_dir = f"{flare_dir}/fit"
    results = f'{out_dir}/results.csv'
    fig_dir=f'{out_dir}/figures'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    #Load data
    observation_table = pd.read_hdf(observation_file, "obs")
    df = pd.read_csv(results,parse_dates=True,index_col=0)
    # df = pd.read_hdf(results,parse_dates=True)
    net_counts = observation_table.loc[flare_num][f'{instrument}_lc']
    times = df.index
    flare = observation_table.loc[flare_num]
    peak_time = flare["event_peaktime"]

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
    plt.title(f"flare {flare_num} on {peak_time}")
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
    plt.title(f"flare {flare_num} on {peak_time}")
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
        plt.title(f"flare {flare_num} on {peak_time}")
        ax2 = plt.twinx()
        ax2.grid(visible=False)
        net_counts.plot.line("o--", alpha=0.3, color="black", ax=ax2)
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
        plt.title(f"flare {flare_num} on {peak_time}")
        ax2 = plt.twinx()
        ax2.grid(visible=False)
        net_counts.plot.line("o--", alpha=0.6, color="black", ax=ax2)
        ax2.set_ylabel("Photon flux (Counts/second)")
        plt.gca().xaxis.set_major_formatter(date_formatter)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/{par_i}.png")

    #%% Plot reduced chi2
    par_i='red_Chi'
    plt.figure(figsize=(16, 9))
    par_i_UB = par_i.replace("values", "UB")
    par_i_LB = par_i.replace("values", "LB")
    UB_err = df[par_i_UB] - df[par_i]
    LB_err = df[par_i] - df[par_i_LB]
    plt.plot(
        times, df[par_i] , '--o', label=par_i
    )
    # plt.errorbar(times, df[par_i], elinewidth=1, fmt="o", linestyle="", label=par_i)
    plt.ylabel(par_i)
    plt.xlabel("Time")
    plt.title(f"flare {flare_num} on {peak_time}")
    ax2 = plt.twinx()
    ax2.grid(visible=False)
    net_counts.plot.line("o--", alpha=0.6, color="black", ax=ax2)
    ax2.set_ylabel("Photon flux (Counts/second)")
    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{par_i}.png")
    plt.close('all')


def plot_simult(flare_num, instruments):
    """
    Plots multiple instrument results simulatneously

    Parameters
    ----------
    flare_num : int, flare number in observation_file
    instruments : list, list of instruments to plot

    Returns
    -------

    """
    fit_dirs = []

    flare_dir = f"./data/pha/flare_num_{flare_num}"
    fit_dirs = [f"{flare_dir}/{instrument}/fit" for instrument in instruments]
    results_files = [f"{fit_dir}/results.csv" for fit_dir in fit_dirs]
    fig_dir = f"{flare_dir}/figures"
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    # Load data
    observation_table = pd.read_hdf(observation_file, "obs")
    df_list = []
    net_counts_list = []
    times_list = []
    pars_list = []
    pars_len = 0

    for i, instrument in enumerate(instruments):
        df = pd.read_csv(results_files[i], parse_dates=True, index_col=0)
        net_counts = observation_table.loc[flare_num][f"{instrument}_lc"]
        times = df.index
        df_list.append(df)
        net_counts_list.append(net_counts)
        times_list.append(times)
        if len(df.columns)>pars_len:
            pars = df.columns[0:-1:4]
            pars_len = len(df.columns)
    flare = observation_table.loc[flare_num]
    peak_time = flare["event_peaktime"]
    #%% plots
    for par_i in pars[4::2]:
        plt.figure(figsize=(16, 9))
        for instrument, times, df in zip(
            instruments, times_list, df_list
        ):
            if par_i in df.columns:
                par_i_UB = par_i.replace("values", "UB")
                par_i_LB = par_i.replace("values", "LB")
                UB_err = df[par_i_UB] - df[par_i]
                LB_err = df[par_i] - df[par_i_LB]
                plt.errorbar(
                    times,
                    df[par_i],
                    yerr=(LB_err, UB_err),
                    elinewidth=3,
                    linestyle="",
                    label=instrument,
                )
                # plt.errorbar(times, df[par_i], elinewidth=1, fmt="o", linestyle="", label=par_i)
                plt.ylabel(par_i)
                plt.xlabel("Time")
                plt.title(f"flare {flare_num} on {peak_time}")
                # net_counts.plot.line("o--", alpha=0.6 , ax=ax2,label=instrument)
        ax2 = plt.twinx()
        for instrument,net_counts in zip(instruments,net_counts_list) :
            net_counts.plot.line("o--", alpha=0.3,  ax=ax2,label=instrument)
        ax2.grid(visible=False)
        ax2.set_ylabel(r"Irradiance ($W/m^2$)")
        plt.gca().xaxis.set_major_formatter(date_formatter)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{fig_dir}/{par_i}.png")
        #%% Plot for individual temps and norms
    for par_i in pars[:4]:
        plt.figure(figsize=(16, 9))
        for instrument, times, df in zip(
            instruments, times_list, df_list
        ):
            if par_i in df.columns:
                par_i_UB = par_i.replace("values", "UB")
                par_i_LB = par_i.replace("values", "LB")
                UB_err = df[par_i_UB] - df[par_i]
                LB_err = df[par_i] - df[par_i_LB]
                plt.errorbar(
                    times,
                    df[par_i],
                    yerr=(LB_err, UB_err),
                    elinewidth=3,
                    linestyle="",
                    label=instrument,
                )
                # plt.errorbar(times, df[par_i], elinewidth=1, fmt="o", linestyle="", label=par_i)
                plt.ylabel(par_i)
                plt.xlabel("Time")
                plt.title(f"flare {flare_num} on {peak_time}")
                # net_counts.plot.line("o--", alpha=0.6 , ax=ax2,label=instrument)
        ax2 = plt.twinx()
        for instrument,net_counts in zip(instruments,net_counts_list) :
            net_counts.plot.line("o--", alpha=0.3,  ax=ax2,label=instrument)
        ax2.grid(visible=False)
        ax2.set_ylabel(r"Irradiance ($W/m^2$)")
        plt.gca().xaxis.set_major_formatter(date_formatter)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{fig_dir}/{par_i}.png")

        #%% Plot reduced chi2
    par_i = "red_Chi"
    plt.figure(figsize=(16, 9))
    for instrument,times,df in zip(instruments,times_list,df_list):
        par_i_UB = par_i.replace("values", "UB")
        par_i_LB = par_i.replace("values", "LB")
        UB_err = df[par_i_UB] - df[par_i]
        LB_err = df[par_i] - df[par_i_LB]
        plt.plot(times, df[par_i], "--o", label=par_i)
        # plt.errorbar(times, df[par_i], elinewidth=1, fmt="o", linestyle="", label=par_i)
    plt.ylabel(par_i)
    plt.xlabel("Time")
    plt.title(f"flare {flare_num} on {peak_time}")
    ax2 = plt.twinx()
    for instrument,net_counts in zip(instruments,net_counts_list) :
        net_counts.plot.line("--", alpha=0.3,  ax=ax2,label=instrument)
    ax2.grid(visible=False)
    ax2.set_ylabel(r"Irradiance ($W/m^2$)")
    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{fig_dir}/{par_i}.png")
    plt.close('all')
