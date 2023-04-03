#!/usr/bin/env python3
"""
File: plot_fits.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: A code to plot fits made with the other codes
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter

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


def find_par_value(df, par_i, MASK=False):
    """
    A function to find par values from the dataframe and output values that can be used in plt.errorbar.

    Parameters
    ----------
    df : Dataframe with par values
    par_i : name of the parameter
    MASK : Whether or not to mask par values with error codes other than 'FFFFFFFF', optional

    Returns
    -------
    TODO

    """
    times_par = df.index
    par_i_UB = par_i.replace("values", "UB")
    par_i_LB = par_i.replace("values", "LB")
    par_i_err_code = par_i.replace("values", "err_code")
    UB_err = df[par_i_UB] - df[par_i]
    LB_err = df[par_i] - df[par_i_LB]
    vals = df[par_i]
    if MASK:
        mask = df[par_i_err_code] == "FFFFFFFFF"
        UB_err = UB_err[mask]
        LB_err = LB_err[mask]
        vals = vals[mask]
        times_par = times_par[mask]

    return times_par, vals, LB_err, UB_err


def plot_individual(
    instrument, data_dir, flare_num, out_dir=None, PLOT_LIGHTCURVE=True, MASK=False
):
    """
    Function to plot individual components of the fit.

    Parameters
    ----------
    instrument :str, name of the intrumner i.e 'xsm' or 'daxss'
    data_dir : str,path to where all the flare folders are presennt
    flare_num : int, flare's number in the observation_table
    out_dir : str, output dir to which figures should be saved. If none goes to default.
    PLOT_LIGHTCURVE : bool, optional. Whether to plot light curve along with values.
    MASK : Whether or not to mask par values with error codes other than 'FFFFFFFF', optional

    Returns
    -------

    """
    flare_dir = f"{data_dir}/flare_num_{flare_num}/{instrument}"
    if out_dir is None:
        out_dir = f"{flare_dir}/fit"
    results = f"{flare_dir}/fit/results.csv"
    fig_dir = f"{out_dir}/figures"
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    # Load data
    observation_table = pd.read_hdf(observation_file, "obs")
    df = pd.read_csv(results, parse_dates=True, index_col=0)
    # df = pd.read_hdf(results,parse_dates=True)
    times = df.index
    flare = observation_table.loc[flare_num]
    peak_time = flare["event_peaktime"]

    #%% Plot Individual
    pars = df.columns[0:-1:4]
    plt.figure(figsize=(16, 9))
    for par_i in pars[:2]:
        times_pars, vals, LB_err, UB_err = find_par_value(df, par_i, MASK)
        plt.errorbar(
            times_pars,
            vals,
            yerr=(LB_err, UB_err),
            elinewidth=1,
            capsize=6,
            linestyle="",
            label=par_i,
        )
        # plt.ylabel(par_i)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("log(T) (K)")
    plt.title(f"flare {flare_num} on {peak_time}")
    ax2 = plt.twinx()
    ax2.grid(visible=False)
    if PLOT_LIGHTCURVE:
        net_counts = observation_table.loc[flare_num][f"{instrument}_lc"]
        net_counts.plot.line("o--", alpha=0.6, color="black", ax=ax2)
    ax2.set_ylabel(r"Irradinace ($W/m^2$)")
    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/Temperature_evolution.png")

    plt.figure(figsize=(16, 9))
    for par_i in pars[2:4]:
        times_pars, vals, LB_err, UB_err = find_par_value(df, par_i, MASK)
        plt.errorbar(
            times_pars,
            vals,
            yerr=(LB_err, UB_err),
            elinewidth=1,
            capsize=6,
            linestyle="",
            label=par_i,
        )
        # plt.ylabel(par_i)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Emission Measure $(10^{46} cm^{-3})$")
    plt.title(f"flare {flare_num} on {peak_time}")
    ax2 = plt.twinx()
    ax2.grid(visible=False)
    if PLOT_LIGHTCURVE:
        net_counts.plot.line("o--", alpha=0.6, color="black", ax=ax2)
    ax2.set_ylabel(r"Irradinace ($W/m^2$)")
    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/emission_measure_evolution.png")

    #%%Plot individual elements

    for par_i in pars[4::2]:
        plt.figure(figsize=(16, 9))
        times_pars, vals, LB_err, UB_err = find_par_value(df, par_i, MASK)
        plt.errorbar(
            times_pars,
            vals,
            yerr=(LB_err, UB_err),
            elinewidth=1,
            capsize=6,
            linestyle="",
            label=par_i,
        )
        plt.ylabel(par_i)
        plt.xlabel("Time")
        plt.title(f"flare {flare_num} on {peak_time}")
        ax2 = plt.twinx()
        ax2.grid(visible=False)
        if PLOT_LIGHTCURVE:
            net_counts.plot.line("o--", alpha=0.3, color="black", ax=ax2)
        ax2.set_ylabel(r"Irradinace ($W/m^2$)")
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
            times,
            df[par_i],
            yerr=(LB_err, UB_err),
            elinewidth=1,
            capsize=6,
            linestyle="",
            label=par_i,
        )
        plt.ylabel(par_i)
        plt.xlabel("Time")
        plt.title(f"flare {flare_num} on {peak_time}")
        ax2 = plt.twinx()
        ax2.grid(visible=False)
        if PLOT_LIGHTCURVE:
            net_counts.plot.line("o--", alpha=0.6, color="black", ax=ax2)
        ax2.set_ylabel(r"Irradinace ($W/m^2$)")
        plt.gca().xaxis.set_major_formatter(date_formatter)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/{par_i}.png")

    #%% Plot reduced chi2
    par_i = "red_Chi"
    plt.figure(figsize=(16, 9))
    par_i_UB = par_i.replace("values", "UB")
    par_i_LB = par_i.replace("values", "LB")
    UB_err = df[par_i_UB] - df[par_i]
    LB_err = df[par_i] - df[par_i_LB]
    plt.plot(times, df[par_i], "o", label=par_i)
    plt.ylabel(par_i)
    plt.xlabel("Time")
    plt.title(f"flare {flare_num} on {peak_time}")
    ax2 = plt.twinx()
    ax2.grid(visible=False)
    if PLOT_LIGHTCURVE:
        net_counts.plot.line("o--", alpha=0.6, color="black", ax=ax2)
    ax2.set_ylabel(r"Irradinace ($W/m^2$)")
    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{par_i}.png")
    plt.close("all")


def plot_simult(flare_dir, flare_num, instruments, fig_dir=None, MASK=False):
    """
    Plots multiple instrument results simulatneously

    Parameters
    ----------
    flare_num : int, flare number in observation_file
    instruments : list, list of instrument names to plot
    fig_dir : str, output dir to which figures should be saved. If none goes to default.
    MASK : Whether or not to mask par values with error codes other than 'FFFFFFFF', optional
    """
    fit_dirs = []

    flare_dir = f"{flare_dir}/flare_num_{flare_num}"
    fit_dirs = [f"{flare_dir}/{instrument}/fit" for instrument in instruments]
    results_files = [f"{fit_dir}/results.csv" for fit_dir in fit_dirs]
    if fig_dir is None:
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
        if instrument != "simult":
            net_counts = observation_table.loc[flare_num][f"{instrument}_lc"]
        else:
            net_counts = []
        net_counts_list.append(net_counts)
        times = df.index
        df_list.append(df)
        times_list.append(times)
        if len(df.columns) > pars_len:
            pars = df.columns[0:-1:4]
            pars_len = len(df.columns)
    flare = observation_table.loc[flare_num]
    peak_time = flare["event_peaktime"]
    #%% plots
    for par_i in pars[4::2]:
        plt.figure(figsize=(16, 9))
        for instrument, times, df in zip(instruments, times_list, df_list):
            if par_i in df.columns:
                times_pars, vals, LB_err, UB_err = find_par_value(df, par_i, MASK)
                plt.errorbar(
                    times_pars,
                    vals,
                    yerr=(LB_err, UB_err),
                    elinewidth=1,
                    capsize=6,
                    linestyle="",
                    label=par_i,
                )
                plt.ylabel(par_i)
                plt.xlabel("Time")
                plt.title(f"flare {flare_num} on {peak_time}")
        ax2 = plt.twinx()
        for instrument, net_counts in zip(instruments, net_counts_list):
            net_counts.plot.line("o--", alpha=0.3, ax=ax2, label=instrument)
        ax2.grid(visible=False)
        ax2.set_ylabel(r"Irradiance ($W/m^2$)")
        plt.gca().xaxis.set_major_formatter(date_formatter)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{fig_dir}/{par_i}.png")
        #%% Plot for individual temps and norms
    for par_i in pars[:4]:
        plt.figure(figsize=(16, 9))
        for instrument, times, df in zip(instruments, times_list, df_list):
            if par_i in df.columns:
                times_pars, vals, LB_err, UB_err = find_par_value(df, par_i, MASK)
                plt.errorbar(
                    times_pars,
                    vals,
                    yerr=(LB_err, UB_err),
                    elinewidth=1,
                    capsize=6,
                    linestyle="",
                    label=par_i,
                )
                plt.ylabel(par_i)
                plt.xlabel("Time")
                plt.title(f"flare {flare_num} on {peak_time}")
        ax2 = plt.twinx()
        for instrument, net_counts in zip(instruments, net_counts_list):
            if instrument != "simult":
                net_counts.plot.line("o--", alpha=0.3, ax=ax2, label=instrument)
        ax2.grid(visible=False)
        ax2.set_ylabel(r"Irradiance ($W/m^2$)")
        plt.gca().xaxis.set_major_formatter(date_formatter)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{fig_dir}/{par_i}.png")

        #%% Plot reduced chi2
    par_i = "red_Chi"
    plt.figure(figsize=(16, 9))
    for instrument, times, df in zip(instruments, times_list, df_list):
        times_pars, vals, LB_err, UB_err = find_par_value(df, par_i, MASK)
        plt.errorbar(
            times_pars,
            vals,
            yerr=(LB_err, UB_err),
            elinewidth=1,
            capsize=6,
            linestyle="",
            label=par_i,
        )
    plt.ylabel(par_i)
    plt.xlabel("Time")
    plt.title(f"flare {flare_num} on {peak_time}")
    ax2 = plt.twinx()
    for instrument, net_counts in zip(instruments, net_counts_list):
        if instrument != "simult":
            net_counts.plot.line("--", alpha=0.3, ax=ax2, label=instrument)
    ax2.grid(visible=False)
    ax2.set_ylabel(r"Irradiance ($W/m^2$)")
    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{fig_dir}/{par_i}.png")
    plt.close("all")
