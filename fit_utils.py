#!/usr/bin/env python3
"""
File: fit_utils.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: Module containing functions for fitting data in XSPEC for solar flares
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xspec as xp


def do_grppha(
    file_list,
    out_put_file_list,
    threshold_counts,
):
    """
    The function will do gpb pha on files based on cutoff cps.

    Parameters
    ----------
    file_list : list of files
    cutoff_cps : cutoff cps
    out_put_file_list : list of output file names.
    """
    for in_file, out_file in zip(file_list, out_put_file_list):
        command = f"grppha infile='{in_file}' outfile='!{out_file}' comm='GROUP MIN {threshold_counts}&exit' "
        os.system(command)


class chisoth_2T:
    """class for running 2T isothermal models"""

    PHA_file_list = None
    FIP_elements = None
    FIRST_TIME = True
    flare_dir = None
    colum_names = None
    arf_files_list = None
    m = None

    def __init__(self, PHA_files, arf_file_list, flare_dir):
        """
        Initialises few things for xspec

        Parameters
        ----------
        PHA_files : list of PHA files to perform fit on
        arf_files : arf files corresponding to the PHA_files
        flare_dir : directory to save results to

        """
        self.PHA_file_list = PHA_files
        self.arf_file_list = arf_file_list
        self.flare_dir = flare_dir
        xp.AllModels.lmod("chspec", dirPath=f"{os.path.expanduser('~')}/chspec/")
        xp.AllData.clear()
        xp.AllModels.clear()
        xp.Fit.query = "no"  # No asking for uesr confirmation while fitting
        xp.Plot.xAxis = "keV"
        xp.Plot.yLog = True
        xp.Plot.xLog = False
        xp.Xset.parallel.leven = 6
        xp.Plot.device = "/xw"

    def init_chisoth(
        self, FIP_elements, max_red_chi=100.0, sigma=1.0, xcm_file="2T.xcm"
    ):
        """
        Initialises 2T chisothermal model with elements in FIP_elements.

        Parameters
        ----------
        FIP_elements : list of elements, these elements will be alway included in fitting.
        max_red_chi : float,  maximum allowed red_chi square value for error calculations.
        sigma : float, sigma for error calculation

        """
        self.FIP_elements = FIP_elements
        self.other_pars = ["logT", "norm"]
        self.fit_pars = self.other_pars + self.FIP_elements
        self.suffix = ["", "_2"]

        # Creating column names for pandas later
        self.colum_names = []
        for fit_pars_i in self.fit_pars:
            for suffix_i in self.suffix:
                self.colum_names.append(fit_pars_i + suffix_i + "_values")
                self.colum_names.append(fit_pars_i + suffix_i + "_UB")
                self.colum_names.append(fit_pars_i + suffix_i + "_LB")
                self.colum_names.append(fit_pars_i + suffix_i + "_err_code")
        self.colum_names.append("Chi")
        self.colum_names.append("red_Chi")
        xp.Xset.restore(xcm_file)
        self.m = xp.AllModels(1, "flare")
        m = self.m
        # Dictionary that will be used to unfreeze parameters
        self.FIP_unfreeze_dict = {}
        self.temperature_unfreeze_dict = {
            eval("m.chisoth.logT.index"): "6.8,0.01,,,,,",
            eval("m.chisoth_2.logT.index"): "7.2,0.01,,,,,",
        }

        self.other_par_index = []
        model_components_list = m.componentNames
        for component_i in model_components_list:
            for other_pars_i in self.other_pars:
                idx_temp = eval(f"m.{component_i}.{other_pars_i}.index")
                self.other_par_index.append(idx_temp)

        self.FIP_par_index = []
        for FIP_el in FIP_elements:
            idx_temp = eval(f"m.chisoth.{FIP_el}.index")
            par_1 = eval(f"m.chisoth.{FIP_el}")
            par_2 = eval(f"m.chisoth_2.{FIP_el}")
            par_2.link = par_1
            idx_temp = eval(f"m.chisoth.{FIP_el}.index")
            self.FIP_par_index.append(idx_temp)
            self.FIP_unfreeze_dict[idx_temp] = ",0.01,,,,,"

        self.all_par_index = self.other_par_index + self.FIP_par_index
        self.err_string = f"maximum {max_red_chi} {sigma} flare:" + "".join(
            [str(i) + " " for i in self.all_par_index]
        )  # error string to be used later with xspec err command
        print(f"error string is {self.err_string}")

    def load_spectra(self, file_idx):
        """
        Loads spectra to XSPEC based on whether there is a single file or multiple
        files.

        Parameters
        ----------
        file_idx : index of the file in self.PHA_file_list to load.

        """
        PHA_files = self.PHA_file_list[file_idx]
        arf_files = self.arf_file_list[file_idx]
        if type(PHA_files) != list:
            PHA_files = [PHA_files]
            arf_files = [arf_files]

        for i, spectra_i in enumerate(PHA_files):
            xp.AllData += spectra_i
            if arf_files[i] != "USE_DEFAULT":
                xp.AllData(i + 1).response.arf = arf_files[i]

    def find_fit_elements(
        self,
        cps,
        energies,
        threshold=10,
        element_line_dict={"S": 2.45, "Ar": 3.2, "Ca": 3.9, "Fe": 6.5},
    ):
        """
        Function to generate a list of elements to consider based on cps in spectra.
        Function takes dictionary of elemental lies, add the element to the list
        of elements to consider if the number of counts after the line energy
        is more than the threshold.

        Parameters
        ----------
        cps : array, count per second of spectra.
        energy_bins: array, of energy bins
        threshold : minimum counts per second for which elements to considered.
        element_line_dict : a dictionary with keys as elements and values line energies in keV.

        Returns
        -------
        list of elements for fitting.

        """
        fit_element_list = []
        # Dictionary with keys as elements and values as line energys
        for element_i in element_line_dict:
            mask = energies > element_line_dict[element_i]
            if np.sum(cps[mask]) > threshold:
                fit_element_list.append(element_i)
        return fit_element_list

    def plot_fit(self, out_file):
        """
        Plots the rusults of current fit to the file

        Parameters
        ----------
        out_file : string, path name to save the plot to.
        Returns
        -------
        TODO

        """
        ##% Plot
        # Stuff required for plotting
        # xp.Plot.device = "/xs"
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
        ax[0].errorbar(
            x, y, xerr=x_err, yerr=y_err, linestyle="None", fmt="k", alpha=0.6
        )
        ax[0].set_yscale("log")
        ax[0].set_ylim(bottom=1)
        ax[0].set_ylabel("counts/s/keV")
        ax[1].hlines(
            0,
            min(chix),
            max(chix),
        )
        ax[1].errorbar(
            chix,
            chi,
            xerr=chix_err,
            yerr=chi_err,
            linestyle="None",
            fmt="k",
            alpha=0.6,
        )
        ax[1].set_ylabel("(data-model)/error")
        fig.supxlabel("Energy (keV)")
        plt.savefig(out_file)
        plt.close()

    def fit(self, min_E, max_E=15.0, cutoff_cps=1.0, do_dynamic_elements=False):
        """
        fits the data iwth models.

        Parameters
        ----------
        min_E : minimum energy cut off for fitting
        max_E : maxmimum energy cut off for fitting
        cutoff_cps : minimum number of counts to be considered to be inclued in fit.
        dynamic_elements : bool, will dynamically add more elements to fit depending on counts.

        Returns
        -------
        df, pandas dataframe with results
        """

        xp.AllData.clear()
        m = self.m
        out_dir = f"{self.flare_dir}/fit"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        xp.AllData.clear()
        self.par_vals = []
        PHA_file_array = np.array(self.PHA_file_list)

        # Check if the there is multiple instruments
        if PHA_file_array.ndim == 1:
            file_names = [
                os.path.basename(PHA_file).removesuffix(".pha")
                for PHA_file in PHA_file_array
            ]
        else:
            file_names = [
                "simult" + os.path.basename(PHA_file).removesuffix(".pha")[-30:]
                for PHA_file in PHA_file_array[:, 0]
            ]
        #%% Fit
        for i, f_name in enumerate(file_names):
            m.setPars(self.temperature_unfreeze_dict)
            m.setPars(self.FIP_unfreeze_dict)
            xp.AllData.clear()
            logFile = xp.Xset.openLog(f"{out_dir}/{f_name}.log")
            self.load_spectra(i)
            xp.Fit.statMethod = "chi"

            # Some dynamic adjustments for fitting
            for i in range(xp.AllData.nSpectra):
                # temp_max_E = max_E
                s = xp.AllData(i + 1)
                s.ignore(f"**-{min_E}")
                s.ignore(f"{max_E}-**")
                counts = np.array(s.values)
                energies = np.array(s.energies)
                # Implementing dynamic maximum cut off
                idxs = np.where(counts < cutoff_cps)[0][0]
                if idxs.size > 0:
                    cutoff_idx = np.where(counts < cutoff_cps)[0][0]
                    cutoff_energy = energies[cutoff_idx][1]
                    if cutoff_energy < max_E:
                        s.ignore(f"{cutoff_energy}-**")
            # Implementing dynamic addition of elements
            # if do_dynamic_elements:

            # Satrt fitting
            m.setPars(self.temperature_unfreeze_dict)
            xp.Fit.renorm()
            xp.Fit.perform()
            xp.Fit.renorm()
            xp.Fit.perform()
            xp.Fit.renorm()
            xp.Fit.perform()
            n = 0
            while xp.Fit.testStatistic > 150 and n < 5:
                n += 1
                xp.Fit.renorm()
                xp.Fit.perform()
            # Finding errors
            xp.Fit.error(self.err_string)
            xp.Xset.save(f"{out_dir}/{f_name}.xcm")
            xp.Xset.closeLog()
            temp_col = []
            for fit_pars_i in self.fit_pars:
                for suffix_i in self.suffix:
                    m_par_i = eval(f"m.chisoth{suffix_i}.{fit_pars_i}")
                    temp_col.append(m_par_i.values[0])
                    temp_col.append(m_par_i.error[0])
                    temp_col.append(m_par_i.error[1])
                    temp_col.append(m_par_i.error[2])
            temp_col.append(xp.Fit.testStatistic)
            temp_col.append(xp.Fit.testStatistic / xp.Fit.dof)
            self.par_vals.append(temp_col)
            self.plot_fit(f"{out_dir}/{f_name}.png")
        #%% Make a data frame
        times = [
            os.path.basename(PHA_file_i).removesuffix(".pha")[-29:]
            for PHA_file_i in self.PHA_file_list
        ]
        times = pd.to_datetime(times)

        df = pd.DataFrame(self.par_vals, columns=self.colum_names, index=times)
        df.to_csv(f"{out_dir}/results.csv")
        df.to_hdf(f"{out_dir}/results.h5", "results")
        return df
