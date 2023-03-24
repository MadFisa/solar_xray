#!/usr/bin/env python3
"""
File: models.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: Module containing models for fitting data in XSPEC for solar flares
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xspec as xp


class model:

    """Base class for all models"""

    def __init__(
        self,
        PHA_files_list,
        arf_files_list,
        output_dir,
    ):
        """
        Initialisation takes list of PHA_files_list,arf_files,output_dir.

        Parameters
        ----------
        PHA_files_list : TODO
        arf_files_list : TODO
        output_dir : TODO


        """
        self.PHA_files_list = PHA_files_list
        self.arf_files_list = arf_files_list
        self.output_dir = output_dir
        xp.Fit.query = "no"  # No asking for uesr confirmation while fitting
        xp.Plot.xAxis = "keV"
        xp.Plot.yLog = True
        xp.Plot.xLog = False
        xp.Xset.parallel.leven = 6
        xp.Plot.device = "/xw"

    def load_spectra(self, file_idx):
        """
        Loads spectra to XSPEC based on whether there is a single file or multiple
        files.

        Parameters
        ----------
        file_idx : index of the file in self.PHA_files_list to load.

        """
        PHA_files = self.PHA_files_list[file_idx]
        arf_files = self.arf_files_list[file_idx]
        if type(PHA_files) != list:
            PHA_files = [PHA_files]
            arf_files = [arf_files]

        for i, spectra_i in enumerate(PHA_files):
            xp.AllData += spectra_i
            if arf_files[i] != "USE_DEFAULT":
                xp.AllData(i + 1).response.arf = arf_files[i]

    def plot_fit(self, out_file):
        """
        Plots the rusults of current fit to the file

        Parameters
        ----------
        out_file : string, path name to save the plot to.

        """
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


class chisoth(model):
    """A template class for chisoth models"""

    def __init__(
        self,
        PHA_files_list,
        arf_files_list,
        output_dir,
        FIP_elements,
        other_pars=["logT", "norm"],
        xcm_file=None,
    ):
        super(chisoth, self).__init__(PHA_files_list, arf_files_list, output_dir)
        self.xcm_file = xcm_file
        self.FIP_elements = FIP_elements
        self.other_pars = other_pars

        # Load model
        xp.Fit.statMethod = "chi"
        xp.AllModels.lmod("chspec", dirPath=f"{os.path.expanduser('~')}/chspec/")
        xp.AllData.clear()
        xp.AllModels.clear()

    def find_fit_elements(
        self,
        cps,
        energies,
        element_line_dict,
        threshold=10,
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
        element_line_dict : a dictionary with keys as elements and values line energies in keV. Used to dynamically add elements to fit.
        threshold : minimum counts per second for which elements to considered.

        Returns
        -------
        list of elements for fitting.

        """
        self.element_line_dict = element_line_dict
        fit_element_list = []
        # Dictionary with keys as elements and values as line energys
        for element_i in self.element_line_dict:
            mask = energies > self.element_line_dict[element_i]
            if np.sum(cps[mask]) > threshold:
                fit_element_list.append(element_i)
        return fit_element_list

    def setup_pars(self, elements_list):
        """
        Function that will take a list of element and unfreezes them for fitting.

        Parameters
        ----------
        elements_list : list, of elements to unfreeze

        Returns
        -------

        elem_index_dict : dictionary that gives parameter indexes corresponding to elements.

        """

        elem_index_dict = {}
        for elem in elements_list:
            elem_par = eval(f"self.m.chisoth.{elem}")
            elem_idx = elem_par.index
            elem_index_dict[elem] = elem_idx
            elem_par.frozen = False
        return elem_index_dict

    def create_rows(self, fit_pars, suffix):
        """
        Creates dictionary to which will be used create rows of dataframe later.

        Parameters
        ----------
        fit_pars :list, parameters that was used in fitting,
        suffix :list, suffix of models

        Returns
        -------
        TODO

        """
        temp_col = {}
        for fit_pars_i in fit_pars:
            for suffix_i in suffix:
                m_par_i = eval(f"self.m.chisoth{suffix_i}.{fit_pars_i}")
                par_col_prefix = fit_pars_i + suffix_i
                temp_col[f"{par_col_prefix}_values"] = m_par_i.values[0]
                temp_col[f"{par_col_prefix}_UB"] = m_par_i.error[0]
                temp_col[f"{par_col_prefix}_LB"] = m_par_i.error[1]
                temp_col[f"{par_col_prefix}_err_code"] = m_par_i.error[2]
        temp_col["Chi"] = xp.Fit.testStatistic
        temp_col["red_Chi"] = xp.Fit.testStatistic / xp.Fit.dof
        print(f"Fit results are: {temp_col}")
        return temp_col

    def notice_fit_energies(self, min_E, max_E, cutoff_cps):
        """
        Adjust the energy range needs to be used. Initially the range between
        min_E and max_E is used. Then the energy bins with less than cutoff cps
        is ignored within this range.

        Parameters
        ----------
        min_E : float, minimum energy to notice
        max_E : float, maximum energy to notice
        cutoff_cps : float, cutoff cps beyond which spectreum won't be counted

        Returns
        -------
        TODO

        """
        for i in range(xp.AllData.nSpectra):
            # temp_max_E = max_E
            s = xp.AllData(i + 1)
            s.ignore(f"**-{min_E}")
            s.ignore(f"{max_E}-**")
            counts = np.array(s.values)
            energies = np.array(s.energies)
            # Implementing dynamic maximum cut off
            idxs = np.where(counts < cutoff_cps)[0]
            if idxs.size > 0:
                cutoff_idx = idxs[0]
                cutoff_energy = energies[cutoff_idx][1]
                if cutoff_energy < max_E:
                    s.ignore(f"{cutoff_energy}-**")

    def find_dynamic_elements(self, element_line_dict):
        """TODO: Docstring for find_dynamic_elements.

        Parameters
        ----------
        element_line_dict : a dictionary with keys as elements and values line energies in keV. Used to dynamically add elements to fit.
        refer to find_fit_elements documentation for more information.

        Returns
        -------
        TODO

        """
        fit_elements = self.FIP_elements
        for i in range(xp.AllData.nSpectra):
            s = xp.AllData(i + 1)
            counts = np.array(s.values)
            energies = np.array(s.energies)
            dyn_elements = self.find_fit_elements(
                counts, energies[:, 0], element_line_dict=element_line_dict
            )
            for dyn_element_i in dyn_elements:
                if dyn_element_i not in fit_elements:
                    fit_elements.append(dyn_element_i)
        return fit_elements

    def create_err_string(self, elements_list, max_red_chi, sigma):
        """

        Parameters
        ----------
        elements_list : list, list of elements.
        max_red_chi : float, maximum reduced chi for which the error to be calculated
        sigma : sigma for which error is calculated

        """
        self.elem_par_idx = []
        for element_i in elements_list:
            idx_elem = eval(f"self.m.chisoth.{element_i}.index")
            self.elem_par_idx.append(idx_elem)
        self.all_par_idx = self.other_pars_idx + self.elem_par_idx
        self.err_string = f"maximum {max_red_chi} {sigma} flare:" + "".join(
            [str(i) + " " for i in self.all_par_idx]
        )  # error string to be used later with xspec err command
        print(f"error string is {self.err_string}")

    def prefit(self, output_dir):
        """
        Function containing stuff to do before prefitting

        Parameters
        ----------
        out_dir : directory to output files to .

        Returns
        -------
        file_names, a list of file names that will be used as prefix to create outputs
        out_dir, directory to which fit results will be written to

        """

        if output_dir is not None:
            self.output_dir = output_dir

        xp.AllData.clear()
        out_dir = f"{self.output_dir}/fit"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        self.par_vals = []
        PHA_files_array = np.array(self.PHA_files_list)

        # Check if the there is multiple instruments
        if PHA_files_array.ndim == 1:
            file_names = [
                os.path.basename(PHA_file).removesuffix(".pha")
                for PHA_file in PHA_files_array
            ]
        else:
            file_names = [
                "simult" + os.path.basename(PHA_file).removesuffix(".pha")[-30:]
                for PHA_file in PHA_files_array[:, 0]
            ]
        return file_names

    def do_fit(
        self,
        file_names,
        out_dir,
        min_E,
        max_E,
        cutoff_cps,
        do_dynamic_elements,
        do_error_calculation,
        element_line_dict,
        suffix,
    ):
        """
        A function that ill iterate through output file_names and do fitting.

        Parameters
        ----------
        file_names : file_names for output
        min_E : minimum energy cut off for fitting
        max_E : maxmimum energy cut off for fitting
        cutoff_cps : minimum number of counts to be considered to be inclued in fit.
        do_dynamic_elements : bool, will dynamically add more elements to fit depending on counts.
        do_error_calculation: bool, whether to do errorr calculation or not.
        max_red_chi : float,  maximum allowed red_chi square value for error calculations.
        sigma : float, sigma for error calculation
        element_line_dict : a dictionary with keys as elements and values line energies in keV. Used to dynamically add elements to fit.
        refer to find_fit_elements documentation for more information.

        Returns
        -------
         dictionary of parvals.

        """
        self.par_vals = []
        for i, f_name in enumerate(file_names):
            xp.AllData.clear()
            xp.AllModels.clear()
            logFile = xp.Xset.openLog(f"{out_dir}/{f_name}.log")
            xp.Xset.restore(self.xcm_file)
            self.m = xp.AllModels(1, "flare")
            self.load_spectra(i)
            self.notice_fit_energies(min_E, max_E, cutoff_cps)

            # Some dynamic adjustments for fitting
            # Implementing dynamic addition of elements
            if do_dynamic_elements:
                fit_elements = self.find_dynamic_elements(element_line_dict)
            else:
                fit_elements = self.FIP_elements
            self.fit_elements_list.append(fit_elements)
            print(f"Elements for fitting are {fit_elements}")
            # Start fitting with just temperatures left free
            other_pars_unfreeze_dict = {
                idx_i: ",0.01,,,,," for idx_i in self.other_pars_idx
            }
            self.m.setPars(other_pars_unfreeze_dict)
            xp.Fit.renorm()
            xp.Fit.perform()
            xp.Fit.renorm()
            xp.Fit.perform()
            xp.Fit.renorm()
            xp.Fit.perform()

            # Unfreeze elemets
            elments_par_dict = self.setup_pars(fit_elements)
            xp.Fit.renorm()
            xp.Fit.perform()
            xp.Fit.renorm()
            xp.Fit.perform()
            n = 0
            red_chi = xp.Fit.testStatistic / xp.Fit.dof
            while red_chi > 2 and n < 5:
                n += 1
                xp.Fit.renorm()
                xp.Fit.perform()
                red_chi = xp.Fit.testStatistic / xp.Fit.dof
            # Finding errors
            self.create_err_string(fit_elements, self.max_red_chi, self.sigma)
            xp.Fit.error(self.err_string)
            fit_pars = self.other_pars + fit_elements

            # Store the parameter values for later to turn into dataframe
            temp_row = self.create_rows(fit_pars, suffix)
            self.par_vals.append(temp_row)
            with open(f"{out_dir}/{f_name}.json", "w") as fp:
                json.dump(temp_row, fp)
            xp.Xset.save(f"{out_dir}/{f_name}.xcm")  # Save model to xcm file
            xp.Xset.closeLog()
            self.plot_fit(f"{out_dir}/{f_name}.png")
        return self.par_vals

    def save_fit(self, out_dir):
        """
        Saves current fit result to output_dir

        Returns
        -------
        TODO

        """
        times = [
            os.path.basename(PHA_file_i).removesuffix(".pha")[-29:]
            for PHA_file_i in self.PHA_files_list
        ]
        self.times = pd.to_datetime(times)
        df = pd.DataFrame(self.par_vals, index=self.times)
        # Lets rearrange the columns for to keep Chi and reduced Chi at last
        old_cols = list(df.columns)
        old_cols.remove("Chi")
        old_cols.remove("red_Chi")
        new_cols = old_cols + ["Chi", "red_Chi"]
        self.df = df[new_cols]
        df = self.df
        df.to_csv(f"{out_dir}/results.csv")
        df.to_hdf(f"{out_dir}/results.h5", "results")
        return df


class chisoth_2T(chisoth):
    """class for running 2T isothermal models"""

    colum_names = None
    arf_files_list = None
    m = None
    fit_elements_list = []

    def __init__(
        self,
        PHA_files_list,
        arf_files_list,
        output_dir,
        FIP_elements,
        other_pars=["logT", "norm"],
        xcm_file="2T.xcm",
    ):
        """
        Initialises few things for xspec

        Parameters
        ----------
        PHA_files : list of PHA files to perform fit on
        arf_files : arf files corresponding to the PHA_files
        flare_dir : directory to save results to
        FIP_elements : list of elements, these elements will be alway included in fitting.
        other_pars : non-element fit pars.
        xcm_file : str, path to xcm file defing the model.
        """
        super().__init__(
            PHA_files_list,
            arf_files_list,
            output_dir,
            FIP_elements,
            other_pars,
            xcm_file,
        )

        # Figure out parameter index for other_pars which doesnot change during fitting
        self.other_pars_idx = []
        xp.Xset.restore(self.xcm_file)
        self.m = xp.AllModels(1, "flare")
        for component_i in self.m.componentNames:
            for other_par_i in self.other_pars:
                idx_temp = eval(f"self.m.{component_i}.{other_par_i}.index")
                self.other_pars_idx.append(idx_temp)

    def fit(
        self,
        min_E,
        max_E=15.0,
        output_dir=None,
        cutoff_cps=1.0,
        do_dynamic_elements=False,
        do_error_calculation=True,
        max_red_chi=100.0,
        sigma=1.0,
        element_line_dict={"S": 2.45, "Ar": 3.2, "Ca": 3.9, "Fe": 6.5},
    ):
        """
        fits the data with models.

        Parameters
        ----------
        min_E : minimum energy cut off for fitting
        max_E : maxmimum energy cut off for fitting
        cutoff_cps : minimum number of counts to be considered to be inclued in fit.
        do_dynamic_elements : bool, will dynamically add more elements to fit depending on counts.

        do_error_calculation: bool, whether to do errorr calculation or not.
        max_red_chi : float,  maximum allowed red_chi square value for error calculations.
        sigma : float, sigma for error calculation
        element_line_dict : a dictionary with keys as elements and values line energies in keV. Used to dynamically add elements to fit.
        refer to find_fit_elements documentation for more information.
        Returns
        -------
        df, pandas dataframe with results
        """
        self.max_red_chi = max_red_chi
        self.sigma = sigma
        suffix = ["", "_2"]  # TODO: Make this more general
        file_names = self.prefit(output_dir)
        #%% Fit
        # Iterate through th files
        par_vals = self.do_fit(
            file_names=file_names,
            out_dir=f"{self.output_dir}/fit",
            min_E=min_E,
            max_E=max_E,
            cutoff_cps=cutoff_cps,
            do_dynamic_elements=do_dynamic_elements,
            do_error_calculation=do_error_calculation,
            element_line_dict=element_line_dict,
            suffix=suffix,
        )
        #%% Make a data frame

        df = self.save_fit(out_dir=f"{self.output_dir}/fit")

        return df


class chisoth_2T_multi(chisoth_2T):

    """a 2T model for multi instruments."""

    def __init__(
        self,
        PHA_files_list,
        arf_files_list,
        output_dir,
        FIP_elements,
        other_pars=["logT", "norm", "factor"],
        xcm_file="2T_multi.xcm",
    ):
        """TODO: to be defined.

        Parameters
        ----------
        PHA_files : list of PHA files to perform fit on
        arf_files : arf files corresponding to the PHA_files
        flare_dir : directory to save results to
        FIP_elements : list of elements, these elements will be alway included in fitting.
        other_pars : non-element fit pars.
        xcm_file : str, path to xcm file defing the model.

        """
        chisoth.__init__(
            PHA_files_list,
            arf_files_list,
            output_dir,
            FIP_elements,
            other_pars,
            xcm_file,
        )
        self.other_pars_idx = []
        xp.Xset.restore(self.xcm_file)
        self.m = xp.AllModels(1, "flare")
        for component_i in self.m.componentNames:
            for other_par_i in self.other_pars[:-1]:
                idx_temp = eval(f"self.m.{component_i}.{other_par_i}.index")
                self.other_pars_idx.append(idx_temp)
        self.other_pars_idx.append(xp.AllModels(2, "flare").constant.factor.index)
