#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: models.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: A module containing codes for making and running isotheraml plasma
models based on ChiantiPy.
"""

import os
from string import Template

import ChiantiPy.core as ch
import numpy as np
import xarray as xr

from data_utils import read_tables

h = 6.627e-34  # planks constant
c = 299792458  # velocity of light
conversion_factor = 12.398  # wvl in Armstrong to energy in keV

# try:
# if os.environ['XUVTOP']:
# chianti_dir = os.environ['XUVTOP']
# print(f"Chianti directory is {chianti_dir}")
# except KeyError:
# print("Chianti environment variable not set")

elements_list = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pg",
    "Ag",
    "Cd",
    "In",
    "Sn",
]

sun_radius = 695700e5  # cm
sun_distance = 149597870.7e5  # cm
Sr = np.pi * sun_radius**2 / sun_distance**2


def create_tables(
    elements_list,
    logT_list,
    energies,
    density=1e9,
    save_dir="./tables",
    kargs={"proc": 4},
):
    """Creates a lookup table for the elements at unit abundance for given
    temperature, energies and density.

    Parameters
    ----------
    elements_list : list, of elements
    logT_list : Array, of base 10 log of temperatures in Kelvin.
    energies : Array, of energies in keV
    density :float, density in cm^-3, optional
    save_dir :string, directory for saving the files, optional
    kargs :dictionary, keyword arguments to be passed to Chianti.mspectrum  as
    key value pairs, for example {procs:8}

    Returns
    -------
    TODO

    """
    wvl = conversion_factor / energies  # Converts from keV to Armstrongs
    if not os.path.exists(save_dir):
        print(f"folder {save_dir} does not exist, creating the folder")
        os.mkdir(save_dir)
    for element in elements_list:
        save_path = f"{save_dir}/{element}.nc"
        if os.path.exists(save_path):
            print(f"===== table {save_path} already exist, skipping the element =====")
            # s = ch.spectrum(temp, dens, wvl, verbose=1,abundance='unity',elementList=[element])
        else:
            print(f"===== creating table {save_path} =====")
            s = ch.mspectrum(
                np.power(10, logT_list),
                density,
                wvl,
                verbose=1,
                abundance="unity",
                elementList=[element],
                **kargs,
            )
            energy_model, flux_model = convert_to_photons_keV(
                s.Spectrum["wavelength"], s.Spectrum["intensity"]
            )
            da = xr.DataArray(
                flux_model,
                coords={"temperature": logT_list, "energy": energy_model},
                name=f"Intensity",
            )
            da.attrs[
                "Description"
            ] = f"Intensity for all emission from {element} from an isothermal model as calculated by ChianiPy"
            da.attrs["Element"] = element
            da.attrs["Electron Density"] = f"{density} cm^-3"
            da.attrs["Energy"] = "KeV units"
            da.attrs["Temperature"] = "log_10 of temperature in Kelvin"
            da.attrs["Flux"] = "photons/Kev cm^-2 per em(cm^-5)"
            da.to_netcdf(save_path)


def create_abund_file(abund_dict, file_name):
    """
    creates abundance file in the specified directory and returns contents as string

    input
    --------------
    abund_dict : dictionary, of abundances with keys as elemental names (i.e Fe,H etc)
    file_name : string, file_name to save

    output
    --------------
    new_abund_file : string of abundance file contents
    """
    with open("template.abund", "r") as fh:
        abundance_file_template = Template(fh.read())
    abund_file_subst = abundance_file_template.safe_substitute(abund_dict)
    new_abund_file = []
    for line in abund_file_subst.splitlines():
        if not "$" in line:
            new_abund_file.append(line + "\n")
    with open(file_name, "w") as fh:
        fh.writelines(new_abund_file)
    return new_abund_file


def convert_to_photons_keV(wvl, intensity):
    """
    Takes in a spectrum in ergs/s and wavelength in Armstrong, converts it into
    photons/keV and KeV.

    Input
    --------------
    wvl : Wavelength in Armstrong
    intensity : Flux in erg/s/Armstrong

    Output
    --------------
    energy_keV : energy of wavlength in keV
    photons_keV : flux in photon/keV

    """
    energy_keV = conversion_factor / wvl
    flux_erg_keV = intensity / conversion_factor * np.power(wvl, 2)
    photon_energy_erg = h * c / (wvl * 1e-10) * 1e7
    flux_photons_keV = flux_erg_keV / photon_energy_erg
    return energy_keV, flux_photons_keV


class singTplasma_model:
    def __init__(
        self, tables_dir, E_lo=None, E_hi=None, T_lo=None, T_hi=None, tables_list=None
    ):
        data_table = read_tables(tables_dir, tables_list)
        self.lookup_table = data_table.sel(
            temperature=slice(T_lo, T_hi), energy=slice(E_lo, E_hi)
        ).to_array(dim="element")
        self.logT_max = self.lookup_table.temperature.data[-1]
        self.logT_min = self.lookup_table.temperature.data[0]
        self.E_max = self.lookup_table.energy.data[-1]
        self.E_min = self.lookup_table.energy.data[0]

    def calculate_spectrum(self, logT, em, abundance):
        """Calcualated spectrum for a list of values of logT,em and elemental
        abundances for energies between E_lo and E_hi. Gives the flux a t earth in phtons cm^-2 keV^-1.

        Parameters
        ----------
        logT : array, of log temperatures.
        em : array, of emission measures (in cm^-5) of same length as logT
        abdundances : array, of elemental abundance. Should have the shape
        (len(logT),30) where each row contain abundance for atomic numbers 1 to 30.
        E_lo : lower limit for energies to calculte in keV, optional
        E_hi : upper limit for energies to calculte in keV, optional

        Returns
        -------
        flux_model: DataArray with calculated fluxes

        """

        self.params = xr.Dataset(
            data_vars={
                "emissivity": (("temperature"), em),
                "abundances": (("temperature", "element"), abundance),
            },
            coords={"temperature": logT, "element": elements_list[:30]},
        )
        interpolated = (
            self.lookup_table.interp(
                temperature=logT,
                method="linear",
                assume_sorted=True,
                kwargs={"copy": False},
            )
            * Sr
            * self.params["emissivity"]
        )
        self.flux_model = xr.dot(
            interpolated, self.params["abundances"], dims=["element"]
        )
        return self.flux_model
