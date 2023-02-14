#!/usr/bin/env python3
"""
File: parse_xsm_lightcurve.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: A code to test parsing xsm light curve. This will combine all the 
light curves specified in file_list_path to singel file and write it out as netcdf.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from data_utils import xsm_time_parser

xsm_lc_path = "./data/2022"
xsm_lc_file_path = "ch2_xsm_flux_20220313.txt"
files_list_path = "./data/2022/files.txt"


with open(files_list_path) as fh:
    file_list = fh.readlines()

xsm_lc_df_list = []

for file in file_list:
    file = file.replace(".", xsm_lc_path, 1).strip()
    xsm_lc_df_i = pd.read_csv(
        # f"{xsm_lc_path}/{xsm_lc_file_path}",
        file,
        sep="\s+",
        # usecols=[0, 1, 2, 4],  # There is an extra space bewtween last two columns.
        names=["time", "flux (1-8A)", "flux (1-15keV)", "beWindowFlag"],
        index_col="time",
        parse_dates=True,
    )
    xsm_lc_df_i.index = xsm_time_parser(xsm_lc_df_i.index)
    xsm_lc_df_list.append(xsm_lc_df_i)

xsm_lc = pd.concat(xsm_lc_df_list)
xsm_xr = xr.Dataset.from_dataframe(xsm_lc)
xsm_xr["flux (1-8A)"].attrs["units"] = "W/m^2"
xsm_xr["flux (1-15keV)"].attrs["units"] = "W/m^2"
xsm_xr["beWindowFlag"].attrs[
    "Description"
] = "Berillium Window flag. Flux won't be correct whenever true"
op_nc_path = "./data/xsm_2022_lc.nc"
xsm_xr.to_netcdf(op_nc_path)
