import glob
import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_utils import create_daxss_pha, read_daxss_data
from fit_utils import chisoth_2T, do_grppha

plt.style.use("fivethirtyeight")
DAXSS_file = "./data/daxss_solarSXR_level1_2022-02-14-mission_v2.0.0.ncdf"
observation_file = "./flare_filtering/data/flare_observation.h5"
# bin_size = "27S"


xsm_origin_time = datetime(2017, 1, 1)
utc2met = lambda time: (time - xsm_origin_time).total_seconds()
daxss_data = read_daxss_data(DAXSS_file)
observation_table = pd.read_hdf(observation_file, "obs")
#%% Create PHA files
def create_pha_files(flare_num, bin_size=None):
    """
    creates pha files from data and flare_number
    Parameters
    ----------
    flare_num : int, flare_number(index) in the observation_table.

    Returns
    -------
    TODO

    """
    flare_dir = f"./data/pha/flare_num_{flare_num}/simult"
    flare = observation_table.loc[flare_num]
    time_beg = flare["event_starttime"]
    time_end = flare["event_endtime"]

    daxss_flare = daxss_data.sel(time=slice(time_beg, time_end))
    arf_path = "./data/minxss_fm3_ARF.fits"
    rmf_path = "./data/minxss_fm3_RMF.fits"
    out_dir = f"{flare_dir}/orig_pha"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    pp = create_daxss_pha(
        daxss_flare,
        out_dir=out_dir,
        arf_path=arf_path,
        rmf_path=rmf_path,
        bin_size=bin_size,
    )
    for resampled_i in pp:
        time_i = pd.Timestamp(resampled_i[0])
        time_beg = pd.Timestamp(resampled_i[1].time.isel(time=0).to_pandas())
        time_end = pd.Timestamp(resampled_i[1].time.isel(time=-1).to_pandas())
        met_i_beg = utc2met(time_beg)
        met_i_end = utc2met(time_end)
        year = time_i.year
        month = time_i.month
        day = time_i.day

        root_dir = f"./data/xsm/data/{year}/{str(month).zfill(2)}/{str(day).zfill(2)}"
        file_basename = f'ch2_xsm_{time_i.strftime("%Y%m%d")}_v1_level'
        l1file = f"{root_dir}/raw/{file_basename}1.fits"
        hkfile = f"{root_dir}/raw/{file_basename}1.hk"
        safile = f"{root_dir}/raw/{file_basename}1.sa"
        gtifile = f"{root_dir}/calibrated/{file_basename}2.gti"
        outfile = (
            f"{flare_dir}/orig_pha/XSM_{np.datetime_as_string(time_i.to_numpy())}.pha"
        )

        command = f"xsmgenspec l1file={l1file} specfile={outfile} spectype='time-integrated' hkfile={hkfile} safile={safile} gtifile={gtifile} tstart={met_i_beg} tstop={met_i_end} "
        os.system(command)


def fit_simult(
    flare_num,
    FIP_elements=["Mg", "Si", "S", "Ar", "Ca", "Fe"],
    CREATE=False,
    BIN=False,
    cutoff_cps=1,
    min_E=1.3,
    max_E=10.0,
    bin_size="27S"
):
    """
    function to daxss fitting

    Parameters
    ----------
    flare_num : int, index of flare in the list.
    FIP_elements : list of elements to be considered as FIP.
    CREATE : bool, Whether to create_pha_files.
    BIN : bool, Whetther to bin the pha files using grbpha
    cutoff_cps: float, threshold cps to bin i.e the grppha
    min_E : float, minimum energy of spectra to start fit from
    max_E : float, optional, maximum energy to fit from

    Returns
    -------
    Datframe with results

    """
    flare_dir = f"./data/pha/flare_num_{flare_num}/simult"
    if not os.path.isdir(f"{flare_dir}/orig_pha/"):
        CREATE = True
    if CREATE:
        shutil.rmtree(f"{flare_dir}/orig_pha/", ignore_errors=True)
        resamp = create_pha_files(flare_num, bin_size=bin_size)

    xsm_PHA_file_list = glob.glob(f"{flare_dir}/orig_pha/XSM*.pha")
    daxss_PHA_file_list = [orig_i.replace("XSM_", "DAXSS_") for orig_i in xsm_PHA_file_list]

    orig_PHA_file_list = xsm_PHA_file_list + daxss_PHA_file_list
    orig_PHA_file_list.sort()
    grpd_PHA_file_list = [
        orig_i.replace("/orig_pha/", "/grpd_pha/") for orig_i in orig_PHA_file_list
    ]

    if not os.path.isdir(f"{flare_dir}/grpd_pha/"):
        BIN = True
    if BIN:
        shutil.rmtree(f"{flare_dir}/grpd_pha/", ignore_errors=True)
        os.makedirs(f"{flare_dir}/grpd_pha")
        do_grppha(orig_PHA_file_list, grpd_PHA_file_list, cutoff_cps)
        # os.system(f"cp {flare_dir}/orig_pha/*.arf {flare_dir}/grpd_pha/")


    PHA_file_list = []
    arf_file_list = []
    for daxss_orig_i, xsm_orig_i in zip(daxss_PHA_file_list, xsm_PHA_file_list):
        temp = [
            daxss_orig_i.replace("/orig_pha/", "/grpd_pha/"),
            xsm_orig_i.replace("/orig_pha/", "/grpd_pha/"),
        ]
        PHA_file_list.append(temp)
    for xsm_PHA_file_i in xsm_PHA_file_list:
        xsm_arf_file = xsm_PHA_file_i.removesuffix(".pha") + (".arf")
        arf_file_list.append(["USE_DEFAULT", xsm_arf_file])
    chiso = chisoth_2T(PHA_file_list, arf_file_list, flare_dir)
    FIP_elements = ["Mg", "Si", "S", "Ar", "Fe"]
    chiso.init_chisoth(FIP_elements, error_sigma=6.00)
    df = chiso.fit(min_E, max_E)
    return df
