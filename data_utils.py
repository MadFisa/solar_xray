"""
File: data_utils.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: A module conatining utilities requied for daxss and xsm data processing.
"""

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from astropy.io import fits
from netCDF4 import Dataset

# A function to parse XSM MET
xsm_time_parser = lambda col: pd.to_datetime(col, origin="2017/01/01", unit="s")


def read_minxss_data(file_name):
    """
    Reads minxss data from a file and creates xarray.

    Input
    --------
    file_name : data_file

    Output
    ----------
    da_irrad : radiance xarray
    da_irrad_err : radiance error xarray
    """
    minxss_data = Dataset(file_name, "r", format="NETCDF4")
    irradinace = minxss_data.variables["X123.IRRADIANCE"]
    energy = minxss_data.variables["X123.ENERGY"]
    raw_dates = minxss_data.variables["X123_TIME.HUMAN"][0, :, :19]
    dates = [d.tobytes().decode("utf-8") for d in raw_dates.data]
    dates = pd.to_datetime(dates)
    da_irrad = xr.DataArray(
        irradinace,
        dims=("time", "energy"),
        coords={"time": dates, "energy": energy[0, 0]},
    )
    da_irrad_err = xr.DataArray(
        minxss_data.variables["X123.IRRADIANCE_UNCERTAINTY"][0],
        dims=("time", "energy"),
        coords={"time": dates, "energy": energy[0, 0]},
    )
    return da_irrad, da_irrad_err


def read_xsm_data(pha_file):
    """
    Reads Chandrayaan 2 averaged spectra  xsm data from a file and creates xarray dataset.

    Input
    --------
    file_name : data_file

    Output
    ----------
    xsm_data : xarray dataset
    """
    # data_origin_time = "2017-01-01"
    with fits.open(pha_file) as pha_reader:
        pha_data = pha_reader[1].data
        rmf_file = pha_reader[1].header["RESPFILE"]
        arf_file = pha_reader[1].header["ANCRFILE"]
    counts = pha_data["COUNTS"]
    channels = pha_data["CHANNEL"][0]
    stat_err = pha_data["STAT_ERR"]
    sys_err = pha_data["SYS_ERR"]
    exposure = pha_data["EXPOSURE"]
    t_start = pha_data["TSTART"]
    t_stop = pha_data["TSTOP"]
    t_mid = (t_start + t_stop) / 2

    time_start = xsm_time_parser(t_start)
    time_stop = xsm_time_parser(t_stop)
    time_mid = xsm_time_parser(t_mid)
    xsm_data = xr.Dataset(
        data_vars={
            "cps": (("time", "channel"), counts),
            "stat_err": (("time", "channel"), stat_err),
            "sys_err": (("time", "channel"), sys_err),
            "exposure": (("time"), exposure),
        },
        coords={
            "channel": channels,
            "time": time_mid,
            "time_start": (("time"), time_start),
            "time_stop": (("time"), time_stop),
        },
    )

    # cps = xsm_data["counts"] / xsm_data["exposure"]

    # xsm_data["cps"] = cps
    xsm_data.attrs["rmf_file"] = rmf_file
    xsm_data.attrs["arf_file"] = arf_file

    return xsm_data


def create_xarray(filename):
    """
    Converts data in from minxss to xarray.

    Input
    ----------
    filename : path to the netcdf4 file

    Output
    ---------
    ds : A dataset with irradiance and corresponding error
    """
    minxss_data = xr.open_dataset(filename)
    irradiance = minxss_data["X123.IRRADIANCE"]
    irradiance_err = minxss_data["X123.IRRADIANCE_UNCERTAINTY"]
    energy = minxss_data["X123.ENERGY"]
    raw_dates = minxss_data["X123_TIME.HUMAN"]
    dates = [d.tobytes().decode("UTF-8") for d in raw_dates.data[0]]
    dates = pd.to_datetime(dates)
    irradiance_modified = irradiance.squeeze("structure_elements")
    irradiance_err_modified = irradiance_err.squeeze("structure_elements")
    ds = xr.Dataset(
        {
            "irradiance": (["time", "energy"], irradiance_modified.values),
            "irradiance_err": (["time", "energy"], irradiance_err_modified.values),
        },
        coords={"time": dates, "energy": energy.values[0, 0, :]},
        attrs=irradiance.attrs,
    )
    return ds


def create_daxss_pha(
    data_array,
    arf_path="minxss_fm3_ARF.fits",
    rmf_path="minxss_fm3_RMF.fits",
    out_dir="./",
    bin_size=None
):
    """
    Creates PHA files for datarray passed of DAXSS data.
    Expects the data array to be sliced version of that returned by read_daxss_data() to the required time.
    Parameters
    ----------
    data_array : Datarray, similar to the one returned by read_daxss_data.
    out_dir : directory to output the PHA files to, optional
    arf_path : path to arf file
    rmf_path : path to rmf file
    bin_size : string, bin size to bin, expected in format for pd.resample.

    Returns
    -------


    """
    hdr_dummy = fits.Header()
    hdr_data = fits.Header()
    hdr_dummy["MISSION"] = "InspireSat-1"
    hdr_dummy["TELESCOP"] = "InspireSat-1"
    hdr_dummy["INSTRUME"] = "DAXSS"
    hdr_dummy["ORIGIN"] = "LASP"
    hdr_dummy["CREATOR"] = "DAXSSPlotterUtility_v1"
    hdr_dummy["CONTENT"] = "Type-I PHA file"

    # Data Header
    hdr_data["MISSION"] = "InspireSat-1"
    hdr_data["TELESCOP"] = "InspireSat-1"
    hdr_data["INSTRUME"] = "DAXSS"
    hdr_data["ORIGIN"] = "LASP"
    hdr_data["CREATOR"] = "DAXSSPlotterUtility_v1"
    hdr_data["CONTENT"] = "SPECTRUM"
    hdr_data["HDUCLASS"] = "OGIP"
    hdr_data["LONGSTRN"] = "OGIP 1.0"
    hdr_data["HDUCLAS1"] = "SPECTRUM"
    hdr_data["HDUVERS1"] = "1.2.1"
    hdr_data["HDUVERS"] = "1.2.1"

    hdr_data["AREASCAL"] = "1"
    hdr_data["BACKSCAL"] = "1"
    hdr_data["CORRSCAL"] = "1"
    hdr_data["BACKFILE"] = "none"

    hdr_data["CHANTYPE"] = "PHA"
    hdr_data["POISSERR"] = "F"

    hdr_data["CORRFILE"] = "none"
    hdr_data["EXTNAME"] = "SPECTRUM"
    hdr_data["FILTER"] = "Be/Kapton"
    hdr_data["EXPOSURE"] = "9"
    hdr_data["DETCHANS"] = "1000"
    hdr_data["GROUPING"] = "0"
    # CALDB = "/home/DAXSS_AED/Chianti_codes/xsm/caldb"
    channel_number_array = np.arange(1, 1001, dtype=np.int32)
    hdr_data["RESPFILE"] = rmf_path
    hdr_data["ANCRFILE"] = arf_path
    daxss_data_selected = data_array.isel(energy=slice(6, 1006))
    channel_number_array = np.arange(1, 1001, dtype=np.int32)
    if bin_size is not None:
        resampled = daxss_data_selected["cps"].resample(time=bin_size)
        counts = resampled.sum()
        statistical_error_array = (
            daxss_data_selected["cps_precision"]
            .resample(time=bin_size)
            .apply(calc_shot_noise)
        )
        cps_accuracy = (
            daxss_data_selected["cps_accuracy"].resample(time=bin_size).mean()
        )  # This is probbly wrong. Need to figure out.
    else:
        counts = daxss_data_selected["cps"]
        statistical_error_array = daxss_data_selected["cps_precision"]
        cps_accuracy = daxss_data_selected["cps_accuracy"]
    systematic_error_array = cps_accuracy / counts

    # Creating and Storing the FITS File
    time_ISO_array = counts.time
    #%% Create the array
    c1 = channel_number_array
    for i, time in enumerate(time_ISO_array):
        c2 = counts.isel(time=i)
        c3 = statistical_error_array.isel(time=i)
        c4 = systematic_error_array.isel(time=i)  # Accuracy = Systematic Error.
        c4[np.isnan(c4)] = 0
        file_name = f"DAXSS_{np.datetime_as_string(time.data)}.pha"
        hdr_dummy["FILENAME"] = file_name
        hdr_dummy["DATE"] = np.datetime_as_string(time.data)
        hdr_data["FILENAME"] = hdr_dummy["FILENAME"]
        hdr_data["DATE"] = hdr_dummy["DATE"]
        # Data
        hdu_data = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="CHANNEL", format="J", array=c1),
                fits.Column(name="RATE", format="E", array=c2),
                fits.Column(name="STAT_ERR", format="E", array=c3),
                fits.Column(name="SYS_ERR", format="E", array=c4),
            ],
            header=hdr_data,
        )
        dummy_primary = fits.PrimaryHDU(header=hdr_dummy)
        hdul = fits.HDUList([dummy_primary, hdu_data])
        filename_fits = f"{out_dir}/{file_name}"
        hdul.writeto(filename_fits, overwrite=True)
    if bin_size is not None:
        return resampled
    else:
        return


def read_tables(table_dir, tables_list=None):
    """
    Creates a xarray dataset with all the elements loaded. Reads all the tables
    in the directory if table list is not provided.

    Parameters
    ----------
    table_dir : Path to directory containing lookup tables.

    table_list: List of tables to read

    Returns
    -------
    ds : A dataset conatining the elements as datarrays.

    """
    if tables_list is None:
        tables_list = os.listdir(table_dir)
    ds = xr.open_dataset(f"{table_dir}/{tables_list[0]}")
    ds = ds.rename({"Intensity": tables_list[0].split(".nc")[0]})
    for element in tables_list[1:]:
        print(f"Read table for {element}")
        save_path = f"{table_dir}/{element}"
        el_da = xr.open_dataset(save_path).rename(
            {"Intensity": element.split(".nc")[0]}
        )
        ds = xr.merge([ds, el_da], combine_attrs="drop_conflicts")

    return ds


def read_daxss_data(file_name):
    """
    Reads minxx data from a file and creates xarray.

    Input
    --------
    file_name : data_file

    Output
    ----------
    da_irrad : radiance xarray
    da_irrad_err : radiance error xarray
    """
    daxss_data = Dataset(file_name, "r", format="NETCDF4")
    irradiance = daxss_data.variables["IRRADIANCE"]
    energy = daxss_data.variables["ENERGY"]
    raw_dates = daxss_data.variables["TIME_ISO"][:, :19]
    dates = [d.tobytes().decode("utf-8") for d in raw_dates.data]
    dates = pd.to_datetime(dates)
    irradiance_err = daxss_data.variables["IRRADIANCE_UNCERTAINTY"]
    cps = daxss_data.variables["SPECTRUM_CPS"]
    cps_err = daxss_data.variables["SPECTRUM_CPS_ACCURACY"]
    cps_precision = daxss_data.variables["SPECTRUM_CPS_PRECISION"]

    ds = xr.Dataset(
        data_vars={
            "cps": (("time", "energy"), cps),
            "cps_accuracy": (("time", "energy"), cps_err),
            "cps_precision": (("time", "energy"), cps_precision),
            "irradiance": (("time", "energy"), irradiance),
            "irradiance_uncert": (("time", "energy"), irradiance_err),
        },
        coords={"time": dates, "energy": energy[0]},
    )
    return ds


def create_rmf_dataarray(
    rmf_path, matrix_col="MATRIX", ebound_col="EBOUNDS", no_of_channels=None
):
    """Creates rmf file datarray with dimension as 'energy_bins' and 'channels'.
        'energy_bins' stands for energy bin number of incoming photons and
        'channel' corresponds to the detected channel

    Parameters
    ----------
    rmf_path : string, path to rmf_file

    Returns
    -------
    resp_mat : DataArray, Dimensions as 'energy_bins' and 'channels'
    bin_number : DataArray, Dimension as 'energy'

    """
    with fits.open(rmf_path) as reader:
        rmf_file = reader[matrix_col].data
        rmf_channel = reader[ebound_col].data
        if no_of_channels is None:
            no_of_channels = reader[matrix_col].header["DETCHANS"]

    # Create Response Matrix with dimension as energy bins and Channel numbers
    channel_number = np.arange(1, no_of_channels + 1)
    channel_energy_mid = (rmf_channel["E_MIN"] + rmf_channel["E_MAX"]) / 2

    energy_bin_mid = (rmf_file["ENERG_LO"] + rmf_file["ENERG_HI"]) / 2
    rmf_matrix = np.array(list(rmf_file["MATRIX"]))

    resp_mat = xr.DataArray(
        rmf_matrix,
        dims=["energy_bins", "channel"],
        coords={
            "energy_bins": np.arange(len(energy_bin_mid)),
            "energy_bin_mid": (("energy_bins"), energy_bin_mid),
            "energy_bin_lo": (("energy_bins"), rmf_file["ENERG_LO"]),
            "energy_bin_hi": (("energy_bins"), rmf_file["ENERG_HI"]),
            "channel": channel_number,
            "channel_energy_mid": (("channel"), channel_energy_mid),
            "channel_energy_lo": (("channel"), rmf_channel["E_MIN"]),
            "channel_energy_hi": (("channel"), rmf_channel["E_MAX"]),
        },
    )
    return resp_mat


def create_arf_dataarray(arf_path):
    """Creates arf file with dimension 'energy_bin' corresponding to a energy bin number

    Parameters
    ----------
    arf_file : string, Path to arf fits file
    -------
    eff_area : Datarray, contains spectral response with one dimension as 'energy_bins'

    """
    with fits.open(arf_path) as reader:
        arf_file = reader[1].data
    energy_bin_mid = (arf_file["ENERG_LO"] + arf_file["ENERG_HI"]) / 2
    if arf_file.columns[0].name == "ARF_NUM":
        arf_num = arf_file["ARF_NUM"]
        eff_area = xr.DataArray(
            arf_file["SPECRESP"],
            dims=["arf_num", "energy_bins"],
            coords={
                "arf_num": (("arf_num"), arf_num),
                "energy_bins": (("energy_bins"), np.arange(energy_bin_mid.shape[1])),
                "energy_bin_mid": (("arf_num", "energy_bins"), energy_bin_mid),
                "energy_bin_lo": (("arf_num", "energy_bins"), arf_file["ENERG_LO"]),
                "energy_bin_hi": (("arf_num", "energy_bins"), arf_file["ENERG_HI"]),
            },
        )
    else:

        eff_area = xr.DataArray(
            arf_file["SPECRESP"],
            dims=["energy_bins"],
            coords={
                "energy_bins": np.arange(len(energy_bin_mid)),
                "energy_bin_mid": (("energy_bins"), energy_bin_mid),
                "energy_bin_lo": (("energy_bins"), arf_file["ENERG_LO"]),
                "energy_bin_hi": (("energy_bins"), arf_file["ENERG_HI"]),
            },
        )
    return eff_area


class instrument:

    """A class for isntrument properties. Initialise by giving paths to rmf and
    arf files."""

    def __init__(self, rmf_file_path=None, arf_file_path=None):
        """Give rmf and arf file paths to initialise

        Parameters
        ----------
        rmf_file_path : string, path to rmf file
        arf_file_path : string, path to arf file


        """
        self.rmf = None
        self.arf = None
        self.response = None
        if rmf_file_path is not None:
            self.rmf = create_rmf_dataarray(rmf_file_path)
        if arf_file_path is not None:
            self.arf = create_arf_dataarray(arf_file_path)
        self.create_response()

    def convolve_with_reponse(self, flux_model):
        """Convolves given model flux with response to obatain at instrument
        detected counts.

        Parameters
        ----------
        flux_model : DataArray, of flux with energy_bin as its coordinates

        Returns
        -------
        flux_model_convolved: DatArray of detected flux with coordinate as channel

        """
        flux_model_convolved = xr.dot(flux_model, self.response, dims=["energy_bins"])
        return flux_model_convolved

    def create_response(self):
        """Creates response matrix from given arf and rmf files
        Returns
        -------
        response : xr.DataArray, response Matrix

        """
        if self.arf is None and self.rmf is None:
            warnings.warn("Neither arf nor rmf is set, Response matrix will not be set")
            response = None
        elif self.arf is not None and self.rmf is not None:
            response = self.arf * self.rmf
        else:
            if self.arf is not None:
                response = self.arf
                warnings.warn(
                    "Only arf is set, arf will be considered as the sole response"
                )
            else:
                response = self.rmf
                warnings.warn(
                    "Only rmf file is set, rmf will be considered as the sole response"
                )
        self.response = response
        return response

    def read_rmf(
        self, rmf_path, matrix_col="MATRIX", ebound_col="EBOUNDS", no_of_channels=None
    ):
        """reads rmf file

        Parameters
        ----------
        rmf_file_path : string, path to rmf file

        Returns
        -------
        rmf: DataArray of rmf

        """
        self.rmf = create_rmf_dataarray(
            rmf_path,
            matrix_col=matrix_col,
            ebound_col=ebound_col,
            no_of_channels=no_of_channels,
        )
        return self.rmf

    def read_arf(self, arf_file_path):
        """reads arf file

        Parameters
        ----------
        rmf_file_path : string, path to rmf file

        Returns
        -------
        arf: DataArray of rmf

        """
        self.arf = create_arf_dataarray(arf_file_path)
        return self.arf

def calc_shot_noise(da):
    """
    Calculates shot noise for resample from input array of individual shot noises.

    Parameters
    ----------
    da : data array like

    Returns
    -------
    TODO

    """
    err2 = da**2
    return np.sqrt(err2.sum(dim="time"))

class DaXSS_instrument(instrument):

    """Class for taking care of DaXSS instrument"""

    def __init__(self, rmf_file_path=None, arf_file_path=None):
        """Give rmf and arf file paths to initialise

        Parameters
        ----------
        rmf_file_path : string, path to rmf file
        arf_file_path : string, path to arf file


        """
        self.rmf = None
        self.arf = None
        self.response = None
        self.rmf = self.read_rmf(
            rmf_file_path, matrix_col=1, ebound_col=2, no_of_channels=1000
        )
        self.arf = self.read_arf(arf_file_path)
        self.create_response()

    read_data = read_daxss_data  # Aliasing for reading data


class XSM_instrument(instrument):

    """Class for taking care of XSM instrument"""

    read_data = read_xsm_data  # Aliasing for reading data
    data_origin_time = "2017-01-01"

    def __init__(self, rmf_file_path=None, arf_file_path=None):
        """Give rmf and arf file paths to initialise

        Parameters
        ----------
        rmf_file_path : string, path to rmf file
        arf_file_path : string, path to arf file


        """
        instrument.__init__(self, rmf_file_path, arf_file_path)

    def met2UTC(self, met_time):
        """Converts Mission Elapse Time to UTC

        Parameters
        ----------
        met_time : float, MET in seconds.

        Returns
        -------
        UTC : pandas DateTime object

        """

        UTC = pd.to_datetime(met_time, origin=self.data_origin_time, unit="s")
        return UTC

    def load_pha(self, pha_path):
        """Loads a pha file and associated rmf and arf files.

        Parameters
        ----------
        pha_path : string, path of pha file

        Returns
        -------


        """
        with fits.open(pha_path) as pha_reader:
            pha_data = pha_reader[1].data
            rmf_file = pha_reader[1].header["RESPFILE"]
            arf_file = pha_reader[1].header["ANCRFILE"]
            if pha_reader["PRIMARY"].header["CONTENT"] == "Type II PHA file":
                t_start = pha_data["TSTART"]
                t_stop = pha_data["TSTOP"]
                TYPE_II = True
                t_mid = (t_start + t_stop) / 2
                time_start = pd.to_datetime(
                    t_start, origin=self.data_origin_time, unit="s"
                )
                time_stop = pd.to_datetime(
                    t_stop, origin=self.data_origin_time, unit="s"
                )
                time_mid = pd.to_datetime(t_mid, origin=self.data_origin_time, unit="s")

            else:
                t_start_str = pha_reader[1].header["HISTORY"][9]
                t_stop_str = pha_reader[1].header["HISTORY"][10]
                t_start = float(t_start_str.strip("Tstart ="))
                t_stop = float(t_stop_str.strip("Tstop ="))
                TYPE_II = False

        counts = pha_data["COUNTS"]
        channels = pha_data["CHANNEL"]
        stat_err = pha_data["STAT_ERR"]
        sys_err = pha_data["SYS_ERR"]
        if TYPE_II:
            xsm_data = xr.Dataset(
                data_vars={
                    "cps": (("time", "channel"), counts),
                    "stat_err": (("time", "channel"), stat_err),
                    "sys_err": (("time", "channel"), sys_err),
                },
                coords={
                    "channel": (("channel"), channels[0]),
                    "time": (("time"), time_mid),
                    "time_start": (("time"), time_start),
                    "time_stop": (("time"), time_stop),
                },
            )
        else:

            xsm_data = xr.Dataset(
                data_vars={
                    "cps": (
                        ("channel"),
                        counts,
                    ),
                    "stat_err": (("channel"), stat_err),
                    "sys_err": (("channel"), sys_err),
                },
                coords={
                    "channel": channels,
                },
            )

            # cps = xsm_data["counts"] / xsm_data["exposure"]

            # xsm_data["cps"] = cps

            xsm_data.attrs["t_start"] = self.met2UTC(t_start)
            xsm_data.attrs["t_stop"] = self.met2UTC(t_stop)
        xsm_data.attrs["rmf_file"] = rmf_file
        xsm_data.attrs["arf_file"] = arf_file
        self.spectra = xsm_data
        working_dir = os.path.dirname(os.path.abspath(pha_path))

        self.read_rmf(f"{self.spectra.attrs['rmf_file']}")
        if self.spectra.attrs["arf_file"] != "":
            self.read_arf(f"{working_dir}/{self.spectra.attrs['arf_file']}")
        else:
            warnings.warn("No arf file specified in pha file")
        self.create_response()
        self.spectra = self.spectra.assign_coords(
            {
                "channel_energy_mid": (("channel"), self.rmf.channel_energy_mid.data),
                "channel_energy_lo": (("channel"), self.rmf.channel_energy_lo.data),
                "channel_energy_hi": (("channel"), self.rmf.channel_energy_hi.data),
            }
        )
        if TYPE_II:
            self.PHA_type = 2
        else:
            self.PHA_type = 1

    def convolve_with_reponse(self, flux_model, arf_number):
        """Convolves given model flux with response to obatain at instrument
        detected counts.

        Parameters
        ----------
        flux_model : DataArray, of flux with energy_bin as its coordinates

        Returns
        -------
        flux_model_convolved: DatArray of detected flux with coordinate as channel

        """
        flux_model_convolved = xr.dot(
            flux_model, self.response.sel(arf_num=arf_number), dims=["energy_bins"]
        )
        return flux_model_convolved
