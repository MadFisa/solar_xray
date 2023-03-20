"""
File: instruments.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: Module containing fitiing codes for instruments
"""
from fit_utils import do_grppha
from data_utils import create_daxss_pha,read_daxss_data
import shutil
import os

class instrument:
    """Parent class for instruents"""
    def __init__(self,output_dir,PHA_file_list=None,arf_file_list=None,):
        """
        Parameters
        ----------
        output_dir : directory where all outputs will be saved.
        PHA_file_list : list of PHA files to be loaded.
        arf_file_list : list of arf files to use, 
                        use string 'USE_DEFAULT' as elements of list to use default arf file in PHA file.

        """
        self.name = "instrument"
        self.output_dir = output_dir
        self.set_pha_files(PHA_file_list, arf_file_list)

    def set_pha_files(self,PHA_file_list=None,arf_file_list=None):
        """
        Parameters
        ----------
        PHA_file_list : list of PHA files to be loaded.
        arf_file_list : list of arf files to use, 
                        use string 'USE_DEFAULT' as elements of list to use default arf file in PHA file.

        """
        self.PHA_file_list = PHA_file_list
        self.arf_file_list = arf_file_list
        
    def create_pha_files(self,time_beg,time_end,bin_size):
        """
        Function that should create time integrated pha files for a given begin and end time.

        Parameters
        ----------
        time_beg : begining of required time
        time_end : end of required time
        bin_size : bin_size, are required by pd.resamp, eg 27 second = '27S'
        output_dir : directory to output, files will be put in output/orig_pha.
        GROUP: Whethere to do grppha or not

        Returns
        -------
        TODO

        """
        self.time_beg = time_beg
        self.time_end = time_end
        self.bin_size = bin_size

    def do_grouping(self, min_count, PHA_file_list=None):
        """
        Do grppha on given files if any given, otherwise does it on self.PHA_file_list.
        Also replace self.PHA_Files with new grouped files

        Parameters
        ----------
        min_count : float, minimum number of counts in each bin
        PHA_file_list : list, of files to grppha on

        Returns
        -------
        file_names of new PHA_Files

        """
        if PHA_file_list is not None:
            self.PHA_file_list = PHA_file_list
        shutil.rmtree(f"{self.output_dir}/grpd_pha/", ignore_errors=True)
        out_PHA_file_list = [
            pha_file_i.replace("/orig_pha/", "/grpd_pha/") for pha_file_i in self.PHA_file_list
        ]
        os.makedirs(f"{self.output_dir}/grpd_pha")
        do_grppha(out_PHA_file_list, PHA_file_list, min_count)
        self.PHA_file_list = out_PHA_file_list
        return out_PHA_file_list

    def fit(self,model_class,model_args,fit_args):
        """
        Function to fit the given model. The model class should have two functions.
        One constructor, which will take a PHA_file_list,arf_file_list,output_dir.
        Second a fit function, which take kwargs as arguments

        Parameters
        ----------
         model_calss: Class name of the model to use.
         model_args : dictionary, dictionary of arguments to model constructor.
         fit_args : dictionary, dictionary of arguments to fit function

        Returns
        -------
         out_out from fit

        """
        self.model = model_class(**model_args)
        self.fit_out = self.model.fit(**fit_args)
        return self.fit_out
        


class DAXSS(instrument):

    """Class to handle InspireSat 1 DAXSS"""

    def __init__(self,output_dir,PHA_file_list=None,arf_file_list=None,):
        """TODO: to be defined. """
        instrument.__init__(self,output_dir,PHA_file_list=None,arf_file_list=None,)


    def load_data(self, DAXSS_file,arf_path,rmf_path):
        """
        Function to load daxss data

        Parameters
        ----------
        DAXSS_file : str, path to DAXSS netcdf file.
        arf_path : str, path to DAXSS arf file.
        rmf_path : str, path to DAXSS rmf file.

        Returns
        -------
        TODO

        """
        
        self.DAXSS_file = DAXSS_file
        self.daxss_data = read_daxss_data(self.DAXSS_file)
        self.arf_path = arf_path
        self.rmf_path = rmf_path
        
    def create_pha_files(self,time_beg,time_end,bin_size,out_dir):
        """
        Creates daxss pha files for given duration with given bin sizes.
        Need to run DAXSS.load_data and load data and rmf file before hand.

        Parameters
        ----------
        time_beg : beginning of time for which PHA files to be created.
        time_end : beginning of time for which PHA files to be created.
        bin_size : str,bin size for binning. Expected in form of pandas.resample. i.e 27 second = '27S'
        out_dir : directory to ouput file.

        Returns
        -------
        A list of pha file names.

        """
        super.create_pha_files(time_beg,time_end,bin_size)
        self.daxss_flare = self.daxss_data.sel(time=slice(time_beg, time_end))
        out_dir=out_dir+"/orig_pha"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        self.PHA_file_list = create_daxss_pha(
            self.daxss_flare,
            out_dir=out_dir,
            arf_path=self.arf_path,
            rmf_path=self.rmf_path,
            bin_size=self.bin_size,
        )
        self.set_pha_files(self.PHA_file_list,['USE_DEFAULT']*len(self.PHA_file_list))
        return self.PHA_file_list
