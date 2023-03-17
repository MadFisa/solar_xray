"""
File: instruments.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: Module containing fitiing codes for instruments
"""
from fit_utils import do_grppha
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

        """
        self.name = "instrument"
        self.PHA_file_list = PHA_file_list
        self.arf_file_list = arf_file_list
        self.output_dir = output_dir
        
    def create_pha_files(self,time_beg,time_end,bin_size):
        """
        Function that should create time integrated pha files for a given begin and end time.

        Parameters
        ----------
        time_beg : begining of required time
        time_end : end of required time
        bin_size : bin_size, are required by pd.resamp, eg 27 second = '27S'
        output_dir : directory to output
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
        


