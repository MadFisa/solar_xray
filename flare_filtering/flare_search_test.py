#!/usr/bin/env python3
"""
File: goes_flare.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: A sample script to try sifting through Goes flare data.
"""
from sunpy.net import Fido
from sunpy.net import attrs as a

event_type = "FL"
# tstart = "2013/10/28"
# tend = "2013/10/29"
tstart = "2022/02/14"
tend = "2022/11/30"
result = Fido.search(a.Time(tstart, tend),
                     a.hek.EventType(event_type),
                     a.hek.FL.GOESCls > "B1.0",
                     a.hek.OBS.Observatory == 'GOES')

# %% Show the results
print(result.show("hpc_bbox", "refs"))

hek_results = result["hek"]
new_table = hek_results["event_starttime", "event_peaktime",
                        "event_endtime", "fl_goescls", "ar_noaanum"]

# %% Save results
# hek_results.write("FebToNov_M1_flares.h5", format="hdf5")
# hek_results.write("FebToNov_M1_flares.fits", format="fits")
new_table.write("FebToNov_M1_flares_slimmed.h5", format="hdf5")
new_table.write("FebToNov_M1_flares_slimmed.csv", format="csv")
