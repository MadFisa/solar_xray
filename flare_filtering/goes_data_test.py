#!/usr/bin/env python3
"""
File: goes_data_test.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: A test code to grab GOES flare lightcurves.
"""
import matplotlib.pyplot as plt
from astropy.visualization import time_support
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy import time as tm
from sunpy import timeseries as ts
import numpy as np
# tr = a.Time('2011-06-07 04:00', '2011-06-07 12:00')
results = Fido.search(
    a.Time("2020-11-20 00:00", "2020-11-21 23:00"), a.Instrument("XRS") | a.hek.EventType('FL'))

# %% Fetch the data
goes_result = results['xrs']
file_goes = Fido.fetch(goes_result)

# %% Load the data and take a look
goes_16 = ts.TimeSeries(file_goes[3])
plt.figure()
goes_16.peek()
# %% Convert to data frame and filter for good data
df = goes_16.to_dataframe()
df_filtered = df[(df["xrsa_quality"] == 0) & (df["xrsb_quality"] == 0)]
goes_16 = ts.TimeSeries(df, goes_16.meta, goes_16.units)

# %% Lets try plotting
goes_16.plot(columns=["xrsb"])
plt.show()

# %% Lets look at HEK data
hek_results = results['hek']
new_table = hek_results["event_starttime", "event_peaktime",
                        "event_endtime", "fl_goescls", "ar_noaanum"]
C_mask = new_table['fl_goescls'] > 'C1'
C_flare = new_table[C_mask]
flare_starttime = new_table['event_starttime'][0]
flare_endtime = new_table['event_endtime'][0]

# %% Lets truncate and find derivative
time_range = tm.TimeRange(flare_starttime, flare_endtime)
goes_flare = goes_16.truncate(time_range)
# goes_flare = goes_16.truncate(flare_starttime, flare_endtime)

time_support()  # Enable support for astropy.Time instances in matplotlib
fig, ax = plt.subplots()
ax.plot(goes_flare.time, np.gradient(goes_flare.quantity("xrsb")))
ax.set_ylabel("Flux (Wm$^{-2}$$s^{-1}$)")
fig.autofmt_xdate()
plt.show()
plt.figure()
goes_flare.plot(columns=["xrsb"])
fig.autofmt_xdate()
# %% Load the data and take a look
goes_17 = ts.TimeSeries(file_goes[1])
# goes_17.peek()
# %% Convert to data frame and filter for good data
df = goes_17.to_dataframe()
df_filtered = df[(df["xrsa_quality"] == 0) & (df["xrsb_quality"] == 0)]
goes_17 = ts.TimeSeries(df, goes_16.meta, goes_16.units)
# %% Lets plot both satelites data together
time_range = tm.TimeRange(flare_starttime, flare_endtime)
goes16_flare = goes_16.truncate(time_range)
goes17_flare = goes_17.truncate(time_range)
goes16_flare.plot(columns=["xrsb"])
goes17_flare.plot(columns=["xrsb"])
# goes_flare = goes_16.truncate(flare_starttime, flare_endtime)
