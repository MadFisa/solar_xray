"""
File: DAXSS_pyspec_customModel_preflare.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: Code for fitting DAXSS data with xspec custom model
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xspec as xp
from data_utils import create_daxss_pha, read_daxss_data

keV_to_MK = 11604562.9141 / 1e6
dates_init = [
    "2022-03-15T23:20:00",
    "2022-03-31T18:00:00",
    "2022-05-05T13:32:00",
    "2022-03-22T20:20:00",
    "2022-05-03T08:00:00",
    "2022-03-22T11:00:00",
    "2022-03-07T22:40:00",
    "2022-03-22T01:20:00",
]
dates_stop = [
    "2022-03-15T23:30:00",
    "2022-03-31T23:00:00",
    "2022-05-05T13:50:00",
    "2022-03-22T20:30:00",
    "2022-05-03T08:38:00",
    "2022-03-22T11:13:00",
    "2022-03-07T22:55:00",
    "2022-03-22T01:40:00",
]

spec_num = 0
date_init = "2022-03-15T22:39:00"
date_stop = "2022-03-15T22:40:00"
DAXSS_file = "/home/sac/Asif/Chianti_codes/main/daxss_solarSXR_level1_2022-02-14-mission_v2.0.0.ncdf"
daxss_data = read_daxss_data(DAXSS_file)
time_sel = slice(date_init, date_stop)
temp_flare_data = daxss_data.sel(time=time_sel)

# flare_data = temp_flare_data

flare_data = temp_flare_data.mean(dim="time", keepdims=True)
flare_data = flare_data.assign_coords(time=[temp_flare_data.time.data[0]])

#%% Create PHA files
CREATE = True
PHA_dir = f"./PHA_Files/prl_model/flare_{dates_init[spec_num]}/preflare_prl"
arf_path = "/home/sac/Asif/Chianti_codes/main/minxss_fm3_ARF.fits"
rmf_path = "/home/sac/Asif/Chianti_codes/main/minxss_fm3_RMF.fits"

if CREATE:
    if not os.path.isdir(PHA_dir):
        os.makedirs(PHA_dir)
    create_daxss_pha(flare_data, out_dir=PHA_dir, arf_path=arf_path, rmf_path=rmf_path)

##% setup
# PHA_file = "minxss_fm3_PHA_2022-03-15T23-20-41Z.pha"
PHA_file_list = [
    f"{PHA_dir}/DAXSS_{np.datetime_as_string(time)}.pha"
    for time in flare_data.time.data
]

xp.AllModels.lmod("chspec",dirPath='/home/sac/chspec/')
PHA_file = PHA_file_list[0]
xp.AllData.clear()
xp.AllModels.clear()
# xp.Xset.abund = "file ./feldman_coronal_ext.txt"
xp.Fit.query = "no"  # No asking for uesr confirmation while fitting
xp.Fit.statMethod = "chi"
xp.Plot.xAxis = "keV"
# xp.Plot.xAxis = "channel"
xp.Plot.yLog = True
xp.Plot.xLog = False
table_dir = "/home/sac/Asif/Chianti_codes/main/chspec_CHIANTIv10.fits"

##% prepare Spectrum

xp.AllData.clear()
# freeze_dict = {i: ",-1,,,,," for i in range(1, 32)}
# freeze_dict = {i: ",-1,,,,," for i in range(1, 64)}
logFile = xp.Xset.openLog(f"{PHA_dir}/preflare.log")
s = xp.Spectrum(PHA_file)
s.ignore("**-0.7 3.5-**")
# m = xp.Model("chisoth", "preflare", setPars=freeze_dict)
m = xp.Model("chisoth", "preflare")
xp.Fit.renorm()
FIP_elements = ["Fe", "Si", "S", "Mg"]
model_components_list = m.componentNames
fit_pars = ["logT", "norm", *FIP_elements]
FIP_par_index = []


for component_i in model_components_list:
    for FIP_el in FIP_elements:
        idx_temp = eval(f"m.{component_i}.{FIP_el}.index")
        FIP_par_index.append(idx_temp)

err_string = 'preflare:1,'+''.join([str(i)+',' for i in FIP_par_index])
# for model_component_i in model_components_list:
# m_i = getattr(m, model_component_i)
# for FIP_element_i in FIP_elements:
# par_FIP = getattr(m_i, FIP_element_i)

## First lets figure out temperature
xp.Fit.renorm()
m.chisoth.logT.frozen = False
# m.chitable_2.logT.frozen = False
xp.Fit.renorm()
xp.Fit.perform()
# freeze_dict = { i:',-1,,,,,' for i in range(1,m.nParameters+1)}

# for pars_i in m.chitable.parameterNames:
# par_temp = getattr(m.chitable, pars_i)
# par_temp.frozen = True
# Lets freeze all pars using a dictionary
unfreeze_dict = {i: ",0.1,,,," for i in FIP_par_index}
# m.setPars("6.3,1,,,,,", unfreeze_dict)
m.setPars(unfreeze_dict)

##% Fit
xp.Fit.renorm()
xp.Fit.perform()
xp.Xset.save(f"{PHA_dir}/preflare_prl")  # , info="m")
xp.Xset.closeLog()

##% Plot
fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(16,9))
xp.Plot.device = "/xs"
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
ax[0].plot(x,model,drawstyle='steps-mid')
ax[0].errorbar(x,y,xerr=x_err,yerr=y_err,linestyle='None',fmt='k')
ax[0].set_yscale('log')
ax[0].set_ylabel('counts/s/keV')
ax[1].hlines(0,min(chix),max(chix), )
ax[1].errorbar(chix,chi,xerr=chix_err,yerr=chi_err,linestyle='None',fmt='k')
ax[1].set_ylabel('(data-model)/error')
fig.supxlabel('Energy (keV)')
plt.savefig(f"{PHA_dir}/fit.png")
plt.show()

#%%
np.savetxt("times.txt",[date_init,date_stop],fmt='%s',)
