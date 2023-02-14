import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xspec as xp
from data_utils import create_daxss_pha, read_daxss_data

plt.style.use("fivethirtyeight")
keV_to_MK = 11604562.9141 / 1e6
dates_init = [
    "2022-03-15T23:20:00",
    "2022-03-31T18:00:00",
    "2022-05-05T13:32:00",
    "2022-03-22T20:20:00",
    "2022-05-03T08:00:00",
    "2022-03-22T11:00:00",
    "2022-03-07T22:40:00",
]
dates_stop = [
    "2022-03-15T23:30:00",
    "2022-03-31T23:00:00",
    "2022-05-05T13:50:00",
    "2022-03-22T20:30:00",
    "2022-05-03T08:38:00",
    "2022-03-22T11:13:00",
    "2022-03-07T22:55:00",
]

spec_num = 2
date_init = dates_init[2]
date_stop = dates_stop[2]
# DAXSS_file = "/home/sac/Asif/Chianti_codes/main/daxss_solarSXR_level1_2022-02-14-mission_v2.0.0.ncdf"
DAXSS_file = "/home/sac/Asif/Chianti_codes/main/daxss_solarSXR_level1_2022-02-14-mission_v2.0.0.ncdf"
daxss_data = read_daxss_data(DAXSS_file)
time_sel = slice(date_init, date_stop)

daxss_data = read_daxss_data(DAXSS_file)
flare_data = daxss_data.sel(time=slice(date_init, date_stop))

#%% Create PHA files
CREATE = True
PHA_dir = f"./PHA_Files/prl_model/flare_{date_init}"
arf_path = "/home/sac/Asif/Chianti_codes/main/minxss_fm3_ARF.fits"
rmf_path = "/home/sac/Asif/Chianti_codes/main/minxss_fm3_RMF.fits"
preflare_dir = f"{PHA_dir}/preflare_prl"

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

xp.AllModels.lmod("chspec", dirPath="/home/sac/chspec/")
xp.AllData.clear()
xp.AllModels.clear()
# xp.Xset.abund = "feld"
# xp.Fit.statMethod = "cstat"
xp.Fit.query = "no"  # No asking for uesr confirmation while fitting
# xp.Plot.xAxis = "channel"
xp.Plot.yLog = True
xp.Plot.xLog = False
# table_dir = "/home/asif/Documents/work/all_tables_idl"

##% prepare Spectrum
xp.Xset.parallel.leven = 6
# xp.Xset.parallel.error = 4

FIP_elements = ["Fe", "Ar", "Ca", "Si", "S", "Mg"]
other_pars = ["logT", "norm"]
fit_pars = other_pars + FIP_elements

suffix = ["", "_2"]

# xp.Xset.restore(f"{preflare_dir}/preflare_prl.xcm")
# preflare_m = xp.AllModels(1, "preflare").chisoth

# preflare_par_vals = []
# for par in preflare_m.parameterNames:
# preflare_par_vals.append(eval(f"preflare_m.{par}.values"))
times = flare_data.time.data
colum_names = []
for fit_pars_i in fit_pars:
    for suffix_i in suffix:
        colum_names.append(fit_pars_i + suffix_i + "_values")
        colum_names.append(fit_pars_i + suffix_i + "_UB")
        colum_names.append(fit_pars_i + suffix_i + "_LB")
        colum_names.append(fit_pars_i + suffix_i + "_err_code")
colum_names.append("Chi")

# par_dict = {}
# for i, preflare_par_val_i in enumerate(preflare_par_vals):
# par_dict[63 + i] = f"{preflare_par_val_i[0]},-1,,,,"

# Setting normalisatoins
# par_dict[32] = f"{preflare_par_vals[31][0]/10},,,,,,"
# par_dict[64] = f"{preflare_par_vals[31][0]},,,,,,"
# par_dict[64] = "210,5,,,,,"

# par_dict = {i: ",-1,,,,," for i in range(1, 65)}
# # Setting logT values
# par_dict[1] = "7,0.1,,,, "
# par_dict[33] = "6.8,0.1,,,, "
m = xp.Model("chisoth + chisoth", "flare")
model_components_list = m.componentNames
FIP_par_index = []
other_par_index = []
for component_i in model_components_list:
    for other_pars_i in other_pars:
        idx_temp = eval(f"m.{component_i}.{other_pars_i}.index")
        other_par_index.append(idx_temp)
    # idx_temp = eval(f"m.{component_i}.{FIP_el}.index")

unfreeze_dict = {}
temperature_dict = {
    eval("m.chisoth.logT.index"): ",0.001,,,,,",
    eval("m.chisoth_2.logT.index"): ",0.001,,,,,",
}
for FIP_el in FIP_elements:
    idx_temp = eval(f"m.chisoth.{FIP_el}.index")
    par_1 = eval(f"m.chisoth.{FIP_el}")
    par_2 = eval(f"m.chisoth_2.{FIP_el}")
    par_2.link = par_1
    idx_temp = eval(f"m.chisoth.{FIP_el}.index")
    FIP_par_index.append(idx_temp)
    unfreeze_dict[idx_temp] = ",0.1,,,,,"
all_par_index = other_par_index + FIP_par_index
err_string = "flare:" + "".join([str(i) + "," for i in all_par_index])
xp.AllData.clear()
s = xp.Spectrum(PHA_file_list[0])
xp.Fit.renorm()
m.setPars(temperature_dict)
xp.Fit.perform()
xp.Fit.renorm()
m.setPars(unfreeze_dict)

##% Fit
cutoff_cps = 0.1
par_vals = []
out_dir = f"{PHA_dir}/2T"
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
for time_i, PHA_file in enumerate(PHA_file_list):
    f_name = os.path.basename(PHA_file).removesuffix(".pha")
    temp_xx = flare_data["cps"].sel(energy=slice(0.7, None)).isel(time=time_i)
    cutoff_idx = np.where(temp_xx < cutoff_cps)[0][0]
    cutoff_eng = temp_xx.energy[cutoff_idx]
    xp.AllData.clear()
    s = xp.Spectrum(PHA_file)
    # s.ignore(f'{cutoff_idx +1}-**')
    xp.Fit.statMethod = "chi"
    xp.Plot.xAxis = "keV"
    xp.Plot.yLog = True
    xp.Plot.xLog = False
    logFile = xp.Xset.openLog(f"{out_dir}/{f_name}.log")
    # s.ignore("**-0.7 7.0-**")
    s.ignore(f"**-0.7 {cutoff_eng}-**")
    #%% Set up the model
    xp.Fit.renorm()
    xp.Fit.perform()
    xp.Fit.renorm()
    xp.Fit.perform()
    # if xp.Fit.testStatistic>300:
    n = 0
    while xp.Fit.testStatistic > 350 and n < 5:
        n += 1
        # m.setPars({62: ",20,,,,,"})
        # xp.Fit.renorm()
        xp.Fit.perform()
    xp.Fit.error(err_string)
    xp.Xset.save(f"{out_dir}/{f_name}.xcm")
    xp.Xset.closeLog()
    temp_col = []
    for fit_pars_i in fit_pars:
        for suffix_i in suffix:
            m_par_i = eval(f"m.chisoth{suffix_i}.{fit_pars_i}")
            temp_col.append(m_par_i.values[0])
            temp_col.append(m_par_i.error[0])
            temp_col.append(m_par_i.error[1])
            temp_col.append(m_par_i.error[2])
    temp_col.append(xp.Fit.testStatistic)
    par_vals.append(temp_col)
    ##% Plot
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(16, 9))
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
    ax[0].plot(x, model, drawstyle="steps-mid")
    ax[0].errorbar(x, y, xerr=x_err, yerr=y_err, linestyle="None", fmt="k")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("counts/s/keV")
    ax[1].hlines(
        0,
        min(chix),
        max(chix),
    )
    ax[1].errorbar(chix, chi, xerr=chix_err, yerr=chi_err, linestyle="None", fmt="k")
    ax[1].set_ylabel("(data-model)/error")
    fig.supxlabel("Energy (keV)")
    plt.savefig(f"{out_dir}/{f_name}.png")
    plt.close()

#%% Make a data frame
# par_vals = np.array(par_vals)
df = pd.DataFrame(par_vals, columns=colum_names, index=times)
# df.insert(0,'time',times)
# df[df<0]= 0
# df[df>1e3] = 0
df.to_csv(f"{out_dir}/results.csv")

#%% Plotting
net_counts = flare_data["cps"].sum(dim="energy")

fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(16, 16))
df.plot(
    y="logT_values",
    yerr=(df["logT_values"] - df["logT_LB"], df["logT_UB"] - df["logT_values"]),
    ax=ax[0],
)
df.plot(
    y="logT_2_values",
    # yerr=(df["logT_2_values"] - df["logT_2_LB"], df["logT_2_UB"] - df["logT_2_values"]),
    ax=ax[0],
)
# df.plot(y="logT_2_values", yerr="logT_2_UB", ax=ax[0])
ax[0].set_ylabel("log Temperature ")
df.plot(y="norm_values", yerr="norm_UB", ax=ax[1])
df.plot(y="norm_2_values", yerr="norm_2_UB", ax=ax[1])
ax[1].set_ylabel(r"$EM (10^{26} cm^{-5})$")
FIP_err = [i + "_UB" for i in FIP_elements]

for FIP_element_i in FIP_elements:
    y_val_str = FIP_element_i + "_values"
    y_val_UB = f"{FIP_element_i}_UB"
    y_val_LB = f"{FIP_element_i}_LB"
    er = ((df[y_val_str] - df[y_val_LB], df[y_val_UB] - df[y_val_str]),)
    # df.plot(y=FIP_element_i + "_values", yerr=, ax=ax[2])
    df.plot(
        y=y_val_str,
        yerr=er,
        ax=ax[2],
    )
ax[2].set_ylabel("Abundance")

df.plot(y="Chi", ax=ax[3])
ax[3].set_ylabel("Chi square")

for ax_i in ax:
    ax2 = ax_i.twinx()
    ax2.grid(visible=False)
    # net_counts.plot.line('o--',color='grey',ax=ax2)
    net_counts.plot.line("o--", alpha=0.6, color="black", ax=ax2)

plt.savefig(f"{out_dir}/results.png")
# df.plot(y=FIP_elements,yerr= FIP_err)
plt.show()
