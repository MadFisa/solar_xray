"""
File: create_xcm.py
Author: Asif Mohamed Mandayapuram
Email: asifmp97@gmail.com
Github: github/MadFisa
Description: Code to generate xcm file for ch2isoth_2T
"""


import xspec as xp
import os 

xp.AllModels.lmod("chspec", dirPath=f"{os.path.expanduser('~')}/chspec/")
xp.AllData.clear()
xp.AllModels.clear()
xp.Fit.query = "no"  # No asking for uesr confirmation while fitting
xp.Plot.xAxis = "keV"
xp.Plot.yLog = True
xp.Plot.xLog = False
xp.Xset.parallel.leven = 6
xp.Plot.device = "/xw"

#%% Create model

m = xp.Model("chisoth + chisoth", "flare")
par_names = m.chisoth.parameterNames
for par_i in par_names:
    if par_i!='logT' and par_i!='norm':
        par1 = eval(f"m.chisoth.{par_i}")
        par2 = eval(f"m.chisoth_2.{par_i}")
        print(f"linking {par1.name}")
        par2.link = par1
m.chisoth.logT = 7.2
m.chisoth_2.logT = 6.8
xp.Xset.save("2T.xcm")