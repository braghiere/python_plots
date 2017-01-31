#!/usr/bin/env python

__version__ = "Time-stamp: <2012-08-13 16:52 ycopin@lyopc469>"
__author__ = "Yannick Copin <yannick.copin@laposte.net>"

"""
Example of use of TaylorDiagram. Illustration dataset courtesy of
Michael Rawlins.
Rawlins, M. A., R. S. Bradley, H. F. Diaz, 2012. Assessment of
regional climate model simulation estimates over the Northeast U.S.,
Journal of Geophysical Research, in review.
"""

from taylorDiagram import TaylorDiagram
import numpy as NP
import matplotlib.pyplot as PLT

#PLT.style.use('seaborn-white')
SIZE = 20
PLT.rc('font', size=SIZE)  # controls default text sizes
PLT.rc('axes', titlesize=SIZE)  # fontsize of the axes title
PLT.rc('axes', labelsize=SIZE)  # fontsize of the x any y labels
PLT.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
PLT.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
PLT.rc('legend', fontsize=SIZE)  # legend fontsize
PLT.rc('figure', titlesize=SIZE)  # # size of the figure title


# Reference std
stdrefs = dict(fAPAR=0.272183321891,
               fANIR=0.137619193669,
               albPAR=0.207417234457,
               albNIR=0.133474526073)

# Sample std,rho: Be sure to check order and that correct numbers are placed!
samples = dict(fAPAR=[ [0.280350593497, 0.949910566609, "MAESPA"],
                       [0.223675458769, 0.97240076287, "GORT"],
                       [0.224832072187, 0.938370874985, "Two-stream"],
                       [0.226102484666, 0.980193800587, "Nilson"],
                       [0.294639522219, 0.86190429029, "Kucharik"],
                       [0.267487012651, 0.99440072113, "Pinty"],
                       [0.214321909284, 0.987967078029, "Ni-Meister"],
                       [0.140371206859, 0.831375227844, "Veg$_{frac}$"]],
               fANIR=[ [0.0922739029312, 0.961522291458, "MAESPA"],
                       [0.194068847782, 0.98275733876, "GORT"],
                       [0.117885253626, 0.926533787013, "Two-stream"],
                       [0.0875025786585, 0.974507463867, "Nilson"],
                       [0.142077917177, 0.867741210408, "Kucharik"],
                       [0.12460929225, 0.964835722584, "Pinty"],
                       [0.0743036555776, 0.9871085969, "Ni-Meister"],
                       [0.0655475811586, 0.827950903345, "Veg$_{frac}$"]],
               albPAR=[[0.0120349250548, 0.108779257021, "MAESPA"],
                       [0.227870629005, 0.974496508613, "GORT"],
                       [0.112658128624, 0.940121010462, "Two-stream"],
                       [0.209773750223, 0.994692289645, "Nilson"],
                       [0.27816551937, 0.950502200915, "Kucharik"],
                       [0.205790238274, 0.997428445339, "Pinty"],
                       [0.2175988912, 0.989851512367, "Ni-Meister"], 
                       [0.338467854244, 0.85709841973, "Veg$_{frac}$"]],
               albNIR=[[0.073650044017, 0.563695958109, "MAESPA"],
                       [0.126746493157, 0.738437704804, "GORT"],
                       [0.113043301693, 0.944384630359, "Two-stream"], 
                       [0.148899756127, 0.989793302273, "Nilson"],
                       [0.165566267298, 0.966432775732, "Kucharik"],
                       [0.136146899982, 0.967818436216, "Pinty"],
                       [0.155572682806, 0.981904497648, "Ni-Meister"], 
                       [0.191108508658, 0.905880517134, "Veg$_{frac}$"]])

# Colormap (see http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps)
colors = PLT.matplotlib.cm.Set1(NP.linspace(0,1,len(samples['fAPAR'])))

# Here set placement of the points marking 95th and 99th significance
# levels. For more than 102 samples (degrees freedom > 100), critical
# correlation levels are 0.195 and 0.254 for 95th and 99th
# significance levels respectively. Set these by eyeball using the
# standard deviation x and y axis.

#x95 = [0.01, 0.68] # For Tair, this is for 95th level (r = 0.195)
#y95 = [0.0, 3.45]
#x99 = [0.01, 0.95] # For Tair, this is for 99th level (r = 0.254)
#y99 = [0.0, 3.45]

#x95 = [0.05, 13.9] # For Prcp, this is for 95th level (r = 0.195)
#y95 = [0.0, 71.0]
#x99 = [0.05, 19.0] # For Prcp, this is for 99th level (r = 0.254)
#y99 = [0.0, 70.0]




rects = dict(fAPAR=221,
             fANIR=222,
             albPAR=223,
             albNIR=224)



fig = PLT.figure(figsize=(22,16))
#fig.suptitle("Shortwave\n Radiation\n Partitioning", size='x-large')

for season in ['fAPAR','fANIR','albPAR','albNIR']:

    dia = TaylorDiagram(stdrefs[season], fig=fig, rect=rects[season],
                        label='RAMI4PILPS')

    #dia.ax.plot(x95,y95,color='k')
    #dia.ax.plot(x99,y99,color='k')

    # Add samples to Taylor diagram
    for i,(stddev,corrcoef,name) in enumerate(samples[season]):
        dia.add_sample(stddev, corrcoef,
                       marker='$%d$' % (i+1), ms=22, ls='',
                       #mfc='k', mec='k', # B&W
                       mfc=colors[i], mec=colors[i], # Colors
                       label=name)

    # Add RMS contours, and label them
    contours = dia.add_contours(levels=5, colors='0.5') # 5 levels
    dia.ax.clabel(contours, inline=1, fontsize=14, fmt='%.1f')
    # Tricky: ax is the polar ax (used for plots), _ax is the
    # container (used for layout)
    dia._ax.set_title(season.capitalize())

# Add a figure legend and title. For loc option, place x,y tuple inside [ ].
# Can also use special options here:
# http://matplotlib.sourceforge.net/users/legend_guide.html

fig.legend(dia.samplePoints,
           [ p.get_label() for p in dia.samplePoints ],
           numpoints=1, prop=dict(size='large'), loc='best')

fig.tight_layout()

PLT.savefig('summary_4panel_taylordiagram.png')
PLT.show()
