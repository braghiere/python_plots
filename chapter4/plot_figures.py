import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib import rcParams 
from pylab import * 


#from /rami4pilps/RAMI4PILPS_reference_values import *

#plt.style.use('ggplot')
plt.style.use('seaborn-white')
SIZE = 78
plt.rc('font', size=SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)  # fontsize of the x any y labels
plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)  # legend fontsize
plt.rc('figure', titlesize=SIZE)  # # size of the figure title



data = pd.read_csv('/home/mn811042/src/maespaRAMI/vertical/same_lai_150_different_structures.dat', sep="\t")
maespa = pd.read_csv('/home/mn811042/src/julesRT_struct_2/julesRT_struct/data_comparison/maespa/maespa_values.txt', sep="\t")
gort = pd.read_csv('/home/mn811042/src/julesRT_struct_2/julesRT_struct/data_comparison/gort/gort_values.txt', sep="\t")
struc = pd.read_csv('/home/mn811042/src/julesRT_struct_2/julesRT_struct/data_comparison/struc/struc_values.txt', sep="\t")
twostream = pd.read_csv('/home/mn811042/src/julesRT_struct_2/julesRT_struct/data_comparison/two_stream/twostream_values.txt', sep="\t")
rami4pilps = pd.read_csv('/home/mn811042/src/julesRT_struct_2/julesRT_struct/data_comparison/rami4pilps/rami4pilps.txt', sep="\t")
ni_meister = pd.read_csv('/home/mn811042/src/julesRT_struct_2/julesRT_struct/data_comparison/ni_meister/ni_values.txt', sep="\t")
kucharik = pd.read_csv('/home/mn811042/src/julesRT_struct_2/julesRT_struct/data_comparison/kucharik/kucharik_values.txt', sep="\t")
nilson = pd.read_csv('/home/mn811042/src/julesRT_struct_2/julesRT_struct/data_comparison/nilson/nilson_values.txt', sep="\t")

#jules_par = pd.read_csv('/home/mn811042/src/maespaRAMI/vertical/TS_par_lai_150.dat', sep=",")
#jules_nir = pd.read_csv('/home/mn811042/src/maespaRAMI/vertical/TS_nir_lai_150.dat', sep=",")

#print jules_nir

partition = ['fanir','fapar','albpar','albnir']
LAI = ['050','150','250']
soil = ['BLK','MED','SNW']

for i in partition:
 for j in LAI:
  for k in soil:
   
    x = maespa["SZA"]

    y1 = maespa["%s_%s_%s" % (i,j,k)]
    y2 = gort["%s_%s_%s" % (i,j,k)]
    y3 = struc["%s_%s_%s" % (i,j,k)]
    y4 = twostream["%s_%s_%s" % (i,j,k)]
    y6 = ni_meister["%s_%s_%s" % (i,j,k)]
    y7 = kucharik["%s_%s_%s" % (i,j,k)]
    y8 = nilson["%s_%s_%s" % (i,j,k)]

    x_rami = rami4pilps["SZA"]
    y5 = rami4pilps["%s_%s_%s" % (i,j,k)]


    fig = plt.figure(figsize=(30, 28)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])

    ax0.plot(x_rami[:-1],y5[:-1],'x', mew = 15, ms = 70, c= 'k', label = 'RAMI4PILPS')
    ax0.plot(x[:-1], y1[:-1]   ,'r',linewidth=15, label = 'MAESPA')
    ax0.plot(x[:-1], y2[:-1]   ,'b',linewidth=15, label = 'GORT')

    # Two-stream
    ax0.plot(x[:-1], y4[:-1],'g',linewidth=15, label = 'Two-stream')


    # Parameterization schemes 
    ax0.plot(x[:-1], y8[:-1]   ,'--',color='purple',linewidth=15, label = 'Nilson')
    ax0.plot(x[:-1], y7[:-1]   ,'--',color='gold',linewidth=15, label = 'Kucharik')
    ax0.plot(x[:-1], y3[:-1]   ,'--',color='darkorange',linewidth=15, label = 'Pinty')
    ax0.plot(x[:-1], y6[:-1]   ,'--',color='hotpink',linewidth=15, label = 'Ni-Meister')

    # VEG FRACTION 
    if i == 'fanir' or i == 'fapar':
       if j == '050':
          ax0.plot(x[:-1], y4[:-1]*0.09,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
       if j == '150':
          ax0.plot(x[:-1], y4[:-1]*0.26,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
       if j == '250':
          ax0.plot(x[:-1], y4[:-1]*0.43,'darkgray',linewidth=15, label = 'Veg$_{frac}$')

    if i == 'albnir':
       if j == '050':
          if k == 'BLK':
             ax0.plot(x[:-1], y4[:-1]*0.09 + (1. - 0.09)*0.00,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
          if k == 'MED':
             ax0.plot(x[:-1], y4[:-1]*0.09 + (1. - 0.09)*0.21,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
          if k == 'SNW':
             ax0.plot(x[:-1], y4[:-1]*0.09 + (1. - 0.09)*0.56,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
       if j == '150':
          if k == 'BLK':
             ax0.plot(x[:-1], y4[:-1]*0.26 + (1. - 0.26)*0.00,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
          if k == 'MED':
             ax0.plot(x[:-1], y4[:-1]*0.26 + (1. - 0.26)*0.21,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
          if k == 'SNW':
             ax0.plot(x[:-1], y4[:-1]*0.26 + (1. - 0.26)*0.56,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
       if j == '250':
          if k == 'BLK':
             ax0.plot(x[:-1], y4[:-1]*0.43 + (1. - 0.43)*0.00,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
          if k == 'MED':
             ax0.plot(x[:-1], y4[:-1]*0.43 + (1. - 0.43)*0.21,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
          if k == 'SNW':
             ax0.plot(x[:-1], y4[:-1]*0.43 + (1. - 0.43)*0.56,'darkgray',linewidth=15, label = 'Veg$_{frac}$')

    if i == 'albpar':
       if j == '050':
          if k == 'BLK':
             ax0.plot(x[:-1], y4[:-1]*0.09 + (1. - 0.09)*0.00,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
          if k == 'MED':
             ax0.plot(x[:-1], y4[:-1]*0.09 + (1. - 0.09)*0.12,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
          if k == 'SNW':
             ax0.plot(x[:-1], y4[:-1]*0.09 + (1. - 0.09)*0.96,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
       if j == '150':
          if k == 'BLK':
             ax0.plot(x[:-1], y4[:-1]*0.26 + (1. - 0.26)*0.00,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
          if k == 'MED':
             ax0.plot(x[:-1], y4[:-1]*0.26 + (1. - 0.26)*0.12,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
          if k == 'SNW':
             ax0.plot(x[:-1], y4[:-1]*0.26 + (1. - 0.26)*0.96,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
       if j == '250':
          if k == 'BLK':
             ax0.plot(x[:-1], y4[:-1]*0.43 + (1. - 0.43)*0.00,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
          if k == 'MED':
             ax0.plot(x[:-1], y4[:-1]*0.43 + (1. - 0.43)*0.12,'darkgray',linewidth=15, label = 'Veg$_{frac}$')
          if k == 'SNW':
             ax0.plot(x[:-1], y4[:-1]*0.43 + (1. - 0.43)*0.96,'darkgray',linewidth=15, label = 'Veg$_{frac}$')



    # Define title and title position
    if i == 'fanir': 
       ttl=plt.title(r'fabs - NIR - %s - %s'% (j,k) ,fontsize = 84)
    if i == 'fapar': 
       ttl=plt.title(r'fabs - PAR - %s - %s'% (j,k) ,fontsize = 84)
    if i == 'albnir': 
       ttl=plt.title(r'fref - NIR - %s - %s'% (j,k) ,fontsize = 84)
    if i == 'albpar': 
       ttl=plt.title(r'fref - PAR - %s - %s'% (j,k) ,fontsize = 84)

    ttl.set_position([.5, 1.05])

    if j == '050' and k == 'BLK':
       plt.legend(loc=2)

    #Define distance between axis values and axis line
    ax0.tick_params(axis='both', which='major', pad=20)

    ax0.set_xlim([0.0,87])

    if i == 'albpar' and (k == 'BLK' or k == 'MED'):
       ax0.set_ylim([0.0,0.2])
    else:
       ax0.set_ylim([0.0,1.0])

    ax0.grid(True)

    ax1 = plt.subplot(gs[1])
    x1 = np.array([1])

    my_xticks = ['ISO']
    plt.xticks(x1, my_xticks)

    ax1.plot(x1,y8.iloc[-1], '+', markersize=70,markeredgewidth=15,markeredgecolor='purple',markerfacecolor='None' ) #Nilson
    ax1.plot(x1,y7.iloc[-1], '+', markersize=70,markeredgewidth=15,markeredgecolor='gold',markerfacecolor='None' ) #Kucharik
    ax1.plot(x1,y6.iloc[-1], '+', markersize=70,markeredgewidth=15,markeredgecolor='hotpink',markerfacecolor='None' ) #Ni
    ax1.plot(x1,y3.iloc[-1], '+', markersize=70,markeredgewidth=15,markeredgecolor='darkorange',markerfacecolor='None' ) #Pinty
    ax1.plot(x1,y2.iloc[-1], 's', markersize=70,markeredgewidth=15,markeredgecolor='b',markerfacecolor='None' ) #gort
    ax1.plot(x1,y1.iloc[-1], 'd', markersize=70,markeredgewidth=15,markeredgecolor='r',markerfacecolor='None' )#maespa
    ax1.plot(x1,y4.iloc[-1], 'o', markersize=70,markeredgewidth=15,markeredgecolor='g',markerfacecolor='None' ) #two-stream


    #ax1.plot(x1,y4.iloc[-1]*0.26, '3', markersize=70,markeredgewidth=15,markeredgecolor='g',markerfacecolor='None' )#veg fraction

    # VEG FRACTION 
    if i == 'fanir' or i == 'fapar':
       if j == '050':
          ax1.plot(x1,y4.iloc[-1]*0.09,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
       if j == '150':
          ax1.plot(x1,y4.iloc[-1]*0.26,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
       if j == '250':
          ax1.plot(x1,y4.iloc[-1]*0.43,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')

    if i == 'albnir':
       if j == '050':
          if k == 'BLK':
             ax1.plot(x1,y4.iloc[-1]*0.09 + (1. - 0.09)*0.00,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
          if k == 'MED':
             ax1.plot(x1,y4.iloc[-1]*0.09 + (1. - 0.09)*0.21,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
          if k == 'SNW':
             ax1.plot(x1,y4.iloc[-1]*0.09 + (1. - 0.09)*0.56,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
       if j == '150':
          if k == 'BLK':
             ax1.plot(x1,y4.iloc[-1]*0.26 + (1. - 0.26)*0.00,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
          if k == 'MED':
             ax1.plot(x1,y4.iloc[-1]*0.26 + (1. - 0.26)*0.21,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
          if k == 'SNW':
             ax1.plot(x1,y4.iloc[-1]*0.26 + (1. - 0.26)*0.56,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
       if j == '250':
          if k == 'BLK':
             ax1.plot(x1,y4.iloc[-1]*0.43 + (1. - 0.43)*0.00,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
          if k == 'MED':
             ax1.plot(x1,y4.iloc[-1]*0.43 + (1. - 0.43)*0.21,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
          if k == 'SNW':
             ax1.plot(x1,y4.iloc[-1]*0.43 + (1. - 0.43)*0.56,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')

    if i == 'albpar':
       if j == '050':
          if k == 'BLK':
             ax1.plot(x1,y4.iloc[-1]*0.09 + (1. - 0.09)*0.00,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
          if k == 'MED':
             ax1.plot(x1,y4.iloc[-1]*0.09 + (1. - 0.09)*0.12,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
          if k == 'SNW':
             ax1.plot(x1,y4.iloc[-1]*0.09 + (1. - 0.09)*0.96,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
       if j == '150':
          if k == 'BLK':
             ax1.plot(x1,y4.iloc[-1]*0.26 + (1. - 0.26)*0.00,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
          if k == 'MED':
             ax1.plot(x1,y4.iloc[-1]*0.26 + (1. - 0.26)*0.12,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
          if k == 'SNW':
             ax1.plot(x1,y4.iloc[-1]*0.26 + (1. - 0.26)*0.96,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
       if j == '250':
          if k == 'BLK':
             ax1.plot(x1,y4.iloc[-1]*0.43 + (1. - 0.43)*0.00,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
          if k == 'MED':
             ax1.plot(x1,y4.iloc[-1]*0.43 + (1. - 0.43)*0.12,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')
          if k == 'SNW':
             ax1.plot(x1,y4.iloc[-1]*0.43 + (1. - 0.43)*0.96,'3', markersize=70,markeredgewidth=15,markeredgecolor='darkgray',markerfacecolor='None')


    ax1.plot(x1,y5.iloc[-1], 'x', markersize=70,markeredgewidth=15,markeredgecolor='k',markerfacecolor='None' ) # rami4pilps

    
    if i == 'albpar' and (k == 'BLK' or k == 'MED'):
       ax1.set_ylim([0.0,0.2])
    else:
       ax1.set_ylim([0.0,1.0])
    
    ax1.grid(True)
    ax1.set_yticklabels([])
    ax1.tick_params(axis='x', which='major', pad=20)


    for ax in [ax0]:
        ax.set_xlabel('Solar Zenith Angle',fontdict=None, labelpad=20)
        if i == 'fapar':
           ax.set_ylabel('fAPAR', fontdict=None, labelpad=20)
        if i == 'albpar':
           ax.set_ylabel('albedo PAR', fontdict=None, labelpad=20)
        if i == 'fanir':
           ax.set_ylabel('fANIR', fontdict=None, labelpad=20)
        if i == 'albnir':
           ax.set_ylabel('albedo NIR', fontdict=None, labelpad=20)
  
    plt.tight_layout()
    plt.savefig('/home/mn811042/src/julesRT_struct_2/julesRT_struct/data_comparison/figures/%s_%s_%s.png'% (i,j,k))
    #Status
    print '%s_%s_%s.png done!' % (i,j,k)
    #plt.show()
