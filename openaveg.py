import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import tempfile
from scipy import stats
import matplotlib.lines as mlines

#identifies all the openness files in a folder

list_of_files = glob.glob('./openness*.txt')

num_plots = len(list_of_files)
#print num_plots


# Have a look at the colormaps here and decide which one you'd like:
# http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])

pgap_mat=[]

for fileName in list_of_files:
	# Open file
        
     	with open(fileName, 'r') as f:
                sza = []
                pgap = []
                # Loop over lines and extract variables of interest
		for line in f:
			line = line.strip()
                        columns = line.split()
			sza.append(float(columns[0]))
    			pgap.append(float(columns[1]))
                #print sza, pgap
		pgap_mat.append(pgap)
		plt.plot(sza, pgap)

pgap_mat = np.array(pgap_mat)

# np.mean(pgap_mat,axis=0) > average the columns of your matrix
# np.mean(pgap_mat,axis=1) > average the lines of your matrix
mean_pgap = np.mean(pgap_mat,axis=0)
# np.mean(pgap_mat,axis=1) > standard deviation the columns of your matrix
std_pgap = np.std(pgap_mat,axis=0)
std = (std_pgap/np.sqrt(num_plots))

# calculates the confidence interval of 95% 
conf_int_pgap_95 = stats.t.interval(0.95,num_plots-1,mean_pgap,std)

#calculates the deviation from the confidence intervals and the mean
yerr_l = mean_pgap-conf_int_pgap_95[0]
yerr_u = conf_int_pgap_95[1]- mean_pgap
yerr = (yerr_l,yerr_u)

#print 'mean=',mean_pgap
#print 'ci95=',yerr
print "sza =",sza,"pgap =", mean_pgap,"95%ci =", yerr

plt.xlim([0, 90])
plt.ylim([0, 1])
line1=plt.plot(sza, mean_pgap, color='k',linestyle='--',linewidth=3.0)
line2=plt.plot(sza, conf_int_pgap_95[0], color='k',linestyle='--',linewidth=2.0)
plt.plot(sza, conf_int_pgap_95[1], color='k',linestyle='--',linewidth=2.0)
plt.errorbar(sza, mean_pgap, yerr,linestyle='None',linewidth=3.0)
#Splt.errorbar(sza, mean_pgap, 3*std_pgap,color='r',linestyle='--',linewidth=3.0)
plt.ylabel('Pgap')
plt.xlabel('Solar Zenith Angle')
#write and plot legend
mean_line = mlines.Line2D([], [], color='k',linestyle='--',linewidth=3.0,
                          label='Average')
ci_line = mlines.Line2D([], [], color='k',linestyle='--',linewidth=2.0,
                          label='CI95%')
plt.legend(handles=[mean_line,ci_line])
#show plot
plt.show()

f.close()





