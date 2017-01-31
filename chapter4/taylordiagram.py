#!/usr/bin/env python
# Copyright: This document has been placed in the public domain.

"""
Taylor diagram (Taylor, 2001) test implementation.

http://www-pcmdi.llnl.gov/about/staff/Taylor/CV/Taylor_diagram_primer.htm
"""

__version__ = "Time-stamp: <2012-02-17 20:59:35 ycopin>"
__author__ = "Yannick Copin <yannick.copin@laposte.net>"

import numpy as NP
import matplotlib.pyplot as PLT
import pandas as PD


class TaylorDiagram(object):
    """Taylor diagram: plot model standard deviation and correlation
    to reference (data) sample in a single-quadrant polar plot, with
    r=stddev and theta=arccos(correlation).
    """

    def __init__(self, refstd, fig=None, rect=111, label='_'):
        """Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using mpl_toolkits.axisartist.floating_axes. refstd is
        the reference standard deviation to be compared to.
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd            # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = NP.concatenate((NP.arange(10)/10.,[0.95,0.99]))
        tlocs = NP.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str,rlocs))))

        # Standard deviation axis extent
        self.smin = 0
        self.smax = 1.5*self.refstd

        ghelper = FA.GridHelperCurveLinear(tr,
                                           extremes=(0,NP.pi/2, # 1st quadrant
                                                     self.smin,self.smax),
                                           grid_locator1=gl1,
                                           tick_formatter1=tf1,
                                           )

        if fig is None:
            fig = PLT.figure()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")  # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom") # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")   # "Y axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")

        ax.axis["bottom"].set_visible(False)         # Useless
        
        # Contours along standard deviations
        ax.grid(False)

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        print "Reference std:", self.refstd
        l, = self.ax.plot([0], self.refstd, 'k*',
                          ls='', ms=10, label=label)
        t = NP.linspace(0, NP.pi/2)
        r = NP.zeros_like(t) + self.refstd
        self.ax.plot(t,r, 'k--', label='_')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """Add sample (stddev,corrcoeff) to the Taylor diagram. args
        and kwargs are directly propagated to the Figure.plot
        command."""

        l, = self.ax.plot(NP.arccos(corrcoef), stddev,
                          *args, **kwargs) # (theta,radius)
        self.samplePoints.append(l)

        return l

    def add_contours(self, levels=5, **kwargs):
        """Add constant centered RMS difference contours."""

        rs,ts = NP.meshgrid(NP.linspace(self.smin,self.smax),
                            NP.linspace(0,NP.pi/2))
        # Compute centered RMS difference
        rms = NP.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*NP.cos(ts))
        
        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours


if __name__=='__main__':


    #import files

    struc = PD.read_csv('/home/mn811042/src/julesRT_struct_2/julesRT_struct/data_comparison/struc/struc_values.txt', sep="\t")
    twostream = PD.read_csv('/home/mn811042/src/julesRT_struct_2/julesRT_struct/data_comparison/two_stream/twostream_values.txt', sep="\t")
    rami4pilps = PD.read_csv('/home/mn811042/src/julesRT_struct_2/julesRT_struct/data_comparison/rami4pilps/rami4pilps.txt', sep="\t")
    maespa = PD.read_csv('/home/mn811042/src/julesRT_struct_2/julesRT_struct/data_comparison/maespa/maespa_values.txt', sep="\t")
    gort = PD.read_csv('/home/mn811042/src/julesRT_struct_2/julesRT_struct/data_comparison/gort/gort_values.txt', sep="\t")
    nimeister = PD.read_csv('/home/mn811042/src/julesRT_struct_2/julesRT_struct/data_comparison/ni_meister/ni_values.txt', sep="\t")

 

    # Reference dataset
    x = NP.linspace(0,4*NP.pi,100)
    data = NP.sin(x)
    refstd = data.std(ddof=1)           # Reference standard deviation

    
    #x = maespa["SZA"][:-1]
    #data = maespa["fapar_150_BLK"][:-1]
    #refstd = data.std(ddof=1)

    x = rami4pilps["SZA"]
    data = rami4pilps["fapar_150_BLK"]
    refstd = data.std(ddof=1)

    x = NP.append([rami4pilps["SZA"]],[rami4pilps["SZA"]])
    data = NP.append([rami4pilps["fapar_150_BLK"]],[rami4pilps["fapar_150_MED"]])
    refstd = data.std(ddof=1)
    print data
    #print s1

    
    # Models
    #m1 = data + 0.2*NP.random.randn(len(x))    # Model 1
    m2 = 0.8*data + .1*NP.random.randn(len(x)) # Model 2
    m3 = NP.sin(x-NP.pi/10)                    # Model 3

    #print m3
    m1_fapar_150_BLK = gort["fapar_150_BLK"]
    m1_fapar_150_MED = gort["fapar_150_MED"]
    m1 = NP.array([m1_fapar_150_BLK.iloc[26],m1_fapar_150_BLK.iloc[59],m1_fapar_150_BLK.iloc[82],m1_fapar_150_BLK.iloc[90]])
    m1 = NP.append(m1,NP.array([m1_fapar_150_MED.iloc[26],m1_fapar_150_MED.iloc[59],m1_fapar_150_MED.iloc[82],m1_fapar_150_MED.iloc[90]]))
    #m2 = struc["fapar_150_BLK"]
    #m2 = NP.array([m2.iloc[26],m2.iloc[59],m2.loc[82],m2.iloc[90]])
    #m3 = twostream["fapar_150_BLK"]
    #m3 = NP.array([m3.iloc[26],m3.iloc[59],m3.loc[82],m3.iloc[90]])
    #print m3
 

    
    # Compute stddev and correlation coefficient of models
    samples = NP.array([ [m.std(ddof=1), NP.corrcoef(data, m)[0,1]]
                         for m in (m1,m2,m3)])

    fig = PLT.figure(figsize=(10,4))
    
    ax1 = fig.add_subplot(1,2,1, xlabel='X', ylabel='Y')
    # Taylor diagram
    dia = TaylorDiagram(refstd, fig=fig, rect=122, label="Reference")

    colors = PLT.matplotlib.cm.jet(NP.linspace(0,1,len(samples)))

    ax1.plot(x,data,'ko', label='Data')
    for i,m in enumerate([m1,m2,m3]):
        ax1.plot(x,m, c=colors[i], label='Model %d' % (i+1))
    ax1.legend(numpoints=1, prop=dict(size='small'), loc='best')

    # Add samples to Taylor diagram
    for i,(stddev,corrcoef) in enumerate(samples):
        dia.add_sample(stddev, corrcoef, marker='s', ls='', c=colors[i],
                       label="Model %d" % (i+1))

    # Add RMS contours, and label them
    contours = dia.add_contours(colors='0.5')
    PLT.clabel(contours, inline=1, fontsize=10)

    # Add a figure legend
    fig.legend(dia.samplePoints,
               [ p.get_label() for p in dia.samplePoints ],
               numpoints=1, prop=dict(size='small'), loc='upper right')

    ax1.set_ylim([0.0,1.0])


    PLT.show()

