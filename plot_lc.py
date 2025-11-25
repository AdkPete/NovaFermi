'''
Short routine to collect and plot the results of a light curve
generated with analyze_fermi.py
'''

import matplotlib.pyplot as plt
import sys , os
import analyze_fermi as af
import numpy as np


def plot_light_curve(params, display=False):
    '''
    Function to plot light curve results from analyze_fermi
    Assumes that you are running in the directory with the data
    TODO: Update this routine to accept data directory arguments
    Takes in a parameter dictionary, and will produce a TS plot 
    (Showing TS as a function of time) and will produce a light curve
    complete with uncertainties and upper limits as appropriate.
    All times will be plotted relative to the peak listed in the
    given parameter file.
    
    
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    display : boolean : If true, run plt.show() to display figures
    Returns
    _______
    None
    '''
    dir = "./"
    Flux = []
    Unc = []
    Time = []
    TS = []

    for i in os.listdir(dir):
        fname = os.path.join(dir , i)
        if ".csv" not in fname or "mp" not in fname:
            continue
        f = open(fname)
        for line in f.readlines():
            split_line = line.split(",")
            Flux.append(float(split_line[0]))
            Unc.append(float(split_line[1]))
            TS.append(float(split_line[2]))
            MET = float(split_line[3])
            tpeak = af.met_to_tpeak(MET , params)
            Time.append(tpeak)
            break
        
    Time = np.array(Time)
    TS = np.array(TS)
    Unc = np.array(Unc)
    Flux = np.array(Flux)


    plt.scatter(Time , TS)
    plt.xlabel("Time since Peak (Days)")
    plt.ylabel("TS")
    plt.savefig("TSFig.pdf")
    if display:
        plt.show()
    plt.close()
    
    det = np.where(TS >= 4)
    lim = np.where(TS < 4)

    plt.scatter(Time[det] , Flux[det], color = "blue")
    plt.errorbar(Time[det] , Flux[det] , Unc[det] , ls = 'none', color = "blue")
    #plt.scatter(Time[lim] , Flux[lim] , color = "orange" , marker = "v")
    plt.yscale('log')
    plt.xlabel("Time since peak (days)")
    plt.ylabel("Flux (ph / s / cm$^{-2}$)")
    if display():
        
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    params = af.read_parameters(sys.argv[1])
    plot_light_curve(params)