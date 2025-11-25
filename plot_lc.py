'''
Short routine to collect and plot the results of a light curve
generated with analyze_fermi.py
'''

import matplotlib.pyplot as plt
import sys , os
import analyze_fermi as af
import numpy as np


def plot_light_curve(params, display=False, compile_csv = None):
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
    compile_csv : string : Name of input csv file
    Returns
    _______
    None
    '''
    if compile_csv is None:
        compiled_csv = params["name"] + f"_{int(params['window'])}_lcdata.csv"
        
    Time = []
    TS = []
    Flux = []
    Unc = []
    
    f = open(compiled_csv)
    for i in f.readlines():
        if "TS" in i:
            continue
        sl = i.split(",")
        Time.append(float(sl[0]))
        TS.append(float(sl[2]))
        Flux.append(float(sl[3]))
        Unc.append(float(sl[4]))
        
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
    if display:
        
        plt.show()
    else:
        plt.close()

def compile_data(params, output=None):
    '''
    Utility function to compile all of the multi-processing logs into 
    a singular csv file.
    
    Parameters
    __________
    
    params : dict : parameter dict from read_parameters
    output : string : name of output csv file
    
    Returns
    _______
    None
    
    '''
    if output is None:
        output = params["name"] + f"_{int(params['window'])}_lcdata.csv"
    
    
    dir = "./"
    Flux = []
    Unc = []
    Time = []
    TS = []
    METs = []
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
            METs.append(MET)
            tpeak = af.met_to_tpeak(MET , params)
            Time.append(tpeak)
            break
        
    METs = np.array(METs)
    TS = np.array(TS)
    Flux = np.array(Flux)
    Unc = np.array(Unc)
    Time = np.array(Time)
    
    isort = np.argsort(Time)
    out_file = open(output, "w")
    
    header = "Time since peak (days),Fermi MET (seconds),TS,Flux,Flux"
    header += " Uncertainty\n"
    out_file.write(header)
    
    for ind in isort:
        csv_line = f"{Time[ind]},{METs[ind]},{TS[ind]},{Flux[ind]},{Unc[ind]}"
        out_file.write(csv_line + "\n")
    out_file.close()
    
if __name__ == "__main__":
    params = af.read_parameters(sys.argv[1])
    compile_data(params)
    plot_light_curve(params)