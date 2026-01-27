'''
Short routine to collect and plot the results of a light curve
generated with analyze_fermi.py
'''

import matplotlib.pyplot as plt
import sys , os
import analyze_fermi as af
import numpy as np
from tabulate import tabulate
import scipy.stats as stats

def get_size(width , fraction = 1.0):
    
    """
    Set figure dimensions to avoid scaling in LaTeX.
    Intended to make figures the right size to fit in 1 column documents
    Allows fonts to match the rest of the document exactly (or to just
    give direct font size control).
    
    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_width_in)

    return fig_dim

def plot_TS_search(params):
    fname = params["tsm_outdir"] + "all_fits.csv"
    if not os.path.exists(fname):
        print ("No TS Search results found, exiting")
        return 0
    f = open(fname)
    
    start = []
    window = []
    TS = []
    for i in f.readlines():
        if "#" in i:
            continue
        sl = i.split(",")
        start.append(float(sl[0]))
        window.append(float(sl[1]))
        TS.append(float(sl[2]))
        
    plt.scatter(start , window , c = TS)
    plt.colorbar(label = "TS")
    plt.xlabel("Start Time (days since peak)")
    plt.ylabel("Window Width (days)")
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('both')
    plt.savefig(params["figdir"] + "Fit_Monitor.pdf")
    plt.close()
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

    Time , TS , Unc , Flux , ul2 , ww = load_data(params , compile_csv)
    if len(Time) == 0:
        return 0
        
    if len(sys.argv) > 2:
        tcut = float(sys.argv[2])
    else:
        tcut = -60
        
    ii = np.where(Time > tcut)
    Time = Time[ii]
    TS = TS[ii]
    Unc = Unc[ii]
    Flux = Flux[ii]
    ul2 = ul2[ii]
    ww = ww[ii]
    if len(Time) == 0:
        return 0
    
    ncol = 1 ## Change to 2 for a two-column figure.
    fdim = get_size(244 * ncol)
    fig = plt.figure(figsize = fdim)
    
    plt.rcParams.update({'font.size': 8})
    plt.scatter(Time , TS)
    plt.axhline(4 , ls = ":" , color = "blue")
    plt.xlabel("Time Since Eruption (days)")
    plt.ylabel("TS")
    plt.tight_layout()
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('both')
    plt.savefig(params["figdir"] + "TSFig.pdf")
    if display:
        plt.show()
    plt.close()
    
    det = np.where(TS >= 4)
    lim = np.where(TS < 4)

    ncol = 1 ## Change to 2 for a two-column figure.
    fdim = get_size(244 * ncol)
    fig = plt.figure(figsize = fdim)
    plt.rcParams.update({'font.size': 8})
    plt.scatter(Time[lim] , Flux[lim] , color = "orange" , marker = "v")
    plt.errorbar(Time[lim], Flux[lim] , xerr = ww[lim] , color = "orange" , ls = "none")
    plt.scatter(Time[det] , Flux[det], color = "blue")
    plt.errorbar(Time[det] , Flux[det] , yerr = Unc[det] , xerr = ww[det] , ls = 'none', color = "blue")
    plt.yscale('log')
    plt.xlabel("Time Since Eruption (days)")
    plt.ylabel("Flux (ph / s / cm$^{2}$)")
    plt.tight_layout()
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('both')
    if display:
        plt.savefig(params["figdir"] + "LC.pdf")
        plt.show()
    else:
        plt.savefig(params["figdir"] + "LC.pdf")
        plt.close()

    ncol = 1 ## Change to 2 for a two-column figure.
    fdim = get_size(244 * ncol)
    fig = plt.figure(figsize = fdim)
    plt.rcParams.update({'font.size': 8})
    lim2 = np.where( ( TS < 4 ) & (ul2 > 0) ) 
    plt.subplot(2,1,1)
    plt.scatter(Time[det] , Flux[det], color = "blue")
    plt.errorbar(Time[det] , Flux[det] , yerr = Unc[det] , xerr = ww[det] , ls = 'none', color = "blue")
    plt.scatter(Time[lim] , Flux[lim] , color = "orange" , marker = "v" , alpha = 0.75)
    plt.scatter(Time[lim2] , ul2[lim2] , color = "green" , marker = "x" , alpha = 0.75)
    
    plt.ylabel("Flux (ph / s / cm$^{-2}$)")
    plt.yscale('log')
    plt.tight_layout()
    
    print (ul2)
    plt.subplot(2,1,2)
    plt.scatter(Time[lim2] , Flux[lim2] - ul2[lim2])
    plt.xlabel("Time Since Eruption (days)")
    plt.ylabel("Residual")

    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('both')
    plt.savefig(params["figdir"] + "ULS.pdf")
    plt.close()

def load_data(params , compile_csv = None):
    if compile_csv is None:
        compiled_csv = params["lc_outdir"] + params["name"] + f"_{int(params['window'])}_lcdata.csv"
        if not os.path.exists(compiled_csv):
            print ("No LC data found, exiting")
            return [],[],[],[],[],[]
    Time = []
    TS = []
    Flux = []
    Unc = []
    ul2 = []
    f = open(compiled_csv)
    for i in f.readlines():
        if "TS" in i:
            continue
        sl = i.split(",")
        Time.append(float(sl[0]))
        TS.append(float(sl[2]))
        Flux.append(float(sl[3]))
        Unc.append(float(sl[4]))

        ul2.append(float(sl[5]))
        

    Time = np.array(Time)
    TS = np.array(TS)
    Unc = np.array(Unc)
    Flux = np.array(Flux)
    ul2 = np.array(ul2)
    ww = np.array([params["window"]/2.0] * len(Flux))
    ii = np.where(Flux > 0)
    if np.min(Flux) < 0:
        print ("Warning: At least one Likelihood calculation Failed, F < 0")
    return Time[ii] , TS[ii] , Unc[ii] , Flux[ii] , ul2[ii] , ww[ii]

def TS_hist(params, compile_csv = None):
    
    '''
    Function to display a histogram of TS values, intended for testing the
    significance of a detection. Simply plots a histogram of the TS values
    for all bins more than 60 days before peak. If no such data exists,
    simply returns 0.
    Also, will print out some statistics relevant for testing if the 
    TS values are behaving as expected.
    '''
    
    Time , TS , Unc , Flux , ul2 , ww = load_data(params , compile_csv)
    if len(Time) == 0:
        return
    if np.min(Time) >= -60:
        ## No suitable background data
        print ("No background runs detected")
        return 0
    
    back = np.where(Time < -60)
    
    ncol = 1 ## Change to 2 for a two-column figure.
    fdim = get_size(244 * ncol)
    fig = plt.figure(figsize = fdim)
    plt.rcParams.update({'font.size': 8})
    plt.hist(TS[back] , bins = 10)
    plt.axvline(4 , color = "orange" , ls = "--")
    plt.yscale("log")
    plt.xlabel("Test Statistic")
    plt.ylabel("Number of Trials")
    plt.tight_layout()
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('both')
    plt.savefig(params["figdir"] + "TS_Hist.pdf")
    plt.close()
    
    df = 3
    xarr2 = np.linspace(np.min(0) , 35 , 1000)
    chi2arr = stats.chi2.pdf(xarr2,df)
    
    ncol = 1 ## Change to 2 for a two-column figure.
    fdim = get_size(244 * ncol)
    fig = plt.figure(figsize = fdim)
    plt.rcParams.update({'font.size': 8})
    plt.hist(TS[back] , bins = 10, density = True)
    plt.plot(xarr2 , chi2arr , color = "orange")
    plt.axvline(4 , color = "orange" , ls = "--")
    #plt.xlabel("Test Statistic")
    plt.ylabel("Number of Trials")
    plt.tight_layout()
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('both')
    plt.savefig(params["figdir"] + "TS_Hist_wpdf.pdf")
    plt.close()
    
    
    ncol = 1 ## Change to 2 for a two-column figure.
    fdim = get_size(244 * ncol)
    fig = plt.figure(figsize = fdim)
    plt.rcParams.update({'font.size': 8})
    ## Cumulative distribution of TS values
    xvs = np.linspace(0 , np.max(TS[back]) , 10000)
    yvs = []
    for i in xvs:
        yvs.append(len(np.where(TS[back] <= i)[0]) / len(TS[back]))
    plt.plot(xvs , yvs)
    plt.xlabel("Test Statistic")
    plt.ylabel("Cumulative Distribution")
    plt.tight_layout()
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('both')
    plt.savefig(params["figdir"] + "TS_cdf.pdf")
    plt.close()
    

    print (f"Statistics based on {len(TS[back])} Trials")
    rows = [["Sigma" , "TS" , "Number" , "Fraction (Cumulative)"]]
    table_x = [1 , 4 , 9 , 16 , 25]
    N_old = 0
    for x in table_x:
        Nbin = len(np.where(TS[back] <= x)[0])
        Ntri = len(np.where(TS[back] <= x)[0])
        rows.append([np.sqrt(x) , x ,Ntri, Ntri / len(TS[back])])
    print (tabulate(rows))
    
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
        output = params["lc_outdir"] + params["name"] + f"_{int(params['window'])}_lcdata.csv"
    
    
    dir = params["lc_outdir"] 
    Flux = []
    Unc = []
    Time = []
    TS = []
    METs = []
    ULS = []
    for i in os.listdir(dir):
        fname = os.path.join(dir , i)
        if ".csv" not in fname or "mp" not in fname or str(params["window"]) not in fname or str(params["lcstep"]) not in fname:
            continue
        elif "grid" in fname:
            continue
        f = open(fname)
        for line in f.readlines():
            split_line = line.split(",")
            Flux.append(float(split_line[0]))
            Unc.append(float(split_line[1]))
            TS.append(float(split_line[2]))
            MET = float(split_line[3])
            ul2 = float(split_line[4])
            ULS.append(ul2)
            METs.append(MET)
            tpeak = af.met_to_tpeak(MET , params)
            Time.append(tpeak)
            break
        
    METs = np.array(METs)
    TS = np.array(TS)
    Flux = np.array(Flux)
    Unc = np.array(Unc)
    Time = np.array(Time)
    ULS = np.array(ULS)
    
    if len(METs) == 0:
        print ("No LC Data Found, exiting now")
        return 0
    isort = np.argsort(Time)
    out_file = open(output, "w")
    
    header = "Time since peak (days),Fermi MET (seconds),TS,Flux,Flux"
    header += " Uncertainty\n"
    out_file.write(header)
    
    for ind in isort:
        csv_line = f"{Time[ind]},{METs[ind]},{TS[ind]},{Flux[ind]},{Unc[ind]},{ULS[ind]}"
        out_file.write(csv_line + "\n")
    out_file.close()
    
def TS_Grid(params):
    
    '''
    Plots the results from a TS Grid search
    '''
    
    dir = params["grid_outdir"]
    start = []
    end = []
    TS = []
    Flux = []
    Flux_err = []
   
    for i in os.listdir(dir):
        fname = os.path.join(dir , i)
        
        if ".csv" not in i:
            continue
        elif "grid" not in i:
            continue
        f = open(fname)
        print (fname)
        
        sn1 = i.split("grid")[2]
        sn2 = sn1.split("_")
        start.append(float(sn2[1]))
        end.append(float(sn2[2].split(".")[0]))
        
        for line in f.readlines():
            split_line = line.split(",")
            Flux.append(float(split_line[0]))
            Flux_err.append(float(split_line[1]))
            TS.append(float(split_line[2]))
    if len(TS) == 0:
        print ("No Grid Data Found, Exiting")
        return 0

    TSi = TS.index(max(TS))
    print (f"Maximum TS is {max(TS)}")
    #mark1 = plt.Circle(( end[TSi], start[TSi]) , 0.5, fill=False)
    ncol = 1 ## Change to 2 for a two-column figure.
    fdim = get_size(244 * ncol)
    fig = plt.figure(figsize = fdim)
    
    plt.rcParams.update({'font.size': 8})
    plt.rcParams.update({'lines.markersize':2.5})
    plt.scatter(end , start, c = TS )
    #plt.gca().add_artist(mark1)
    plt.colorbar(label="TS")
    plt.scatter(end[TSi] , start[TSi] , marker = "o", facecolors =  'none',  edgecolors = "black",
                s = plt.rcParams['lines.markersize'] ** 2 * 5)
    plt.ylabel("Start Time (days)")
    plt.xlabel("End Time (days)")
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('both')
    plt.tight_layout()
    plt.savefig(params["figdir"] + "TSGrid.pdf")
    plt.show()
    
    plt.scatter(end , start, c = np.log10(np.array(Flux)))
    #plt.gca().add_artist(mark1)
    plt.colorbar(label="log10(Flux)")
    plt.scatter(end[TSi] , start[TSi] , marker = "o", facecolors =  'none',  edgecolors = "black",
                s = plt.rcParams['lines.markersize'] ** 2 * 5)
    plt.ylabel("Start Time (days)")
    plt.xlabel("End Time (days)")
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('both')
    plt.savefig(params["figdir"] + "FluxGrid.pdf")
    plt.show()
if __name__ == "__main__":
    params = af.read_parameters(sys.argv[1])
    plot_TS_search(params)
    compile_data(params)
    plot_light_curve(params)
    TS_hist(params)
    TS_Grid(params)