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

    TS , Unc , Flux , Time, st, end = read_results(params["lc_logfile"])
    
    if len(Time) == 0:
        return 0
        
    if len(sys.argv) > 2:
        tcut = float(sys.argv[2])
    else:
        tcut = -60
    
    
    #ul2 = ul2[ii]
    ww = (end - st) / (24 * 60 * 60)
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
    return 0

    # todo remove this code since it no longer gets called
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

def compile_bck_data(params):
    '''
    Utility function to compile all of the multi-processing logs into 
    a singular csv file.
    
    Parameters
    __________
    
    params : dict : parameter dict from read_parameters
    
    Returns
    _______
    None
    
    '''
    
    if "bck_outdir" not in params.keys():
        return 0

    output = params["bck_outdir"] + params["name"] + "bck_data.csv"
    
    
    dir = params["bck_outdir"] 
    Flux = []
    Unc = []
    Time = []
    TS = []
    METs = []
    
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
            METs.append(MET)
            tpeak = af.met_to_tpeak(MET , params)
            Time.append(tpeak)
            break
        
    METs = np.array(METs)
    TS = np.array(TS)
    Flux = np.array(Flux)
    Unc = np.array(Unc)
    Time = np.array(Time)
    
    
    if len(METs) == 0:
        print ("No LC Data Found, exiting now")
        return 0
    isort = np.argsort(Time)
    out_file = open(output, "w")
    
    header = "Time since peak (days),Fermi MET (seconds),TS,Flux,Flux"
    header += " Uncertainty\n"
    out_file.write(header)
    
    for ind in isort:
        csv_line = f"{Time[ind]},{METs[ind]},{TS[ind]},{Flux[ind]},{Unc[ind]}"#//,{ULS[ind]}"
        out_file.write(csv_line + "\n")
    out_file.close()
    
    return 0

def load_bck_data(params):
    
    '''
    Function to load background data
    '''
    
    return 0
def load_data(params , compiled_csv = None):
    if compiled_csv is None:
        compiled_csv = params["lc_outdir"] + params["name"] + f"_{int(params['window'])}_lcdata.csv"
        if not os.path.exists(compiled_csv):
            print ("No LC data found, exiting")
            return [],[],[],[],[]
    Time = []
    TS = []
    Flux = []
    Unc = []
    #ul2 = []
    f = open(compiled_csv)
    for i in f.readlines():
        if "TS" in i:
            continue
        sl = i.split(",")
        Time.append(float(sl[0]))
        TS.append(float(sl[2]))
        Flux.append(float(sl[3]))
        Unc.append(float(sl[4]))

        #ul2.append(float(sl[5]))
        

    Time = np.array(Time)
    TS = np.array(TS)
    Unc = np.array(Unc)
    Flux = np.array(Flux)
    #ul2 = np.array(ul2)
    ww = np.array([params["window"]/2.0] * len(Flux))
    ii = np.where(Flux > 0)
    if np.min(Flux) < 0:
        print ("Warning: At least one Likelihood calculation Failed, F < 0")
    return Time[ii] , TS[ii] , Unc[ii] , Flux[ii] , ww[ii]

def read_results(fname):
    
    if not os.path.exists(fname):
        print ("No results found, exiting")
        return [],[],[],[],[],[]
    f = open(fname)
    
    TS = []
    Flux = []
    Unc = []
    Time = []
    st = []
    et = []
    
    for i in f.readlines():
        if "TS" in i:
            continue
        sl = i.split(",")
        
        Flux.append(float(sl[0]))
        Unc.append(float(sl[1]))
        TS.append(float(sl[2]))
        Time.append(float(sl[3]))
        st.append(float(sl[4]))
        et.append(float(sl[5]))
    
    TS = np.array(TS)
    Flux = np.array(Flux)
    Unc = np.array(Unc)
    Time = np.array(Time)
    st = np.array(st)
    et = np.array(et)
    return TS , Flux , Unc , Time , st , et

def TS_hist(params, compile_csv = None):
    
    '''
    Function to display a histogram of TS values, intended for testing the
    significance of a detection. Simply plots a histogram of the TS values
    for all bins more than 60 days before peak. If no such data exists,
    simply returns 0.
    Also, will print out some statistics relevant for testing if the 
    TS values are behaving as expected.
    '''
    
    if "bck_outdir" not in params.keys():
        print ("No Background directory specified, exiting")
        return 0
    

    back_file = params["bck_logfile"]
    
    if not os.path.exists(back_file):
        print ("Error: No Background data found, exiting")
        return 0
    
    TS , Unc , Flux , Time, st , et = read_results(back_file)
    
    
    ncol = 1 ## Change to 2 for a two-column figure.
    fdim = get_size(244 * ncol)
    fig = plt.figure(figsize = fdim)
    plt.rcParams.update({'font.size': 8})
    plt.hist(TS , bins = 20)
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
    plt.hist(TS , bins = 10, density = True)
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
    xvs = np.linspace(0 , np.max(TS) , 10000)
    yvs = []
    for i in xvs:
        yvs.append(len(np.where(TS <= i)[0]) / len(TS))
    plt.plot(xvs , yvs)
    plt.xlabel("Test Statistic")
    plt.ylabel("Cumulative Distribution")
    plt.tight_layout()
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('both')
    plt.savefig(params["figdir"] + "TS_cdf.pdf")
    plt.close()
    

    print (f"Statistics based on {len(TS)} Trials")
    rows = [["Sigma" , "TS" , "Number" , "Fraction (Cumulative)"]]
    table_x = [1 , 4 , 9 , 16 , 25]
    N_old = 0
    for x in table_x:
        Nbin = len(np.where(TS <= x)[0])
        Ntri = len(np.where(TS <= x)[0])
        rows.append([np.sqrt(x) , x ,len(TS) - Ntri, Ntri / len(TS)])
    print (tabulate(rows))
    
    ## Plot a light curve out of the background data.
    plt.figure()
    plt.scatter(Time , Flux)
    plt.errorbar(Time , Flux , yerr = Unc , ls = "none")
    plt.xlabel("Time Since Peak (days)")
    plt.ylabel("Flux (ph / s / cm$^{-2}$)")
    plt.yscale("log")
    plt.tight_layout()
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('both')
    plt.savefig(params["figdir"] + "Bck_LC.pdf")
    plt.close()
    
    ## TS Curve
    plt.figure()
    plt.scatter(Time , TS)
    plt.axhline(4 , ls = ":" , color = "blue")
    plt.ylabel("Test Statistic")
    plt.xlabel("Time Since Peak (days)")
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('both')
    plt.tight_layout()
    plt.savefig(params["figdir"] + "Bck_TS.pdf")
    plt.close()
    
    ## Sanity check to look for overlap between the background bins:
    overlap = False
    for i in range(len(st)):
        for j in range(len(st)):
            if i == j:
                continue
            if st[i] < st[j]:
                if et[i] > st[j]:
                    print (f"Overlap between bins {i} and {j}")
                    print (st[i] , et[i] , st[j] , et[j])
                    overlap = True
    if not overlap:
        print ("No Overlap between background bins detected")
        
    bin_widths = (et - st) / (24 * 60 * 60)
    plt.scatter(Time, TS)
    plt.errorbar(Time , TS , xerr = 0.5 * bin_widths , ls = "none")
    plt.xlabel("Time Since Peak (days)")
    plt.ylabel("Test Statistic")
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('both')
    plt.tight_layout()
    plt.show()
    
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
    # // ULS = []
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
            #ul2 = float(split_line[4])
            #ULS.append(ul2)
            METs.append(MET)
            tpeak = af.met_to_tpeak(MET , params)
            Time.append(tpeak)
            break
        
    METs = np.array(METs)
    TS = np.array(TS)
    Flux = np.array(Flux)
    Unc = np.array(Unc)
    Time = np.array(Time)
    # // ULS = np.array(ULS)
    
    if len(METs) == 0:
        print ("No LC Data Found, exiting now")
        return 0
    isort = np.argsort(Time)
    out_file = open(output, "w")
    
    header = "Time since peak (days),Fermi MET (seconds),TS,Flux,Flux"
    header += " Uncertainty\n"
    out_file.write(header)
    
    for ind in isort:
        csv_line = f"{Time[ind]},{METs[ind]},{TS[ind]},{Flux[ind]},{Unc[ind]}"#//,{ULS[ind]}"
        out_file.write(csv_line + "\n")
    out_file.close()
    
def TS_Grid(params):
    
    '''
    Plots the results from a TS Grid search
    '''
    
    if not os.path.exists(params["grid_logfile"]):
        print ("No TS Grid results found, exiting")
        return 0
    TS , Unc , Flux , Time, st , et = read_results(params["grid_logfile"])
    
    start=  []
    end = []
    for i in range(len(Time)):
        start.append(af.met_to_tpeak(st[i] , params))
        end.append(af.met_to_tpeak(et[i] , params))
    
    start = np.array(start)
    end = np.array(end)
    TSi = list(TS).index(max(TS))
    print (f"Maximum TS is {max(TS)} at time {Time[TSi]}")
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
    plt.close()
    
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
    plt.close()
    
    ## Print some useful stats
    ii = np.where(np.array(TS) >= 9)
    if len(ii[0]) == 0:
        print ("No significant bins detected")
        return 0
    print ("Maximum flux in a significantly detected bin")
    
    print (np.max(np.array(Flux)[ii]))
    print (start[ii[0][0]],end[ii[0][0]])
if __name__ == "__main__":
    params = af.read_parameters(sys.argv[1])
    plot_TS_search(params)
    compile_data(params)
    compile_bck_data(params) 
    plot_light_curve(params)
    TS_hist(params)
    TS_Grid(params)