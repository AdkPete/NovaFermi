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

    TS  , Flux , Unc, upper_lim, st, end = read_results(params, mode = "lc")
    
    if len(Flux) == 0:
        return 0
        
    if len(sys.argv) > 2:
        tcut = float(sys.argv[2])
    else:
        tcut = -60
    
    
    #ul2 = ul2[ii]
    ww = (end - st) / (24 * 60 * 60)
    if len(Flux) == 0:
        return 0
    
    Time = []
    Times = (end + st) / 2.0
    for t in Times:
        Time.append(af.met_to_tpeak(t , params))
    Time = np.array(Time)
    
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
    
    det = np.where(TS >= params["ts_lim"])
    lim = np.where(( TS < params["ts_lim"]) & (upper_lim > 0))
    
    ncol = 1 ## Change to 2 for a two-column figure.
    fdim = get_size(244 * ncol)
    fig = plt.figure(figsize = fdim)
    plt.rcParams.update({'font.size': 8})
    plt.scatter(Time[lim] , upper_lim[lim] , color = "orange" , marker = "v")
    plt.errorbar(Time[lim], upper_lim[lim] , xerr = ww[lim] , color = "orange" , ls = "none")
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


def read_results(params, mode):
    '''
    Function to read in required data.
    mode can either be lc, grid, or bck
    '''
    if mode != "lc" and mode != "grid" and mode != "bck":
        print ("Error: mode must be either lc, grid, or bck")
        raise ValueError
    
    fname = params["result_log"]
    if not os.path.exists(fname):
        print ("No results found, exiting")
        return [],[],[],[],[],[]

    ## Get desired start / stop times
    if mode == "lc":
        
        starts, ends, fheaders = af.get_light_curve_bins(params)
    
    elif mode == "grid":
        starts, ends, fheaders = af.get_grid_bins(params)
        
    elif mode == "bck":
        starts, ends, fheaders = af.get_bck_bins(params)
        
    f = open(fname)
    
    TS = []
    Flux = []
    Unc = []
    st = []
    et = []
    upper_lim = []
    for i in f.readlines():
        if "TS" in i:
            continue
        sl = i.split(",")
        stt = float(sl[4])
        ett = float(sl[5])
        if stt not in starts or ett not in ends:
            continue
        ind = starts.index(stt)
        if ends[ind] != ett:
            continue
        
        Flux.append(float(sl[0]))
        Unc.append(float(sl[1]))
        TS.append(float(sl[2]))
        upper_lim.append(float(sl[3]))
        st.append(float(sl[4]))
        et.append(float(sl[5]))
    
    TS = np.array(TS)
    Flux = np.array(Flux)
    Unc = np.array(Unc)
    upper_lim = np.array(upper_lim)
    st = np.array(st)
    et = np.array(et)
    return TS , Flux , Unc , upper_lim , st , et

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
    
    
    TS  , Flux , Unc, upper_lim , st , et = read_results(params, mode = "bck")
    
    if len(TS) == 0:
        print ("No background data found, exiting")
        return 0
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
    table_x = [1 , 4 , 9 , 16 , 20, 25]
    N_old = 0
    for x in table_x:
        Nbin = len(np.where(TS <= x)[0])
        Ntri = len(np.where(TS <= x)[0])
        rows.append([np.sqrt(x) , x ,len(TS) - Ntri, Ntri / len(TS)])
    print (tabulate(rows))
    
    Times = (et + st) / 2.0
    Time = []
    for t in Times:
        Time.append(af.met_to_tpeak(t , params))
    Time = np.array(Time)
    
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
    

    
def TS_Grid(params, return_TS = False, show = False, title = None):
    
    '''
    Plots the results from a TS Grid search
    '''
    

    
    TS  , Flux , Unc, upper_lim , st , et = read_results(params, mode = "grid")
    
    if len(TS) == 0:
        print ("No grid data found, exiting")
        return 0
    
    start=  []
    end = []
    for i in range(len(st)):
        start.append(af.met_to_tpeak(st[i] , params))
        end.append(af.met_to_tpeak(et[i] , params))
    
    start = np.array(start)
    end = np.array(end)
    TSi = list(TS).index(max(TS))
    Time = (st[TSi] + et[TSi]) / 2.0
    Time = af.met_to_tpeak(Time , params)
    print (f"Maximum TS is {max(TS)} at time {Time}")
    #mark1 = plt.Circle(( end[TSi], start[TSi]) , 0.5, fill=False)
    ncol = 1 ## Change to 2 for a two-column figure.
    fdim = get_size(244 * ncol)
    fig = plt.figure(figsize = fdim)
    
    plt.rcParams.update({'font.size': 8})
    plt.rcParams.update({'lines.markersize':2.5})
    # Build square, edge-to-edge cells from the grid coordinates.
    xvalues = np.sort(np.unique(end))
    yvalues = np.sort(np.unique(start))
    xedges = np.r_[xvalues[0] - (xvalues[1] - xvalues[0]) / 2,
                   (xvalues[:-1] + xvalues[1:]) / 2,
                   xvalues[-1] + (xvalues[-1] - xvalues[-2]) / 2]
    yedges = np.r_[yvalues[0] - (yvalues[1] - yvalues[0]) / 2,
                   (yvalues[:-1] + yvalues[1:]) / 2,
                   yvalues[-1] + (yvalues[-1] - yvalues[-2]) / 2]
    TSgrid = np.full((len(yvalues), len(xvalues)), np.nan)
    for x, y, value in zip(end, start, TS):
        TSgrid[np.searchsorted(yvalues, y), np.searchsorted(xvalues, x)] = value
    plt.pcolormesh(xedges, yedges, TSgrid, shading='flat')
    #plt.gca().add_artist(mark1)
    plt.colorbar(label="TS")
    plt.scatter(end[TSi] , start[TSi] , marker = "s", facecolors =  'none',  edgecolors = "black",
                s = plt.rcParams['lines.markersize'] ** 2 * 5)
    plt.ylabel("Start Time (days)")
    plt.xlabel("End Time (days)")
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('both')
    plt.tight_layout()
    plt.title(title)
    plt.savefig(params["figdir"] + "TSGrid.pdf")
    if show:
        plt.show()
    plt.close()
    
    Fluxgrid = np.full((len(yvalues), len(xvalues)), np.nan)
    for x, y, value in zip(end, start, Flux):
        Fluxgrid[np.searchsorted(yvalues, y), np.searchsorted(xvalues, x)] = np.log10(value)
    plt.pcolormesh(xedges, yedges, Fluxgrid, shading='flat')
    #plt.gca().add_artist(mark1)
    plt.colorbar(label="log10(Flux)")
    plt.scatter(end[TSi] , start[TSi] , marker = "s", facecolors =  'none',  edgecolors = "black",
                s = plt.rcParams['lines.markersize'] ** 2 * 5)
    plt.ylabel("Start Time (days)")
    plt.xlabel("End Time (days)")
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('both')
    plt.savefig(params["figdir"] + "FluxGrid.pdf")
    plt.close()
    
    if return_TS:
        return max(TS), Time[TSi]
    
    ## Print some useful stats
    ii = np.where(np.array(TS) >= 9)
    if len(ii[0]) == 0:
        print ("No significant bins detected")
        return max(TS), Time[TSi]
    print ("Maximum flux in a significantly detected bin")
    
    print (np.max(np.array(Flux)[ii]))
    print (start[ii[0][0]],end[ii[0][0]])
    return max(TS), Time[TSi]

if __name__ == "__main__":
    params = af.read_parameters(sys.argv[1])
    plot_TS_search(params)
    plot_light_curve(params)
    TS_hist(params)
    TS_Grid(params)