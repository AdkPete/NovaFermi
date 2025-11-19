'''
Written by Peter Craig (craigpe1@msu.edu)
Last updated 11/15/25

Runs Fermi data analysis for novae
Includes functions to:
1. Run a binned likelihood analysis to check TS value for a nova
2. Generate a light curve
3. Search for maximum TS value
'''


import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os, sys
import datetime as dtime
import tabulate
import multiprocessing as mp

import gt_apps as my_apps
from GtApp import GtApp

import pyLikelihood
from BinnedAnalysis import *


### Some global variables: May need to update to run on your systems

## Directory containing diffuse emission and catalog files
cal_dir = "/Users/Peter/Documents/Research/Novae/Fermi/Diffuse/"

### Start with some useful background / setup functions

def setup_events_file(clobber=False):
    
    '''
    Simple function to setup files listing all data, and identifies the
    spacecraft file. Just leave all data / spacecraft files in current
    directory and this will prep data as need be.
    
    Parameters
    __________
    clobber: boolean : If true, will overwite any existing event list
    
    Returns
    _______
    infile : string : name of event filename
    scfile : string : name of spacecraft file
    
    '''
    event_file = "events.txt"

    if os.path.exists(event_file) and clobber:
        os.remove(event_file)
        
    if not os.path.exists(event_file):
        f = open(event_file , "w")
        output = ""
        for i in os.listdir():
            if "_PH" in i and ".fits" in i:
                output += i + "\n"
        f.write(output[:-1])
        f.close()
    for i in os.listdir():
        if "_SC" in i and ".fits" in i:
            scfile = i

    
    infile = '@events.txt'
    return infile , scfile

def get_met(date_time):
    '''
    Function to compute Fermi MET
    
    Parameters
    __________
    date_time : datetime object : Should contain time you'd like to 
        convert to MET
    Returns
    _______
    MET : float : Fermi MET in seconds
    
    '''
    dtref = dtime.datetime(year=2001, month = 1, day=1, hour = 0, minute=0,
                        second=0 ,  tzinfo=dtime.timezone.utc)
    MET = date_time - dtref
    return MET.total_seconds()

def read_parameters(pfile):
    '''
    Function to read analysis parameter file
    All parameter options should get set in this file
    See template parameter file for available parameters
    
    Parameters
    __________
    pfile : string : name of parameter file
    
    Returns
    _______
    params : dict : contains all analysis options and parameters
    '''
    
    f = open(pfile)
    params = {}
    for i in f.readlines():
        sl = i.split("\t")
        while "" in sl:
            sl.remove("")
        if i[0] == "#" or len(sl) == 1:
            continue
        key = sl[0].strip()
        
        ## First, handle special cases
        if key == "nproc" or key == "N_ebin" or key == "TSPix": 
            ## Should be integer
            params[key] = int(sl[1])
            
        elif len(sl[1].split("-")) == 3 & len(sl[1].split("-")) == 3:
            ## This is a time entry
            ## Will convert to MET

            date = sl[1].split(" ")[0]
            time = sl[1].split(" ")[1]
            year = int(date.split("-")[0])
            month = int(date.split("-")[1])
            day = int(date.split("-")[2])
            hour = int(time.split(":")[0])
            minute = int(time.split(":")[1])
            second = int(time.split(":")[2])
            stime = dtime.datetime(year=year,month=month,day=day,hour=hour,
                        minute=minute,second=second,tzinfo=dtime.timezone.utc)
            MET = get_met(stime)
            params[key] = MET
        
        elif sl[1].strip().lower() == "now":
            ## Get MET of right now
            MET = get_met(dtime.datetime.now(tz=dtime.timezone.utc))
            params[key] = MET
        elif "none" in sl[1].lower() or "N/A" in sl[1].lower():
            continue
        ## General handling of other params

        ## Boolean options first
        elif sl[1].strip().lower() == "yes":
            params[key] = True
        
        elif sl[1].strip().lower() == "no":
            params[key] = False
        
        ## Floats / strings last

        else:
            try:
                params[key] = float(sl[1])
            except:
                params[key] = sl[1].strip()
    if "infile" not in params.keys() or "scfile" not in params.keys():
        ## If data is not specified, auto-detect data files.
        infile , scfile = setup_events_file(clobber=False)
        params["infile"] = infile
        params["scfile"] = scfile
        
    if "input_model" not in params.keys():
        params["input_model"] = params["name"] + "_input_model.xml"
        
    return params

def print_params(params):
    
    '''
    Simple function to print out our parameters
    
    Parameters
    __________
    params : dict : parameter dictionary from read_parameters
    
    Returns
    _______
    None
    '''
    print ("\nParameters for Fermi Analysis:")
    rows = []
    for key in params.keys():
        rows.append([key,str(params[key])])
    print (tabulate.tabulate(rows) + "\n")

def gen_model(params, clobber):
    '''
    Function to create an input model file
    
        
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    clobber : boolean : If true, overwrite existing files
    
    Returns
    ________
    None
    '''
    

    dfname = cal_dir + "gll_psc_v32.xml"
    gti = "{source}{fheader}_filtered_gti.fits"
    model_fname = params["input_model"]
    
    if os.path.exists(model_fname) and not clobber:
        return 0
    xml_command = f' make4FGLxml {dfname} --event_file {gti} --output_name'
    xml_command += f'{model_fname} --free_radius 5.0 --norms_free_only'
    xml_command += f'True --sigma_to_free 25 --variable_free True'
    subprocess.run(xml_command, shell=True)
    
    input("Please edit input_model to include source models. Then, hit enter")
    
def data_selection(params, tstart, tend, clobber, fheader):
    '''
    Function to run the data selection
    Runs the gtselect and mktime FermiTools tasks to run the data selection
    Will produce the following FITS files:
    source_filtered.fits (gtselect)
    source_filtered_gti.fits (mktime)
    
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    tstart : float : time (MET) for data start
    tstart : float : time (MET) for data end
    clobber : boolean : If true, overwrite existing files
    fheader : string : Unique ID added to file names to avoid name conflicts
    
    Returns
    ________
    None
    '''
    
    ##gtselect first

    out_name = f'{params["name"]}{fheader}_filtered.fits'
    my_apps.filter['evclass'] = 128
    my_apps.filter['evtype'] = 3
    my_apps.filter['ra'] = params["ra"]
    my_apps.filter['dec'] = params["dec"]
    my_apps.filter['rad'] = params["roi"]
    my_apps.filter['emin'] = params["emin"]
    my_apps.filter['emax'] = params["emax"]
    my_apps.filter['zmax'] = 90
    my_apps.filter['tmin'] = tstart
    my_apps.filter['tmax'] = tend
    my_apps.filter['infile'] = params["infile"]
    my_apps.filter['outfile'] = out_name

    ## Run gtselect

    if not os.path.exists(out_name) or clobber:
        my_apps.filter.run()
    
    gtiname = f'{params["name"]}{fheader}_filtered_gti.fits'
    my_apps.maketime['scfile'] = params["scfile"]
    my_apps.maketime['filter'] = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'
    my_apps.maketime['roicut'] = 'no'
    my_apps.maketime['evfile'] = f'{params["name"]}{fheader}_filtered.fits'
    my_apps.maketime['outfile'] = gtiname

    if not os.path.exists(gtiname) or clobber:
        my_apps.maketime.run()

def lt_exp_maps(params , clobber , fheader):
    '''
    Generates livetime cubes and exposure maps for binned likelihood
    Fermi analysis
    
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    clobber : boolean : If true, overwrite existing files
    fheader : string : Unique ID added to file names to avoid name conflicts
    
    Returns
    ________
    None
    '''
    
    ## Compute the LiveTime Cube
    ltcube = f'{params["name"]}{fheader}_ltCube.fits'

    my_apps.expCube['evfile'] = f'{params["name"]}{fheader}_filtered_gti.fits'
    my_apps.expCube['scfile'] = params["scfile"]
    my_apps.expCube['outfile'] = ltcube
    my_apps.expCube['zmax'] = 90
    my_apps.expCube['dcostheta'] = 0.025
    my_apps.expCube['binsz'] = 1

    if not os.path.exists(ltcube) or clobber:
        my_apps.expCube.run()


    ## Build Exposure Map
    expmap = f'{params["name"]}{fheader}_BinnedExpMap.fits'

    expCube2= GtApp('gtexpcube2','Likelihood')

    expCube2['infile'] = ltcube
    expCube2['cmap'] = 'none'
    expCube2['outfile'] = expmap
    expCube2['irfs'] = 'P8R3_SOURCE_V3'
    expCube2['evtype'] = '3'
    expCube2['nxpix'] = int(360/params["pix_sc"])
    expCube2['nypix'] = int(180/params["pix_sc"])
    expCube2['binsz'] = params["pix_sc"]
    expCube2['coordsys'] = 'CEL'
    expCube2['xref'] = params["ra"]
    expCube2['yref'] = params["dec"]
    expCube2['axisrot'] = 0
    expCube2['proj'] = 'AIT'
    expCube2['ebinalg'] = 'LOG'
    expCube2['emin'] = params["emin"]
    expCube2['emax'] = params["emax"]
    expCube2['enumbins'] = params["N_ebin"]

    if not os.path.exists(expmap) or clobber:
        expCube2.run()
        
def gen_srcmap(params, clobber, fheader):
    '''
    Function to generate source maps for binned likelihood analysis
    
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    clobber : boolean : If true, overwrite existing files
    fheader : string : Unique ID added to file names to avoid name conflicts
    
    Returns
    ________
    None
    '''
    src_name = f'{params["name"]}{fheader}_srcmap.fits'
    my_apps.srcMaps['expcube'] = f'{params["name"]}{fheader}_ltcube.fits'
    my_apps.srcMaps['cmap'] = f'{params["name"]}{fheader}_filtered_ccube.fits'
    my_apps.srcMaps['srcmdl'] = params["input_model"]
    my_apps.srcMaps['bexpmap'] = f'{params["name"]}{fheader}_BinnedExpMap.fits'
    my_apps.srcMaps['outfile'] = src_name
    my_apps.srcMaps['irfs'] = 'P8R3_SOURCE_V3'
    my_apps.srcMaps['evtype'] = '3'

    if not os.path.exists(src_name) or clobber:
        my_apps.srcMaps.run()
        
def bin_data(params, clobber, fheader):
    '''
    Function to generate counts maps / cubes
    
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    clobber : boolean : If true, overwrite existing files
    fheader : string : Unique ID added to file names to avoid name conflicts
    
    Returns
    ________
    None
    '''
    
    my_apps.evtbin['evfile'] = f'{params["name"]}{fheader}_filtered_gti.fits'
    my_apps.evtbin['outfile'] = f'{params["name"]}{fheader}_filtered_cmap.fits'
    my_apps.evtbin['scfile'] = params["scfile"]
    my_apps.evtbin['algorithm'] = 'CMAP'
    my_apps.evtbin['nxpix'] = int(params["roi"] * 2/ params["pix_sc"])
    my_apps.evtbin['nypix'] = int(params["roi"]  * 2/ params["pix_sc"])
    my_apps.evtbin['binsz'] = params["pix_sc"]
    my_apps.evtbin['coordsys'] = 'CEL'
    my_apps.evtbin['xref'] = params["ra"]
    my_apps.evtbin['yref'] = params["dec"]
    my_apps.evtbin['axisrot'] = 0
    my_apps.evtbin['proj'] = 'AIT'
    my_apps.evtbin['ebinalg'] = 'LOG'
    my_apps.evtbin['emin'] = params["emin"]
    my_apps.evtbin['emax'] = params["emax"]
    my_apps.evtbin['enumbins'] = params["N_ebin"]

    if not os.path.exists(f'{params["name"]}{fheader}_filtered_cmap.fits') or clobber:
        my_apps.evtbin.run()

    
    ## Make CCUBE while we are at it:

    npix = int(( np.sqrt(2) * params["roi"] / params["pix_sc"] ))
    my_apps.evtbin['evfile'] = f'{params["name"]}{fheader}_filtered_gti.fits'
    my_apps.evtbin['outfile'] = f'{params["name"]}{fheader}_filtered_ccube.fits'
    my_apps.evtbin['scfile'] = params["scfile"]
    my_apps.evtbin['algorithm'] = 'CCUBE'
    my_apps.evtbin['nxpix'] = npix
    my_apps.evtbin['nypix'] = npix
    my_apps.evtbin['binsz'] = params["pix_sc"]
    my_apps.evtbin['coordsys'] = 'CEL'
    my_apps.evtbin['xref'] = params["ra"]
    my_apps.evtbin['yref'] = params["dec"]
    my_apps.evtbin['axisrot'] = 0
    my_apps.evtbin['proj'] = 'AIT'
    my_apps.evtbin['ebinalg'] = 'LOG'
    my_apps.evtbin['emin'] = params["emin"]
    my_apps.evtbin['emax'] = params["emax"]
    my_apps.evtbin['enumbins'] = params["N_ebin"]

    if not os.path.exists(f'{params["name"]}{fheader}_filtered_ccube.fits') or clobber:
        my_apps.evtbin.run()

def fit_model(params, fheader, get_like, inmod = "No" , opt = 'NewMINUIT'):
    '''
    Function to run the model fitting steps
    This version is the recommended fitting process
    First runs a quick analysis with DRMNFB to get close to the approx sltn.
    Follows this with a NewMINUIT optimization run to finalize the model.
    
    
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    fheader : string : Unique ID added to file names to avoid name conflicts
    get_like : boolean : If true, return logL
    Returns
    ________
    None
    '''
    
    if inmod == "No":
        inmod =params["input_model"]
        
    src_name = f'{params["name"]}{fheader}_srcmap.fits'
    drmnfb_comm = f'gtlike statistic=BINNED cmap={src_name} '
    drmnfb_comm += f'bexpmap={params["name"]}{fheader}_BinnedExpMap.fits '
    drmnfb_comm += f'expcube={params["name"]}{fheader}_ltCube.fits '
    drmnfb_comm += f'srcmdl={inmod} irfs=CALDB'
    drmnfb_comm += f' optimizer=DRMNFB sfile=temp{fheader}.xml '
    
    subprocess.run(drmnfb_comm,shell=True)
    
    obs = BinnedObs(srcMaps=src_name,
            binnedExpMap=f'{params["name"]}{fheader}_BinnedExpMap.fits',
            expCube=f'{params["name"]}{fheader}_ltcube.fits',irfs='P8R3_SOURCE_V3')
    like = BinnedAnalysis(obs,f'temp{fheader}.xml',optimizer=opt)
    likeobj=pyLike.NewMinuit(like.logLike)

    try:
        like.tol = 0.0001
        res = like.fit(verbosity=1,covar=True,optObject=likeobj)
    except:
        try:
            like.tol = 0.01
            res = like.fit(verbosity=1,covar=True,optObject=likeobj)
        except:
            like = BinnedAnalysis(obs,inmod,optimizer="DRMNFB")
            likeobj=pyLike.NewMinuit(like.logLike)
            like.tol = 0.01
            res = like.fit(verbosity=1,covar=True,optObject=likeobj)
    print("ource Convergence Status" , likeobj.getRetCode())
    if get_like:
        
        return res , like.flux(params["name"] , emin = params["emin"]) , like.model["V3890_Sgr"].funcs["Spectrum"]["Prefactor"]
    like.logLike.writeXml(f'fit_model{fheader}.xml')
    Nova_flux = like.flux(params["name"] , emin = params["emin"])
    Nova_flux_err = like.fluxError(params["name"], emin=params["emin"])
    
    TS = like.Ts(f'{params["name"]}')
    
    return Nova_flux , Nova_flux_err , TS


def binned_likelihood(params, tstart , tend , clobber = False, fheader = ""):
    '''
    Function to run the full binned likelihood analysis pipeline
    Will run this once, and produces a TS value, and a flux
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    tstart : float : time (MET) for data start
    tstart : float : time (MET) for data end
    clobber : boolean : If true, overwrite existing files
    fheader : string : Unique ID added to file names to avoid name conflicts
    
    Returns
    ________
    None
    '''
    
    data_selection(params, tstart, tend, clobber , fheader)
    bin_data(params , clobber , fheader)
    lt_exp_maps(params , clobber , fheader)
    gen_model(params, False)
    gen_srcmap(params, clobber, fheader)
    Flux , error , TS = fit_model(params , fheader, False)
    
    return Flux, error, TS

def light_curve_singleproc(params, clobber, log = "results.csv"):
    '''
    Function to build a light curve
    Uses window width in parameter file, and step size
    This is the simple version of this function, useful for debugging
    any issues encountered during the analysis. Runs on a single core.
    
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    clobber : boolean : If true, overwrite existing files
    log : string : file to save lc data
    
    Returns
    ________
    None
    '''
    

    window_half_seconds = 12 * 60 * 60 * params["window"]
    step_seconds = 24 * 60 * 60 * params["lcstep"]
    t = params["start"] + step_seconds / 2.0
    
    Flux = []
    unc = []
    ts_vals = []
    time = []
    if os.path.exists(log) and not clobber:
        print ("LC file exists; exiting")
        return 0
    f = open(log , "w")
    f.close()
    id = 0
    while t + window_half_seconds < params["end"]:
        st = t - window_half_seconds
        et = t + window_half_seconds
        F , F_err , TS = binned_likelihood(params, st, et, clobber, fheader = str(id))
        if TS < 4:
            F = compute_upper_lim(params , str(id))
            F_err = -1
        
        Flux.append(F)
        unc.append(F_err)
        ts_vals.append(TS)
        time.append( (t - params["start"])/24 * 60 * 60)
        f= open(log , "a")
        f.write(f'{F},{F_err},{TS},{t}\n')
        f.close()
        t += step_seconds
        id += 1
        if params["cleanlc"]:
            cleanup(params , str(id))


def likelihood_wrapper(run_pars):
    '''
    Simple wrapper function to run the likelihood analysis using 
    only a single argument. Makes it easier for the multi-processing
    functions to run many likelihood analyses. Will automatically 
    compute upper limits if the test statistic from the main fit is
    less than 4. Set up_lim_lc to no in the parameter file to disable
    this behavior.
    
    Parameters
    __________
    run_pars : list : list of all likelihood parameters.
        Should contain, in this order:
        params : dict : parameter dict from read_parameters
        tstart : float : time (MET) for data start
        tstart : float : time (MET) for data end
        clobber : boolean : If true, overwrite existing files
        fheader : string : Unique ID added to avoid filename conflicts
        log : filename to save data to
        cleanup : boolean : If true, delete large intermediate files
        
    Returns
    ________
    Flux , Flux_Error , TS
    '''
    log_file = run_pars[4] + run_pars[5] + ".csv"
    F , unc , ts = binned_likelihood(*run_pars[0:5])
    if ts < 4:
        F = compute_upper_lim(run_pars[0] , run_pars[4])
        unc = -1
    f = open(log_file , "w")
    f.write(str(F) + "," + str(unc) + "," + str(ts))
    f.close()
    tmid = (run_pars[1] + run_pars[2]) / 2.0
    if run_pars[-1]:
        cleanup(run_pars[0] , run_pars[4])
    return [F , unc , ts , tmid]

def light_curve_multiproc(params , clobber, log="mp_log"):
    '''
    Function to build a light curve
    Uses window width in parameter file, and step size
    This is the multiprocessing version of this function
    
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    clobber : boolean : If true, overwrite existing files
    log : string : base file name to load data
    
    Returns
    ________
    None
    '''
    
    ## Start by setting up our parameter array
    param_array = []
    
    window_half_seconds = 12 * 60 * 60 * params["window"]
    step_seconds = 24 * 60 * 60 * params["lcstep"]
    t = params["start"] + step_seconds / 2.0
    

    
    id = 0
    while t + window_half_seconds < params["end"]:
        
        st = t - window_half_seconds
        et = t + window_half_seconds
        param_row = [params, st, et, clobber, '_' + str(id), log]
        param_row.append( params["cleanlc"])

        param_array.append(param_row)
        t += step_seconds
        id += 1
    
    with mp.Pool(params["nproc"]) as p:
        results = p.map(likelihood_wrapper , param_array)
    np.save(log + ".npy" , results)
    
    Flux = []
    unc = []
    ts = []
    time = []
    
    for i in results:
        Flux.append(i[0])
        unc.append(i[1])
        ts.append(i[2])
        time.append(i[3])
    
    time = np.array(time)
    Flux = np.array(Flux)
    unc = np.array(unc)
    ts = np.array(ts)
    time -= time[0]
    time /= (60 * 60 * 24)
    det = np.where(ts >=4)
    lim = np.where(ts < 4)
    plt.scatter(time[det] , Flux[det], color = "blue")
    plt.errorbar(time[det] , Flux[det] , unc[det] , ls = 'none', color = "blue")
    plt.scatter(time[lim] , Flux[lim] , color = "orange" , marker = "v")
    plt.xlabel("Time Since Start (days)")
    plt.ylabel("Flux (ph / s / cm^2")
    plt.savefig("LC.pdf")
    plt.close()
    return results

def gen_ul_xml(input_file, output_file, name , smodel):
    '''
    Simple function to setup the UL xml files
     Parameters
    __________
    input_file : string : starting model
    output_file : string : output filename
    name : string : name of our source
    flux : string : flux to install for our model
    
    Returns
    ________
    None
    '''
    
    
    insource = False
    infil = open(input_file)
    rf = open(output_file , "w")
    
    for line in infil.readlines():
        
        #line = line.replace('free="1"' , 'free="0"')
       
        if name in line:
            insource = True
            rf.write(smodel)
        
        if insource and "</source>" in line:
            insource = False
            continue
        if insource:

            continue
        rf.write(line)
    rf.close()
    infil.close()
    
def setup_pl(params,flux,index , free = False):
    '''
    Simple function to setup a power law model for a source
    used for upper limit computations
    
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    model : string : name of input xml file

    Returns
    _______
    model : string : string ready to be written into xml file
    '''
    model = f'<source name="{params["name"]}" type="PointSource">'
    model += f'<!-- point source units are cm^-2 s^-1 MeV^-1 -->'
    model += f'<spectrum type="PLSuperExpCutoff">'
    if free:
        model += f'<parameter free="1" max="1000" min="1e-05" name="Prefactor" scale="1e-07" value="{flux}"/>'
    else:
        model += f'<parameter free="0" max="1000" min="1e-05" name="Prefactor" scale="1e-07" value="{flux}"/>'
    model += f'<parameter free="0" max="0" min="-5" name="Index1" scale="1" value="{index}"/>'
    model += f'<parameter free="0" max="1000" min="50" name="Scale" scale="1" value="200"/>'
    model += f'<parameter free="0" max="30000" min="500" name="Cutoff" scale="1" value="2000"/>'
    model += f'<parameter free="0" max="5" min="0" name="Index2" scale="1" value="1.0"/>'
    model += f'</spectrum>'
    model += f'<spatialModel type="SkyDirFunction">'
    model += f'<parameter free="0" max="360." min="-360." name="RA" scale="1.0" value="{params["ra"]}"/>'
    model += f'<parameter free="0" max="90." min="-90." name="DEC" scale="1.0" value="{params["dec"]}"/>'
    model += f'</spatialModel>'
    model += f'</source>'
    return model


    
def compute_upper_lim(params, fheader):
    '''
    Function to compute the upper limit on the flux for a source
    This is an implementation of the profile-likelihood method, makes
    an assumption about what is reasonable for nova fluxes.
    Probably a bit inefficient, but the algorithm is straightforward
    Uses a profile likelihood method to find upper limit at given CL
    Algorithm description: Begins by freezing spectral parameters to
    some standard nova selections (see setup_pl function that sets the
    actual model). Runs one optimizer to fit model with all nova params
    frozen except for the normalization. This provides the max
    likelihood (L0). The goal is then to find the (larger) flux where
    the likelihood L satisfies 1.35 = -2 * (log(L)-log(L0)), currently
    done using a simple bisection root finder that will find the root 
    between the normalization parameter at L0 and the max allowed norm 
    (which is unreasonably bright). Assumes that there is one root in 
    the likelihood criterion. Different confidence levels correspond to 
    different likelihood differences (not currently supported, we 
    assume a 95% CL).
    
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    model : string : name of input xml file

    Returns
    _______
    ULF : float : upper limit flux
    '''
    

    def L(F):
        mod = setup_pl(params,F,-2.1)
        gen_ul_xml(f'fit_model{fheader}.xml',"ul.xml",params["name"],mod)
        logL , Flux , fpar = fit_model(params , fheader, True, inmod="ul.xml")
        return 2 * logL
    
    Fmax = 100
    base = 1e-5
    mod = setup_pl(params,1.0,-2.1, free=True)
    gen_ul_xml(f'fit_model{fheader}.xml',"ul.xml",params["name"],mod)
    

    base_L , base_F , base_p =fit_model(params , fheader, True, inmod="ul.xml")
    base_L *= 2
    
    def f(F):
        nL = L(F)
        return nL - base_L
    
    p_low = np.log10(base_p)
    p_high = np.log10(Fmax)
    L_low = -1.35
    L_high = f(10 ** p_high) - 1.35

    lpar = [p_low , p_high]
    Likes = [L_low , L_high]
    if L_high < 0:
        print ("UL Failure")
        raise ValueError
    from tqdm import tqdm
    for i in tqdm(range(20)):
        mid_p = (p_low + p_high) / 2.0
        L_mid = f(10 ** mid_p) - 1.35
        lpar.append(mid_p)
        Likes.append(L_mid)
        if L_mid * L_low < 0:
            L_high = L_mid
            p_high = mid_p
        else:
            p_low = mid_p
            L_low = L_mid
    
    mod = setup_pl(params,10 ** ((p_low + p_high) / 2.0),-2.1)
    gen_ul_xml(f'fit_model{fheader}.xml',"ul.xml",params["name"],mod)
    logL , Flux_final , fpar = fit_model(params , fheader, True, inmod="ul.xml")
    
    mod = setup_pl(params,10 ** p_low,-2.1)
    gen_ul_xml(f'fit_model{fheader}.xml',"ul.xml",params["name"],mod)
    logL1 , Flux_flow , fpar1 = fit_model(params , fheader, True, inmod="ul.xml")
    
    mod = setup_pl(params,10 ** p_high,-2.1)
    gen_ul_xml(f'fit_model{fheader}.xml',"ul.xml",params["name"],mod)
    logL2 , Flux_fhigh , fpar2 = fit_model(params , fheader, True, inmod="ul.xml")
    print (Flux_final , Flux_flow , Flux_fhigh , 2 * (logL - base_L))

    return Flux_final

def cleanup(params , fheader):
    
    '''
    Short function to remove files produced during a given likelihood
    run. Intent is to reduce file volume when generating light curves.
    
    WARNING!
    This function will delete files; do not run unless you are sure 
    that you want to remove these files. Intended to cleanup all of the
    sizeable files produced during the run.
    
    Parameters
    __________
    fheader: string : file id
    
    Returns
    _______
    '''
    
    
    for i in os.listdir():
        if params["name"] not in i and fheader not in i:
            continue
        if "srcmap" in i: ## Source Maps
            os.remove(i)
        elif "BinnedExpMap" in i or "_ltCube" in i:
            os.remove(i)
        elif "_filtered" in i:
            os.remove(i)
        
        

if __name__ == "__main__":
    
    paramfile = sys.argv[1]
    
    params = read_parameters(paramfile)
    
    print_params(params)
    

    F , F_err , TS = binned_likelihood(params, params["start"], params["end"], False)
    #compute_upper_lim(params, "")
    #print ("Test Statistic" , TS)

    #light_curve_singleproc(params , False)
    light_curve_multiproc(params , False)
    