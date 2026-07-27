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
from astropy.io import fits
import time
import yaml
import gc
import scipy.optimize as opt
import gen_alg as ga
import contextlib
import traceback

# for reading and editing XML files.
from xml.etree import ElementTree as ET

import gt_apps as my_apps
from GtApp import GtApp
from UpperLimits import UpperLimits

import pyLikelihood
from BinnedAnalysis import *


### Some global variables: May need to update to run on your systems



### Start with some useful background / setup functions

def setup_events_file(params, clobber=False):
    
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

    data_dir = params["data_path"]
    
    if os.path.exists(event_file) and clobber:
        os.remove(event_file)
        
    if not os.path.exists(event_file):
        f = open(event_file , "w")
        output = ""
        for i in os.listdir(data_dir):
            if "_PH" in i and ".fits" in i:
                output += data_dir + i + "\n"
        f.write(output[:-1])
        f.close()
    for i in os.listdir(data_dir):
        if "_SC" in i and ".fits" in i:
            scfile = data_dir + i

    
    infile = '@events.txt'
    return infile , scfile

def cal_to_met(date_time):
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

def tpeak_to_met(time , params):
    
    '''
    Function to compute Fermi MET, given a time relative to nova peak.
    
    Parameters
    __________
    time : float : Time since peak (negative for before peak) in days
    params : dict : parameter dict from read_parameters
    
    Returns
    _______
    MET : float : Fermi MET in seconds
    
    '''
    
    peak = params["peak"]
    ts = time * (24 * 60 * 60)
    return peak + ts



def met_to_tpeak(met , params):
    
    '''
    Function to compute the time in days since peak given a Fermi MET
    
    Parameters
    __________
    met : float : Fermi MET
    params : dict : parameter dict from read_parameters
    
    Returns
    _______
    t_peak : float : time since peak in days
    
    '''
    

    peak = params["peak"]
    delta_peak = met - peak
    
    return delta_peak / (24 * 60 * 60)
    
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
    
    with open(pfile, 'r') as f:
        config = yaml.safe_load(f)
        params = config["params"]
        
        params["runlog"] = os.path.join(os.getcwd(), "runtime_log.log")
        
        
        ## convert peak to MET
        date = params["peak"].split(" ")[0]
        time = params["peak"].split(" ")[1]
        year = int(date.split("-")[0])
        month = int(date.split("-")[1])
        day = int(date.split("-")[2])
        hour = int(time.split(":")[0])
        minute = int(time.split(":")[1])
        second = int(float(time.split(":")[2]))
        stime = dtime.datetime(year=year,month=month,day=day,hour=hour,
                    minute=minute,second=second,tzinfo=dtime.timezone.utc)
        MET = cal_to_met(stime)
        params["peak"] = MET
        
        ## Set a few defaults
        if "input_model" not in params.keys() or params["input_model"].lower() == "none":
            params["input_model"] = params["name"] + "_input_model.xml"
        if "infile" not in params.keys() or "scfile" not in params.keys():
            ## If data is not specified, auto-detect data files.
            infile , scfile = setup_events_file(params, clobber=False)
            params["infile"] = infile
            params["scfile"] = scfile
            
        for i in params.keys():
            if "outdir" in i or "figdir" in i:
                if not os.path.exists(params[i]):
                    os.mkdir(params[i])
        cal_dir = params["cal_dir"]
        vns = []
        for i in os.listdir(cal_dir):
            if "gll_psc" in i and ".xml" in i:
                vnumb = int(i.split("gll_psc_")[1].split(".xml")[0].replace("v",""))
                vns.append(vnumb)
        if len(vns) == 0:
            print ("No source catalogs found in cal_dir. Please add a catalog and try again.")
            sys.exit()
        params["source_cat"] = cal_dir + f"gll_psc_v{max(vns)}.xml"
        
        if 'bck_outdir' in params.keys():
            params['bck_output'] = (params['bck_outdir']
                                    + "background_results.csv")
        else:
            warning = "Warning: No background output in parameter file.\n"
            warning += " This parameter file is deprecated; additional params"
            warning += " should be added to the parameter file.\n"
            warning += " See template for details. Will cause error"
            warning += " in future versions.\n"
            
            print (warning)
            
        params['lc_logfile'] = (params['lc_outdir']
                                + "lightcurve_results.csv")
            
        params["avg_logfile"] = (params['av_outdir']
                                + "average_results.csv")
        
        params["grid_logfile"] = (params['grid_outdir']
                                + "grid_results.csv")
        
        params["mts_logfile"] = (params['tsm_outdir']
                                + "mts_results.csv")
        
        params["bck_logfile"] = (params['bck_outdir']
                                + "bck_results.csv")
        return params

def check_status(result_file):
    '''
    Utility function to check status of a run
    Reads in a result file and will return an array of times that have
    results available.
    '''
    
    if not os.path.exists(result_file):
        f = open(result_file , "w")
        f.write('Flux,Flux_Error,TS,Time,met_start,met_end\n')
        f.close()
        return [] , [] , []
    f = open(result_file , "r")
    times = [] 
    starts = []
    ends = []
    for i in f.readlines():
        if i.split(",")[3].strip() == "Time":
            continue
        times.append(float(i.split(",")[3]))
        starts.append(float(i.split(",")[4]))
        ends.append(float(i.split(",")[5]))
    return times , starts, ends

    
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

def load_source_cat(cat_fname):
    '''
    Function to load source catalog into a dictionary for easy access
    to the parameters that we care about
    '''
    tree = ET.parse(cat_fname)
    root = tree.getroot()
    source_cat = {}
    for source in root:
        name = source.get("name")
        fl = source.get("Energy_Flux100")
        var = source.get("Variability_Index")
        tsval  = source.get("TS_Value")
        source_cat[name] = {"fl100": fl, "var": var, "tsval": tsval}
    return source_cat

def prune_model(model_fname, params):

    if not os.path.exists('prune_files'):
        os.mkdir('prune_files')

    src_counts = binned_likelihood(params, tpeak_to_met(params['earliest_time'], params) , tpeak_to_met(params['latest_time'], params), 
    clobber = False, fheader = "prune", silent=False, lock = None, outdir = "prune_files/", compute_counts = True, skip_pruning = True)

    source_cat = load_source_cat(params['source_cat'])
    tree = ET.parse(model_fname)
    root = tree.getroot()

    print ("There are ", len(root), " sources in the model file.")
    bad_sources = []
    for source in root:
    
        spec = source.find("spectrum")
        spat = source.find("spatialModel")

        ROI_Center = source.get("ROI_Center_Distance")

        name = source.get('name')
        if name == (params['name']):
            continue
        if name not in source_cat.keys():
            print(name, 'is not in the source catalog')
            continue
        variability = float(source_cat[name]['var'])
        counts = src_counts[name]

        if ROI_Center is None:
            ROI_Center = 0.0
            ## Is our nova, leave in file

            continue
        # if variability index is the number in other code then its good. Code runs on photons need that from Peter (now)

        if float(ROI_Center) < 10:
            ## Too far from our nova, remove from file
            continue
        
        if variability > 30.58:
            continue
        
        if float(ROI_Center) > 10 and float(ROI_Center) < 15 and counts < 0.1:
            print (f"Removing {source.get('name')} from model file. ROI_Center_Distance = {ROI_Center}")           
            bad_sources.append(source)
        
        if float(ROI_Center) >= 15 and counts < 0.5:
            print (f"Removing {source.get('name')} from model file. ROI_Center_Distance = {ROI_Center}")           
            bad_sources.append(source)

    for sr in bad_sources:
        root.remove(sr)  
    print ("There are ", len(root), " sources  in the model file.")
    ## Save new model file
    tree.write(params['name'] + "_pruned_model.xml")


def gen_model(params, fheader, lock=None, outdir = "./", skip_pruning = False):
    '''
    Function to create an input model file
    Note: This function will not overwrite an existing input model.
    
        
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    
    Returns
    ________
    None
    '''
    
    
    
    gti = outdir + f"{params['name']}{fheader}_filtered_gti.fits"
    model_fname = params["input_model"]
    pruned_fname = outdir + f"{params['name']}_pruned_model.xml"

    if os.path.exists(params['name'] + "_pruned_model.xml"):
        params['input_model'] = params['name'] + "_pruned_model.xml"
        return 0

    if not os.path.exists(model_fname):
        xml_command = f' make4FGLxml {params["source_cat"]} --event_file {gti} --output_name '
        xml_command += f'{model_fname} --free_radius 5.0 --norms_free_only '
        xml_command += f'True --sigma_to_free 25 --variable_free True'
        subprocess.run(xml_command, shell=True)
        if not os.path.exists(f'{model_fname}'):
            print (xml_command)
            message = "make4FGLxml not successful; please run the above command "
            message += "and edit as needed, then hit enter:"
            input(message)
        else:
            input("Please edit input_model to include source models. Then, hit enter")
    if not os.path.exists(pruned_fname) and not skip_pruning and params['use_pruner']:
        scat = load_source_cat(params["source_cat"])
        prune_model(model_fname, params)
        params['input_model'] = params['name'] + "_pruned_model.xml"

    
def data_selection(params, tstart, tend, clobber, fheader, lock=None, outdir = "./"):
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

    out_name = outdir + f'{params["name"]}{fheader}_filtered.fits'
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
        checklocks(lock)
        my_apps.filter.run()
    
    gtiname = outdir + f'{params["name"]}{fheader}_filtered_gti.fits'
    my_apps.maketime['scfile'] = params["scfile"]
    my_apps.maketime['filter'] = '(DATA_QUAL>0)&&(LAT_CONFIG==1)'
    my_apps.maketime['roicut'] = 'no'
    my_apps.maketime['evfile'] = outdir + f'{params["name"]}{fheader}_filtered.fits'
    my_apps.maketime['outfile'] = gtiname

    if not os.path.exists(gtiname) or clobber:
        checklocks(lock)
        my_apps.maketime.run()

def lt_exp_maps(params , clobber , fheader, lock=None, outdir = "./"):
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
    ltcube = outdir + f'{params["name"]}{fheader}_ltCube.fits'

    my_apps.expCube['evfile'] = outdir + f'{params["name"]}{fheader}_filtered_gti.fits'
    my_apps.expCube['scfile'] = params["scfile"]
    my_apps.expCube['outfile'] = ltcube
    my_apps.expCube['zmax'] = 90
    my_apps.expCube['dcostheta'] = 0.025
    my_apps.expCube['binsz'] = 1

    if not os.path.exists(ltcube) or clobber:
        checklocks(lock)
        my_apps.expCube.run()


    ## Build Exposure Map
    expmap = outdir + f'{params["name"]}{fheader}_BinnedExpMap.fits'

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
        checklocks(lock)
        expCube2.run()
        
def gen_srcmap(params, clobber, fheader, lock=None, outdir = "./"):
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
    src_name = outdir + f'{params["name"]}{fheader}_srcmap.fits'
    my_apps.srcMaps['expcube'] = outdir + f'{params["name"]}{fheader}_ltCube.fits'
    my_apps.srcMaps['cmap'] = outdir + f'{params["name"]}{fheader}_filtered_ccube.fits'
    my_apps.srcMaps['srcmdl'] = params["input_model"]
    my_apps.srcMaps['bexpmap'] = outdir + f'{params["name"]}{fheader}_BinnedExpMap.fits'
    my_apps.srcMaps['outfile'] = src_name
    my_apps.srcMaps['irfs'] = 'P8R3_SOURCE_V3'
    my_apps.srcMaps['evtype'] = '3'

    if not os.path.exists(src_name) or clobber:
        checklocks(lock)
        my_apps.srcMaps.run()
    
    ## Check for file merge issues, and if so, run the merger function.
    ti = 0
    n_src_name = src_name + f"_{ti}.fits"
    
    if os.path.exists(n_src_name):
        split_names = []
        while os.path.exists(n_src_name):
            split_names.append(n_src_name)
            ti += 1
            n_src_name = src_name + f"_{ti}.fits"
        print (f"Source map appears to have been split into {len(split_names)} files. Running merger function.")
        src_merger(split_names, src_name)
def bin_data(params, clobber, fheader, lock=None, outdir = "./"):
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
    
    my_apps.evtbin['evfile'] = outdir + f'{params["name"]}{fheader}_filtered_gti.fits'
    my_apps.evtbin['outfile'] = outdir + f'{params["name"]}{fheader}_filtered_cmap.fits'
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

    if not os.path.exists(outdir + f'{params["name"]}{fheader}_filtered_cmap.fits') or clobber:
        checklocks(lock)
        my_apps.evtbin.run()

    
    ## Make CCUBE while we are at it:

    npix = int(( np.sqrt(2) * params["roi"] / params["pix_sc"] ))
    my_apps.evtbin['evfile'] = outdir + f'{params["name"]}{fheader}_filtered_gti.fits'
    my_apps.evtbin['outfile'] = outdir + f'{params["name"]}{fheader}_filtered_ccube.fits'
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

    if not os.path.exists(outdir + f'{params["name"]}{fheader}_filtered_ccube.fits') or clobber:
        checklocks(lock)
        my_apps.evtbin.run()

def fit_model(params, fheader, get_like, inmod = "No" , opt = 'NewMINUIT', silent=False, lock=None, outdir = "./"):
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
    silent : boolean : If true, supress first optimizer output.
    Returns
    ________
    None
    '''
    
    if inmod == "No":
        inmod =params["input_model"]
        
    src_name = outdir + f'{params["name"]}{fheader}_srcmap.fits'
    
    # ! The following is no longer used and should be removed in a future update:
    drmnfb_comm = f'gtlike statistic=BINNED cmap={src_name} '
    drmnfb_comm += f'bexpmap={outdir}{params["name"]}{fheader}_BinnedExpMap.fits '
    drmnfb_comm += f'expcube={outdir}{params["name"]}{fheader}_ltCube.fits '
    drmnfb_comm += f'srcmdl={inmod} irfs=CALDB'
    drmnfb_comm += f' optimizer=DRMNFB sfile={outdir}temp{fheader}.xml '
    
    if silent:
        drmnfb_comm += ">/dev/null "
    
    #checklocks(lock)
    #subprocess.run(drmnfb_comm,shell=True)
    
    obs = BinnedObs(srcMaps=src_name,
            binnedExpMap=f'{outdir}{params["name"]}{fheader}_BinnedExpMap.fits',
            expCube=f'{outdir}{params["name"]}{fheader}_ltCube.fits',irfs='P8R3_SOURCE_V3')
    # // like = BinnedAnalysis(obs,f'{outdir}temp{fheader}.xml',optimizer=opt)
    like = BinnedAnalysis(obs,f'{inmod}',optimizer="DRMNFB")
    likeobj=pyLike.NewMinuit(like.logLike)

    like.tol = 0.01
    
    checklocks(lock)
    res = like.fit(verbosity=1,covar=True,optObject=likeobj)
    like.optimizer = "NewMinuit"
    
    try:
        like.tol = 0.0001
        checklocks(lock)
        res = like.fit(verbosity=1,covar=True,optObject=likeobj)
    except:
        try:
            like.tol = 0.01
            checklocks(lock)
            res = like.fit(verbosity=1,covar=True,optObject=likeobj)
        except:
            like = BinnedAnalysis(obs,inmod,optimizer="DRMNFB")
            likeobj=pyLike.NewMinuit(like.logLike)
            like.tol = 0.01
            checklocks(lock)
            res = like.fit(verbosity=1)
    print("Source Convergence Status" , likeobj.getRetCode())
    if get_like:
        
        return res , like.flux(params["name"] , emin = params["emin"]) , like.model[params["name"]].funcs["Spectrum"]["Prefactor"]
    like.logLike.writeXml(f'{outdir}fit_model{fheader}.xml')
    Nova_flux = like.flux(params["name"] , emin = params["emin"], emax=params["emax"])
    Nova_flux_err = like.fluxError(params["name"], emin=params["emin"], emax=params["emax"])
    
    TS = like.Ts(f'{params["name"]}')
    
    
    return Nova_flux , Nova_flux_err , TS

def get_counts(params, fheader, outdir, opt = 'NewMINUIT', silent=False, lock=None):

    inmod =params["input_model"]

    src_name = outdir + f'{params["name"]}{fheader}_srcmap.fits'
    
    obs = BinnedObs(srcMaps=src_name,
            binnedExpMap=f'{outdir}{params["name"]}{fheader}_BinnedExpMap.fits',
            expCube=f'{outdir}{params["name"]}{fheader}_ltCube.fits',irfs='P8R3_SOURCE_V3')
    # // like = BinnedAnalysis(obs,f'{outdir}temp{fheader}.xml',optimizer=opt)
    like = BinnedAnalysis(obs,f'{inmod}',optimizer="DRMNFB")
    likeobj=pyLike.NewMinuit(like.logLike)

    like.tol = 0.01
    
    checklocks(lock)
    res = like.fit(verbosity=1,covar=True,optObject=likeobj)
    like.optimizer = "NewMinuit"
    
    try:
        like.tol = 0.0001
        checklocks(lock)
        res = like.fit(verbosity=1,covar=True,optObject=likeobj)
    except:
        try:
            like.tol = 0.01
            checklocks(lock)
            res = like.fit(verbosity=1,covar=True,optObject=likeobj)
        except:
            like = BinnedAnalysis(obs,inmod,optimizer="DRMNFB")
            likeobj=pyLike.NewMinuit(like.logLike)
            like.tol = 0.01
            checklocks(lock)
            res = like.fit(verbosity=1)
    print("Source Convergence Status" , likeobj.getRetCode())

    src_counts ={}
    for name in like.sourceNames():
        src_counts[name] = np.sum(like._srcCnts(name))
    return src_counts

def checklocks(lock):
    '''
    Function to manage multiprocessing locks. Intended to make sure that
    no two FermiTools steps get called at exactly the same time. If the lock is
    not available, will wait for it and then sleep for a small amount of time.
    
    '''
    
    if lock is None:
        return 0
    else:
        with lock:
            time.sleep(1)
            
def binned_likelihood(params, tstart , tend , clobber = False, fheader = "", silent=False, lock = None, outdir = "./",
compute_counts = False, skip_pruning = False):

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
    silent : boolean : Flag to suppress terminal output. Only supresses the
                    output from one fit model call, does not handle python outputs
    Returns
    ________
    None
    '''
    
    #return 1e-7,1e-8,157

    runlogname = "runlog" + fheader + ".log"
    
    if params["runlog"]:
        rf = open(runlogname , "a")
        rf.write("Beginning Data Selection\n")
        rf.close()
    
    data_selection(params, tstart, tend, clobber , fheader, lock, outdir)
    

    if params["runlog"]:
        rf = open(runlogname , "a")
        rf.write("Beginning Data Binning\n")
        rf.close()
        
    bin_data(params , clobber , fheader, lock, outdir)
    
    if params["runlog"]:
        rf = open(runlogname , "a")
        rf.write("Building Exposure Maps\n")
        rf.close()
    
    
    lt_exp_maps(params , clobber , fheader, lock, outdir)
    
    if params["runlog"]:
        rf = open(runlogname , "a")
        rf.write("Getting Source Model\n")
        rf.close()
    
    gen_model(params, fheader, lock, outdir = outdir, skip_pruning = skip_pruning)
    
    if params["runlog"]:
        rf = open(runlogname , "a")
        rf.write("Generating Source Map\n")
        rf.close()
    
    gen_srcmap(params, clobber, fheader, lock, outdir)
    
    if params["runlog"]:
        rf = open(runlogname , "a")
        rf.write("Fitting Source Model\n")
        rf.close()
    if compute_counts:
        src_counts = get_counts(params, fheader, outdir)
        return(src_counts)
    Flux , error , TS = fit_model(params , fheader, False, silent=silent,lock=lock, outdir = outdir)
    
    if params["runlog"]:
        rf = open(runlogname , "a")
        rf.write("Building model and residual maps\n")
        rf.close()
        
    generate_residuals(params, clobber, fheader, lock, outdir)
    
    if params["runlog"]:
        rf = open(runlogname , "a")
        rf.write("Likelihood Calculation Complete\n")
        rf.write("-" * 33 + "\n")
        rf.close()

    return Flux, error, TS

def src_merger(split_files,fout):
    
    hdu2 = fits.open(fout)
    new_hdus = []
    new_hdus.append(hdu2[0].copy())
    new_hdus.append(hdu2[1].copy())
    new_hdus.append(hdu2[2].copy())
    hdu2.close()
    names = []
    for fname in split_files:
        hdu0 = fits.open(fname)
        primary = True
        for hdu in hdu0[::]:
            
            try:
                sname = hdu.header["EXTNAME"]
            except KeyError:
                
                sname = hdu.header["HDUNAME"]
            if sname in names:
                continue
            names.append(sname)
            if primary:
                primary = False
                
                hdu.header.pop("HDUNAME", None)

                # ensure EXTNAME consistency
                hdu.header["EXTNAME"] = sname

                # rebuild HDU cleanly (not copy())
                new_hdus.append(fits.ImageHDU(data=hdu.data, header=hdu.header))
                continue
            new_hdus.append(hdu.copy())
        hdu0.close()

    hdul_out = fits.HDUList(new_hdus)
    hdul_out.writeto(fout, overwrite=True)
    hdul_out.close()

def opt_func(x):
    '''
    Brief function that is ready to be minimized by scipy
    and other optimizers. Takes in an array of length 2, and returns
    a function that will evaluate TS value of a given window. The input
    array should contain, in order, the start time for data selection 
    (in days relative to peak) and the length of the data window in days.
    
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    
    Returns:
    f(x) : function : function that evaluates TS as a function of window parameters.
                        Note: Returns -1 * TS to work with minimization algorithms.
    '''
    logfile = "all_fits.csv"
    fheader =  f"_temp_{round(x[0],3)}_{round(x[1],3)}_"
    
    with contextlib.redirect_stdout(None):
        params = read_parameters(sys.argv[1])
        tstart = tpeak_to_met(x[0] , params)
        tend = tpeak_to_met(x[0] + x[1] , params)
        tsm_dir = params["tsm_outdir"]
        try:
            Flux , error , TS = binned_likelihood(params, tstart, tend, False,
                                    fheader, silent = True, outdir = tsm_dir)
        except:
            try:
                Flux , error , TS = binned_likelihood(params, tstart, tend, False,
                                   fheader, silent = True, outdir = tsm_dir)
            except Exception as e:
                ## No luck, we return inf. and will skip this iteration.
                ## Benefit of a Monte-Carlo optimizer, we can just try a new solution.
                ## However, we will produce an error log to track it down later.
                err_log = f"{tsm_dir}elog{fheader}.txt"
                el = open(err_log , "w")
                el.write(f"Binned Likelihood crashed on data between {x[0]} and {x[0] + x[1]}")
                el.write("\n\n\n")
                el.write(str(e))
                el.close()
                return np.inf
    cleanup(params , fheader = fheader, all_files = True, outdir = tsm_dir)
    lf = open(tsm_dir + logfile , "a")
    lf.write(str(x[0]) + "," + str(x[1]) + "," + str(TS) + "\n")
    lf.close()
    print (f"Final Results are: TS = {TS} with data from {x[0]} to {x[0] + x[1]}")
    return -1 * TS


def FermiTools_UpperLim(params, fheader, outdir = "./"):

    '''
    For comparison, here is the FermiTools Upper Limit code
    I've had some reliability issues with this method; seems to be due
    to optimizers trying to exceed parameter boundaries while fitting.
    '''
    
    
    if not os.path.exists(f'{outdir}upper_lim_model{fheader}.xml'):
        mod = setup_pl(params,1.0,-2.1 , free = True)
        gen_ul_xml(f"{outdir}fit_model{fheader}.xml", f'{outdir}upper_lim_model{fheader}.xml',
                params["name"], mod)
    obs = BinnedObs(srcMaps=f"{outdir}{params['name']}{fheader}_srcmap.fits",
                binnedExpMap=f'{outdir}{params["name"]}{fheader}_BinnedExpMap.fits',
                expCube=f'{outdir}{params["name"]}{fheader}_ltCube.fits',
                irfs='P8R3_SOURCE_V3')
    like = BinnedAnalysis(obs,f'{outdir}upper_lim_model{fheader}.xml')
    like.fit(verbosity=3)
    ul = UpperLimits(like)
    try:
        ul[params["name"]].compute(emin=params["emin"],emax=params["emax"])
    except:
        return -1
    print (ul[params["name"]].results)
    flux = float(str(ul[params["name"]].results[0]).split(" ")[0])
    
    return flux



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
        lock : lock object : Lock used to prevent file conflicts
        outdir : string : Location to store all produced files
    Returns
    ________
    Flux , Flux_Error , TS
    '''

    center_t = (run_pars[1] + run_pars[2]) / 2.0
    center_t = met_to_tpeak(center_t, run_pars[0])
    
    
    try:
        F , unc , ts = binned_likelihood(*run_pars[0:5], lock = run_pars[6], outdir = run_pars[7])
        
    except Exception as e:
        err_l = run_pars[7] + "err_log_1_" + run_pars[4] + ".log"
        outlog = open(err_l , "w")
        for par in run_pars:
            outlog.write(str(par) + "\n")
        outlog.write(traceback.format_exc())
        outlog.close()
        try:
            F , unc , ts = binned_likelihood(*run_pars[0:5], lock = run_pars[6], outdir = run_pars[7])
        except:
            err_l = run_pars[7] + "err_log_1_" + run_pars[4] + ".log"
            outlog = open(err_l , "w")
            for par in run_pars:
                outlog.write(str(par) + "\n")
            outlog.write(traceback.format_exc())
            outlog.close()
            return [0,0,0,0]
        # I know what you're thinking ... and yes, this does look insane.
        # However, there is a reason for simply trying again. Sometimes, there
        # are crashes generated by one of the likelihood steps (could be
        # almost any step) that are caused by a missing .par file. This file
        # stores the parameters of the last used run. Many of the FermiTools
        # start by reading in this file, and at some point will delete and
        # rewrite it; if a process tries to read this file at the same time
        # that another file deletes it, we get a crash. So, if we try again
        # the code will pick up at whatever step we left off on, and the file
        # will most likely exist. Nominally the locking mechanisms now
        # fix this, but code is still here for testing.

    
    F2 = -99
    if ts < run_pars[0]["ts_lim"] and run_pars[0]["up_lim_lc"]:
        try:
            F , Flow , Fhigh , DeltaLogL = compute_upper_lim(run_pars[0] , run_pars[4], outdir = run_pars[7])
            
            unc = -1
            '''
            try:
            
                F2 = FermiTools_UpperLim(run_pars[0] , run_pars[4], outdir = run_pars[7])
            except:
                F2 = -1
            '''
        except:
            try:
                F = FermiTools_UpperLim(run_pars[0] , run_pars[4], outdir = run_pars[7])
                #F2 = -99
            except:
                return [0,0,0,0]
            
    runlogname = run_pars[7] + "runlog" + run_pars[4] + ".log"
    if run_pars[0]["runlog"]:
        rf = open(runlogname , "a")
        rf.write("Run Complete; saving to a file\n")
        rf.close()
    
    
    if F >= 0:
        
        
        tmid = (run_pars[1] + run_pars[2]) / 2.0
        tmid = met_to_tpeak(tmid, run_pars[0])
        
        with run_pars[6]:
            
            f = open(run_pars[5] , "a")
            f.write(str(F) + "," + str(unc) + "," + str(ts) + "," + str(tmid))
            f.write("," + str(run_pars[1]) + "," + str(run_pars[2]))
            f.write("\n")
            f.close()
        
    else:
        print ("Warning; Flux is Negative!")
        
    if run_pars[0]["cleanlc"]:
        cleanup(run_pars[0] , run_pars[4], all_files = True, outdir = run_pars[7])
    return [F , unc , ts , tmid]

def light_curve_singleproc(params, clobber, log = "mp_log"):
    '''
    Function to build a light curve
    Uses window width in parameter file, and step size
    This is the multiprocessing version of this function
    This version in particular is only called if nproc = 1
    Mainly intended for debugging, but the behaviour is identical
    to the multiproc version (except single-processed).
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    clobber : boolean : If true, overwrite existing files
    log : string : base file name to load data
    
    Returns
    ________
    None
    '''
    

    ## Get start and end times for LC.
    start = tpeak_to_met(params["lc_start"], params)
    end = tpeak_to_met(params["lc_end"], params)
    lcdir = params["lc_outdir"]
    log = log
    ## Start by setting up our parameter array
    param_array = []
    
    
    window_half_seconds = 12 * 60 * 60 * params["window"]
    step_seconds = 24 * 60 * 60 * params["lcstep"]
    t = start + step_seconds / 2.0
    tpeak_start = met_to_tpeak(start, params)

    
    id = 0
    while t < end:
        
        fheader = f"_{params['window']}_{params['lcstep']}_{tpeak_start}_st{id}"
        st = t - window_half_seconds
        et = t + window_half_seconds
        param_row = [params, st, et, clobber, fheader, log, None, lcdir]
        param_row.append( params["cleanlc"])
        center_t = met_to_tpeak(t , params)
        log_file = param_row[7] + param_row[5] + param_row[4] + f"_{int(center_t)}.csv"
        if not os.path.exists(log_file):
            
            param_array.append(param_row)
        t += step_seconds
        id += 1
    
    results = []
    for parameter_row in param_array:
        
        result = likelihood_wrapper(parameter_row)
        results.append(result)
        #results = p.map(likelihood_wrapper , param_array)
    np.save(log + ".npy" , results)
    
    Flux = []
    unc = []
    ts = []
    time = []
    tpeak = []
    for i in results:
        Flux.append(i[0])
        unc.append(i[1])
        ts.append(i[2])
        time.append(i[3])
        tpeak.append(met_to_tpeak(i[3] , params))
        
    time = np.array(time)
    Flux = np.array(Flux)
    unc = np.array(unc)
    ts = np.array(ts)
    tpeak = np.array(tpeak)

    det = np.where(ts >=4)
    lim = np.where(ts < 4)
    plt.scatter(tpeak[det] , Flux[det], color = "blue")
    plt.errorbar(tpeak[det] , Flux[det] , unc[det] , ls = 'none', color = "blue")
    plt.scatter(tpeak[lim] , Flux[lim] , color = "orange" , marker = "v")
    plt.xlabel("Time since peak (days)")
    plt.ylabel("Flux (ph / s / cm$^{-2}$)")
    plt.savefig(params["figdir"] + "LC.pdf")
    plt.close()
    
    return results

def light_curve_multiproc(params , clobber):
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
 
    start = tpeak_to_met(params["lc_start"], params)
    end = tpeak_to_met(params["lc_end"], params)
    lcdir = params["lc_outdir"]

    ## Start by setting up our parameter array
    param_array = []
    
    window_half_seconds = 12 * 60 * 60 * params["window"]
    step_seconds = 24 * 60 * 60 * params["lcstep"]
    t = start + step_seconds / 2.0
    tpeak_start = met_to_tpeak(start, params)

    status_log, log_starts, log_ends = check_status(params["lc_logfile"])
    
    with mp.Manager() as manager:
        lock = manager.Lock()
        id = 0
        while t < end:

            fheader = f"_{params['window']}_{params['lcstep']}_{tpeak_start}_st{id}"
            st = t - window_half_seconds
            et = t + window_half_seconds
            t_day = met_to_tpeak(t , params)
            skip = False
            for i in range(len(log_starts)):
                if st == log_starts[i] and et == log_ends[i]:
                
                    print (f"Skipping {t_day} as it is already in the log file")
                    t += step_seconds
                    id += 1
                    skip = True
            if skip:
                continue
            
            
            param_row = [params, st, et, clobber, fheader, params['lc_logfile'], lock, lcdir]
            param_row.append( params["cleanlc"])
            center_t = met_to_tpeak(t , params)
            
            
            
            t += step_seconds
            id += 1
            param_array.append(param_row)
        ## maxtasksperchild = 1 is designed to resolve a memory usage problem
        ## Probably mildly inneficient, but better than consuming many GB of
        ## RAM per process.
        print (len(param_array))
        
        
        results = []
        with mp.Pool(processes=params["nproc"], maxtasksperchild=1) as p:
            imres = p.imap(likelihood_wrapper , param_array, chunksize=1)
            for res in imres:
                results.append(res)
    
    return results

def false_positive_rate(params, clobber, log = "mp_log"):
    '''
    Function to compute the false positive rate for a given nova
    This is done by running a light curve analysis on a set of
    time windows prior to the eruption, and then measuring the distribution
    of the test statistics (TS) in those windows. This is a simple way to
    estimate the false positive rate for a given nova, and can be used
    to estimate the significance of any detections in the light curve.
    
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    clobber : boolean : If true, overwrite existing files
    log : string : base file name to load data
    
    Returns
    ________
    None
    '''
    
    start = tpeak_to_met(params["bck_start"], params)
    end = tpeak_to_met(params["bck_end"], params)
    lcdir = params["bck_outdir"]
        # // log =  log
        ## Start by setting up our parameter array
    param_array = []
    
    window_half_seconds = 12 * 60 * 60 * params["window"]
    step_seconds = window_half_seconds * 2.0 ## No overlap for this analysis
    
    t = start + step_seconds / 2.0
    tpeak_start = met_to_tpeak(start, params)

    status_log, log_starts, log_ends = check_status(params["bck_logfile"])
    
    with mp.Manager() as manager:
        lock = manager.Lock()
        id = 0
        while t < end:

            fheader = f"_{params['window']}_{params['lcstep']}_{tpeak_start}_st{id}"
            st = t - window_half_seconds
            et = t + window_half_seconds
            
            skip = False
            for k in range(len(log_starts)):
                if st == log_starts[k] and et == log_ends[k]:
                
                    print (f"Skipping {t} as it is already in the log file")
                    t += step_seconds
                    id += 1
                    skip = True
            if skip:
                continue
            
            param_row = [params, st, et, clobber, fheader, params["bck_logfile"], lock, lcdir]
            param_row.append( params["cleanlc"])
            param_row.append( params["cleanlc"])
            center_t = met_to_tpeak(t , params)
            log_file = param_row[7] + param_row[5] + param_row[4] + f"_{int(center_t)}.csv"
            if not os.path.exists(log_file) or clobber:
                
                param_array.append(param_row)
            t += step_seconds
            id += 1

        ## maxtasksperchild = 1 is designed to resolve a memory usage problem
        ## Probably mildly inneficient, but better than consuming many GB of
        ## RAM per process.
        print (len(param_array))
        
        
        results = []
        with mp.Pool(processes=params["nproc"], maxtasksperchild=1) as p:
            imres = p.imap(likelihood_wrapper , param_array, chunksize=1)
            for res in imres:
                results.append(res)
    
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
    model = f'<source name="{params["name"]}" type="PointSource">\n'
    model += f'<!-- point source units are cm^-2 s^-1 MeV^-1 -->\n'
    model += f'<spectrum type="PLSuperExpCutoff">\n'
    if free:
        model += f'<parameter free="1" max="1000" min="1e-05" name="Prefactor" scale="1e-07" value="{flux}"/>\n'
    else:
        model += f'<parameter free="0" max="1000" min="1e-05" name="Prefactor" scale="1e-07" value="{flux}"/>\n'
    model += f'<parameter free="0" max="-1" min="-3.5" name="Index1" scale="1" value="{index}"/>\n'
    model += f'<parameter free="0" max="1000" min="50" name="Scale" scale="1" value="200"/>\n'
    model += f'<parameter free="0" max="30000" min="500" name="Cutoff" scale="1" value="2000"/>\n'
    model += f'<parameter free="0" max="5" min="0" name="Index2" scale="1" value="1.0"/>\n'
    model += f'</spectrum>\n'
    model += f'<spatialModel type="SkyDirFunction">\n'
    model += f'<parameter free="0" max="360." min="-360." name="RA" scale="1.0" value="{params["ra"]}"/>\n'
    model += f'<parameter free="0" max="90." min="-90." name="DEC" scale="1.0" value="{params["dec"]}"/>\n'
    model += f'</spatialModel>\n'
    model += f'</source>\n'
    return model

def compute_upper_lim(params, fheader, outdir = "./"):
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
    the likelihood L satisfies 2.71 = 2 * (log(L)-log(L0)), currently
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

    opt = 'NewMINUIT'
    runlogname = outdir + "runlog" + fheader + ".log"
    if params["runlog"]:
        rf = open(runlogname , "a")
        rf.write("Initiating Upper Limit Calculation\n")
        rf.close()
    

    if params["runlog"]:
        rf = open(runlogname , "a")
        rf.write("Finding Best Fit Model\n")
        rf.close()
        
    ## Setup model, and find best fit parameters first
    index = -2.1
    src_name = f'{outdir}{params["name"]}{fheader}_srcmap.fits'
    mod = setup_pl(params,1.0,index, free=True)
    gen_ul_xml(f'{outdir}fit_model{fheader}.xml',f"{outdir}ul{fheader}.xml",params["name"],mod)
    
    obs = BinnedObs(srcMaps=src_name,
        binnedExpMap=f'{outdir}{params["name"]}{fheader}_BinnedExpMap.fits',
        expCube=f'{outdir}{params["name"]}{fheader}_ltCube.fits',irfs='P8R3_SOURCE_V3')
    like = BinnedAnalysis(obs,f'{outdir}ul{fheader}.xml',optimizer=opt)
    likeobj=pyLike.NewMinuit(like.logLike)

    
    try:
        like.tol = 0.0001
        res = like.fit(verbosity=1,covar=True,optObject=likeobj)
    except:
        
        print ("Changing Tolerance for fitting")
        like.tol = 0.01
        res = like.fit(verbosity=1,covar=True,optObject=likeobj)
    
    def L(P):
        '''
        Function to run the model fitting and return -logL. The intent
        is to find the point where the return of this function is 
        2.71/2 greater than the best fitting likelihood
        '''
        
        ## Freeze out our model Prefactor
        like.model[params["name"]]["Spectrum"].getParam("Prefactor").setFree(False)
        
        ## Reset prefactor:
        like.model[params["name"]]["Spectrum"]["Prefactor"] = P
        
        like.model[params["name"]]["Spectrum"]["Index1"] = index
        like.model[params["name"]]["Spectrum"].getParam("Index1").setFree(False)
        ## Fit model to compute likelihood
        try:
            like.tol = 0.001
            like.optimizer = opt
            res = like.fit(verbosity=1,covar=True,optObject=likeobj)
        except:
            try:
                print ("Changing Tolerance for fitting")
                like.tol = 0.01
                like.optimizer = "DRMNFB"
                like.fit(verbosity=1)
                like.optimizer="NewMinuit"
                res = like.fit(verbosity=1,covar=True,optObject=likeobj)
            except:
                return np.inf
                
        
        Flux = like.flux(params["name"] , emin = params["emin"], emax = params["emax"])
        
        return res , Flux
    ## Setup starting point for root finding

    Like_cut = 2.71 / 2.0
    L0 = res
    p_low = like.model[params["name"]].funcs["Spectrum"]["Prefactor"]
    p_low = np.log10(p_low)
    Fmax = 0.05
    p_high = np.log10(Fmax)
    L_low = -1 * Like_cut ##Delta is 0 for the max likelihood, so this is DeltaL - 2.71/2.0
    L_high = L(10 ** p_high)[0] - Like_cut
    
    if L_high < 0:
        p_high = 0.5
        L_high = L(0.5)[0] - Like_cut
        if L_high < 0:
            print ("UL Failure, could not find suitable bracket")
        
            return -1
    
    lpar = [p_low , p_high]
    Likes = [L_low , L_high]
    fm = []
    Lm = []
    lmids = []
    N = 1
    flux_mid = "N/A"
    convergence_requirement = 0.002
    step_numb = 0
    max_step = 40
    min_step = 12
    
    while step_numb < max_step: ## 20 steps is sufficient to get to a flux sltn.

        print (f"\n\n Starting step number {step_numb + 1}")
        print (f" Current flux is {flux_mid} \n\n")
        if params["runlog"]:
            rf = open(runlogname , "a")
            rf.write(f"Starting step number {step_numb + 1}\n")
            rf.close()
            
        mid_p = (p_low + p_high) / 2.0
        L_mid , flux_mid= L(10 ** mid_p)
        L_mid = L_mid  - L0 - Like_cut
        fm.append(flux_mid)
        lpar.append(mid_p)
        Likes.append(L_mid)
        lmids.append(mid_p)
        if L_mid * L_low < 0:
            L_high = L_mid
            p_high = mid_p
        else:
            p_low = mid_p
            L_low = L_mid
        print (N)
        N += 1
        
        Lm.append(L_mid)
        ## Check for convergence:
        step_numb += 1
        if len(fm) < 2:
            continue
        if (abs(fm[-1] - fm[-2] ) / fm[-1]) < convergence_requirement:
            if step_numb >= min_step:
                print (f"Flux has Converged in {step_numb} steps")
                break
    try:
        l_low , Flux_low = L(10**p_low)
        l_h , Flux_high = L(10**p_high)
    except:
        Flux_low = -1
        Flux_high = -1
    
    l_mid , Flux_final = L( 10 ** ( (p_high + p_low) / 2.0) ) 
    

    return Flux_final , Flux_low , Flux_high , l_mid - L0

def cleanup(params , fheader, all_files = False, outdir = "./"):
    
    '''
    Short function to remove files produced during a given likelihood
    run. Intent is to reduce file volume when generating light curves.
    
    WARNING!
    This function will delete files; do not run unless you are sure 
    that you want to remove these files. Intended to cleanup all of the
    sizeable files produced during the run.
    
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    fheader: string : file id
    all_fits : boolean : if true, deletes every file except the final results.
    
    Returns
    _______
    '''
    
    print ("DEBUGGING CLEANUP PARAMETERS")
    print (outdir)
    print (params)
    print (fheader)
    for i in os.listdir(outdir):
        
        if (params["name"] not in i and "temp" not in i) or fheader not in i:
            continue
        elif ".csv" in i:
            continue
        
        print (i)
        if "srcmap" in i: ## Source Maps
            os.remove(outdir + i)
        elif "BinnedExpMap" in i or "_ltCube" in i:
            os.remove(outdir + i)
        elif "_filtered" in i:
            os.remove(outdir + i)
        elif ".fits" in i and all_files:
            os.remove(outdir + i)
        elif ".xml" in i and all_files:
            os.remove(outdir + i)
        
def setup_tsmap_xml(params, input_file, outdir):
    '''
    Function to build a xml input file suitable for computing
    background TSMaps. Basically just takes an xml file and strips
    the model for our source. Will also freeze out model parameters
    (otherwise, runtime quickly becomes intractable. Takes in the xml
    file for the model that we want to use.
    
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    input_file : string : name of xml input file
    
    Returns
    _______
    None
    '''

    rf = open(f"{outdir}{params['name']}_fit_TSMap.xml" , "w")
    backf = open(f"{outdir}{params['name']}_fit_backgroundTSMap.xml" , "w")
    
    insource = False
    infil = open(input_file)
    for line in infil.readlines():
        line = line.replace('free="1"' , 'free="0"')
        rf.write(line)
        if params["name"] in line:
            insource = True
        if not insource:
            backf.write(line)
        if insource and "</source>" in line:
            insource = False
    rf.close()
    backf.close()
    infil.close()
    
def TS_Map(params, input_file, clobber):
    '''
    Function to generate TS Maps.
    Will build two files, one with the full source model list, called 
    name_TSmap_resid.fits, and one with the nova model removed, called
    name_TSmap_background_resid.fits.
    
    Parameters
    __________
    params : dict : parameter dict from read_parameters
    input_file : string : name of xml input file
    
    Returns
    _______
    None
    '''

    outdir =params["av_outdir"]
    setup_tsmap_xml(params , input_file, outdir)
    my_apps.TsMap['statistic'] = "BINNED"
    my_apps.TsMap['cmap'] = f'{outdir}{params["name"]}_filtered_ccube.fits'
    my_apps.TsMap['scfile'] = params["scfile"]
    my_apps.TsMap['evfile'] = f"{outdir}{params['name']}_filtered_gti.fits"
    my_apps.TsMap['bexpmap'] = f"{outdir}{params['name']}_BinnedExpMap.fits"
    my_apps.TsMap['expcube'] = f"{outdir}{params['name']}_ltCube.fits"
    my_apps.TsMap['srcmdl'] = f"{outdir}{params['name']}_fit_TSMap.xml"
    my_apps.TsMap['irfs'] = "P8R3_SOURCE_V3"
    my_apps.TsMap['optimizer'] = "NEWMINUIT"
    my_apps.TsMap['outfile'] = f"{outdir}{params['name']}_TSmap_resid.fits"
    my_apps.TsMap['nxpix'] = params["TSPix"]
    my_apps.TsMap['nypix'] = params["TSPix"]
    my_apps.TsMap['binsz'] = params["TSscale"]
    my_apps.TsMap['coordsys'] = "CEL"
    my_apps.TsMap['xref'] = params["ra"]
    my_apps.TsMap['yref'] = params["dec"]
    my_apps.TsMap['proj'] = 'AIT'
    if not os.path.exists(f"{outdir}{params['name']}_TSmap_resid.fits") or clobber:
        my_apps.TsMap.run()

    my_apps.TsMap['statistic'] = "BINNED"
    my_apps.TsMap['cmap'] = f'{outdir}{params["name"]}_filtered_ccube.fits'
    my_apps.TsMap['scfile'] = params["scfile"]
    my_apps.TsMap['evfile'] = f"{outdir}{params['name']}_filtered_gti.fits"
    my_apps.TsMap['bexpmap'] = f"{outdir}{params['name']}_BinnedExpMap.fits"
    my_apps.TsMap['expcube'] = f"{outdir}{params['name']}_ltCube.fits"
    my_apps.TsMap['srcmdl'] = f"{outdir}{params['name']}_fit_backgroundTSMap.xml"
    my_apps.TsMap['irfs'] = "P8R3_SOURCE_V3"
    my_apps.TsMap['optimizer'] = "NEWMINUIT"
    my_apps.TsMap['outfile'] = f"{outdir}{params['name']}_TSmap_background_resid.fits"
    my_apps.TsMap['nxpix'] = params['TSPix']
    my_apps.TsMap['nypix'] = params['TSPix']
    my_apps.TsMap['binsz'] = params["TSscale"]
    my_apps.TsMap['coordsys'] = "CEL"
    my_apps.TsMap['xref'] = params["ra"]
    my_apps.TsMap['yref'] = params["dec"]
    my_apps.TsMap['proj'] = 'STG'
    if not os.path.exists(f"{outdir}{params['name']}_TSmap_background_resid.fits") or clobber:
        my_apps.TsMap.run()
        

def generate_residuals(params, clobber, fheader, lock=None,outdir = "./"):
    '''
    Function to create residuals between the counts map and the model
    map. Simply generates a model map with the FermiTools, then takes
    the difference between that and a similar counts map
    
    Parameters
    ___________
    params : dict : parameter dict from read_parameters
    clobber : boolean : If true, overwrite existing files
    fheader : string : Unique ID added to avoid filename conflicts
    
    Returns
    ________
    None
    '''
    
    ## Generate a source model.
    ## I don't know how to do this in the python interface, so we 
    ## call the FermiTools from the shell with subprocess. Works,
    ## but is not elegant
    mmc = "gtmodel "
    mmc += f"srcmaps={outdir}{params['name']}{fheader}_srcmap.fits "
    mmc += f"srcmdl={outdir}fit_model{fheader}.xml "
    mmc += f"outfile={outdir}{params['name']}_Model{fheader}.fits "
    mmc += "irfs=CALDB "
    mmc += f"expcube={outdir}{params['name']}{fheader}_ltCube.fits "
    mmc += f"bexpmap={outdir}{params['name']}{fheader}_BinnedExpMap.fits"
    print (mmc)
    if not os.path.exists(f"{outdir}{params['name']}_Model{fheader}.fits") or clobber:
        checklocks(lock)
        subprocess.run(mmc,shell=True)
    
    ## Generate cmap for residuals:
    cmap_name = f'{outdir}{params["name"]}{fheader}_filtered_small_cmap.fits'
    npix = int(( np.sqrt(2) * params["roi"] / params["pix_sc"] ))
    my_apps.evtbin['evfile'] = f'{outdir}{params["name"]}{fheader}_filtered_gti.fits'
    my_apps.evtbin['outfile'] = cmap_name
    my_apps.evtbin['scfile'] = params["scfile"]
    my_apps.evtbin['algorithm'] = 'CMAP'
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
    if not os.path.exists(cmap_name) or clobber:
        checklocks(lock)
        my_apps.evtbin.run()
    
    ##Finally, generate the actual residuals

    model_hdu = fits.open(f"{outdir}{params['name']}_Model{fheader}.fits")
    cmap_hdu = fits.open(cmap_name)
    

    plt.imshow(cmap_hdu[0].data[::-1] - model_hdu[0].data[::-1], cmap = "seismic")
    plt.colorbar(label="Residual (Data - Model)")
    plt.savefig(f"{outdir}Residual_{fheader}.pdf")
    plt.close()
    
def find_max_TS(params):
    '''
    Function to determine the ideal window to get the maximum test
    statistic value. Intended to search for evidence of a significant
    detection.
    
    Parameters
    ___________
    params : dict : parameter dict from read_parameters
    
    Returns
    _______
    None
    '''
    

    logfile = "all_fits.csv"
    f = open(logfile , "w")
    f.write("#tstart,windowsize,TS\n")
    f.close()
    x0 = [0, 15]
    
    start_bounds = [params["min_start"] , params["max_start"] ]
    window_bounds = [ params["min_window"] , params["max_window"] ]
    boundaries = [ start_bounds , window_bounds  ] 
    '''
    result = opt.minimize(min_ts , x0)
    optf = open("optimize_res.txt" , "w")
    optf.write(str(result.x[0]) + "," + str(result.x[1]) + "," + str(result.fun))
    optf.write("," + str(result.success))
    optf.close()
    print (result)
    print (result.x)
    '''
    
    Bx , Bf , neval = ga.genetic_algorithm(opt_func , boundaries, popsize = params["popsize"],
                Niter = params["Niter"], nproc = params["nproc"], mutation_rate=params["mutation"])
    print (Bx , Bf)
    return Bx , Bf

def TS_Grid(params , starts , ends):
    ## Start by setting up our parameter array
    param_array = []
    grid_dir = params["grid_outdir"]


    status_log, log_starts, log_ends = check_status(params["grid_logfile"])
    with mp.Manager() as manager:
        lock = manager.Lock()
        id = 0
        
       
        

        for start in starts:
            for end in ends:
                if end <= start:
                    continue
                fheader = f"_grid_{start}_{end}"
                st = tpeak_to_met(start, params)
                et = tpeak_to_met(end, params)
                skip = False
                
                for i in range(len(log_starts)):
                    if st == log_starts[i] and et == log_ends[i]:
                        print (f"Skipping {start} to {end} as it is already in the log file")
                        skip = True
                        break
                if skip:
                    continue
                
                param_row = [params, st, et, False, fheader, params["grid_logfile"], lock, grid_dir]
      
                param_array.append(param_row)
        ## maxtasksperchild = 1 is designed to resolve a memory usage problem
        ## Probably mildly inneficient, but better than consuming many GB of
        ## RAM per process.
        print (len(param_array))
        
        results = []
        with mp.Pool(processes=params["nproc"], maxtasksperchild=1) as p:
            imres = p.imap(likelihood_wrapper , param_array, chunksize=1)
            for res in imres:
                results.append(res)
    

def run_analysis(params):
    

    ##Average Run First
    if params["gen_av"]:
        

        start_time = tpeak_to_met(params["avstart"] , params)
        end_time = tpeak_to_met(params["avend"] , params)
        av_dir = params["av_outdir"]
        mid_time = (start_time + end_time) / 2.0
        tmid = met_to_tpeak(mid_time , params)

        times, log_starts, log_ends = check_status(params["avg_logfile"])
        for i in range(len(log_starts)):
            if start_time == log_starts[i] and end_time == log_ends[i]:
                print (f"Skipping avgerage run as it is already in the log file")
                params["gen_av"] = False
                run_analysis(params) ## Go do everything else
                return 0
        
        print ("Beginning Likelihood Calculations")
        
        start = time.time()
        res = binned_likelihood(params, start_time, end_time, False, outdir = av_dir)
        end = time.time()
        
        print (f"Likelihood calculation finished; runtime is {(end-start)/60.} m")
        
        F , F_err , TS = res
        
        
        if TS < params["av_ts_lim"] and params["up_lim_av"]:
            start = time.time()
            Flux = compute_upper_lim(params , "", outdir = av_dir)[0]
            end = time.time()
            
            # ! Commented out FermiTools Upper limits; nominally no longer
            # ! necessary.
            '''
            s2 = time.time()
            
            try:
                Flux2 = FermiTools_UpperLim(params, "", outdir = av_dir)
            except:
                Flux2 = -99
            e2 = time.time()
            '''
            print (f"My Upper Limit Flux = {Flux}; runtime is {(end-start)/60.} m")
            f = open(params["avg_logfile"] , "a")
            f.write(str(F) + "," + str(0) + "," + str(TS) + "," + str(tmid))
            f.write("," + str(start_time) + "," + str(end_time))
            f.write("\n")
            f.close()
            
        else:
            f = open(params["avg_logfile"] , "a")
            f.write(str(F) + "," + str(F_err) + "," + str(TS) + "," + str(tmid))
            f.write("," + str(start_time) + "," + str(end_time))
            f.write("\n")
            f.close()
        
        print ("Model TS value is", TS)
        
        print ("Model Flux is " , F)
        #params["input_model"] = "fit_model.xml"
        
    ## Compute TS Maps
    if params["gen_ts"]:
        TS_Map(params, params["av_outdir"] + "fit_model.xml", False)
        
    ## Find Max TS
    if params["find_mts"]:
        mx = find_max_TS(params)
        
    
        
    ## Build a light curve
    if params["nproc"] == 1 and params["gen_lc"]:
        light_curve_singleproc(params , params["clobber"])
    elif params["gen_lc"]:
        start = time.time()
        light_curve_multiproc(params , params["clobber"])
        end = time.time()
        
        print (f"Total light curve runtime was {(end-start)/60} minues")
        
    if "gen_bck" in params.keys() and params["gen_bck"]:
        
        false_positive_rate(params , params["clobber"])
        
    if params["gen_grid"]:
        
        starts = np.arange(params["min_start"] , params["max_start"] , params["gridstep"])
        ends = np.arange(params["min_end"] , params["max_end"] , params["gridstep"])
        
        TS_Grid(params, starts , ends)
        

if __name__ == "__main__":
    
    ## Read in parameter file
    paramfile = sys.argv[1]
    params = read_parameters(paramfile)
    print_params(params)

    ## Run our analysis
    run_analysis(params)