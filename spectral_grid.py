
import sys
sys.path.append("/Users/Peter/Documents/Research/Novae/NovaFermi/")
from analyze_fermi import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def new_fit_model(params, ind, cutoff, outdir):
    
    fheader = ""
    
    inmod =params["input_model"]
        
    src_name = outdir + f'{params["name"]}{fheader}_srcmap.fits'

    obs = BinnedObs(srcMaps=src_name,
            binnedExpMap=f'{outdir}{params["name"]}{fheader}_BinnedExpMap.fits',
            expCube=f'{outdir}{params["name"]}{fheader}_ltcube.fits',irfs='P8R3_SOURCE_V3')
    
    like = BinnedAnalysis(obs,f'{inmod}',optimizer="DRMNFB")
    
    
    src = like.logLike.getSource(params["name"])
    spec = src.spectrum()
    par = spec.getParam("Index1")
    par.setValue(ind)
    
    par2 = spec.getParam("Cutoff")
    par2.setValue(int(cutoff))
    
    likeobj=pyLike.NewMinuit(like.logLike)

    like.tol = 0.01
    
    res = like.fit(verbosity=1,covar=True,optObject=likeobj)
    like.optimizer = "NewMinuit"
    

    like.tol = 0.0001
    res = like.fit(verbosity=1,covar=True,optObject=likeobj)


    print("Source Convergence Status" , likeobj.getRetCode())

    like.logLike.writeXml(f'{outdir}fit_model{fheader}.xml')
    Nova_flux = like.flux(params["name"] , emin = params["emin"], emax=params["emax"])
    Nova_flux_err = like.fluxError(params["name"], emin=params["emin"], emax=params["emax"])
    
    TS = like.Ts(f'{params["name"]}')
    
    
    return Nova_flux , Nova_flux_err , TS

def spectral_grid(params, outdir):


    #inds = np.arange(-1 , -3, -0.1)
    #cutoffs = np.arange(500, 6000, 275)
    inds = np.linspace(-2.9 , -1 , 20)
    cutoffs = np.linspace(1275, 5000, 20)
    
    lock = None
    clobber = False
    fheader = ""
    tstart = tpeak_to_met(params["avstart"], params)
    tend = tpeak_to_met(params["avend"], params)
    print (tstart, tend)
    data_selection(params, tstart, tend, clobber , fheader, lock, outdir)
    

        
    bin_data(params , clobber , fheader, lock, outdir)

    
    lt_exp_maps(params , clobber , fheader, lock, outdir)

    
    gen_model(params, fheader, lock, outdir = outdir)
    
    
    gen_srcmap(params, clobber, fheader, lock, outdir)
    
    TS = np.zeros((len(inds), len(cutoffs)))
    Flux = np.zeros((len(inds), len(cutoffs)))
    final_inds = np.zeros((len(inds), len(cutoffs)))
    final_cuts = np.zeros((len(inds), len(cutoffs)))
    
    for i in range(len(inds)):
        for j in tqdm(range(len(cutoffs))):
            final_inds[i, j] = inds[i]
            final_cuts[i, j] = cutoffs[j]
            out_name = outdir + f"save_{inds[i]}_{cutoffs[j]}.csv"
            if os.path.exists(out_name):
                print ("Reading Model W/ Ind = " , inds[i] , " Cutoff = " , cutoffs[j])
                f = open(out_name)
                line = f.readlines()[0]
                Flux[i,j] = float(line.split(",")[0].strip())
                TS[i,j] = float(line.split(",")[1].strip())
                f.close()
                
            else:
                
                print ("Fitting Model W/ Ind = " , inds[i] , " Cutoff = " , cutoffs[j])
                try:
                    Flux[i, j] , error , TS[i, j] = new_fit_model(params , inds[i], cutoffs[j], outdir)
                except:
                    print (f"Model with Ind = {inds[i]} and Cutoff = {cutoffs[j]} failed to fit. Skipping.")
                    continue
                
                f = open(out_name , "w")
                out_line = str(Flux[i,j]) + "," + str(TS[i,j]) + "," + str(error)
                f.write(out_line)
                f.close()
                

            

    plt.scatter(final_inds.flatten(), final_cuts.flatten(), c = TS.flatten())
    plt.xlabel("Spectral Index")
    plt.ylabel("Cutoff Energy (MeV)")
    plt.colorbar(label = "TS")
    plt.scatter(-1.9, 4300, c = "red", marker = "x")
    plt.savefig(outdir + "spec_grid_TS.pdf")
    plt.close()
    
    plt.scatter(final_inds.flatten(), final_cuts.flatten(), c = Flux.flatten())
    plt.xlabel("Spectral Index")
    plt.ylabel("Cutoff Energy (MeV)")
    plt.colorbar(label = "Flux (erg/s/cm$^2$)")
    plt.scatter(-1.9, 4300, c = "red", marker = "x")
    plt.savefig(outdir + "spec_grid_flux.pdf")
    plt.close()
    
if __name__ == "__main__":
    ## Read in parameter file
    paramfile = sys.argv[1]
    params = read_parameters(paramfile)
    print_params(params)
    
    spectral_grid(params, "spec_grid/")