import xml.etree.ElementTree as ET
from astropy.io import fits

# todo for integration into analysis pipeline:
# 1. Port these two functions into analyze_fermi.py
# Everything next happens inside the gen_model function of analyze fermi.
# call load_source_cat. Can get the source cat filename from the param array
# call prune_model
# Update name of input model in param array to the new xml file

## That should get this running. Next up are a few updates to make this really
# do what we want it to do.

# 1. Change name of pruned_model.xml to include name of nova and match other filenames
# 2. Adjust cuts to keep any source with Variability_index > 30.58
# 3. Adjust cuts to keep any source with ROI_Center < 10
# 4. From the energy flux, calculate expected number of photons from each source
        ## assuming all photons arrive with E = 100 MeV (maximizes number of photons for a given energy flux)
# 5. Remove sources with 10 < ROI_Center < 15 and expected number of photons < 0.1
# 6. Remove sources with ROI_Center > 15 with expected number of photons < 0.5


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

def prune_model(model_fname):

    tree = ET.parse("model.xml")
    root = tree.getroot()

    print ("There are ", len(root), " sources in the model file.")
    bad_sources = []
    for source in root:
    
        spec = source.find("spectrum")
        spat = source.find("spatialModel")

        ROI_Center = source.get("ROI_Center_Distance")
        print (ROI_Center)
        if ROI_Center is None:
            ROI_Center = 0.0
            ## Is our nova, leave in file

            continue
        
        if float(ROI_Center) > 15:
            ## Too far from our nova, remove from file
            print (f"Removing {source.get('name')} from model file. ROI_Center_Distance = {ROI_Center}")           
            bad_sources.append(source)

    for sr in bad_sources:
        root.remove(sr)
        
    print ("There are ", len(root), " sources  in the model file.")
    ## Save new model file
    tree.write("pruned_model.xml")
    

scat = load_source_cat("/Users/Peter/Documents/Research/Novae/Fermi/Diffuse/gll_psc_v38.xml")
prune_model("model.xml")

