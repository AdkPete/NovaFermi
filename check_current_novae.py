from plot_lc import TS_Grid
from analyze_fermi import read_parameters
import os, sys
import matplotlib.pyplot as plt
from tabulate import tabulate

def check_all():
    
    result_dir = os.environ['FERMI_MONITOR']

    rows = [ ]
    for i in os.listdir(result_dir):
        ## check if is a directory
        if os.path.isdir(os.path.join(result_dir, i)):
            pwd = os.getcwd()
            os.chdir(os.path.join(result_dir, i))
            paramfile = "parameters.yaml"
            if not os.path.exists(paramfile):
                print(f"parameters.yaml not found in {i}")
                os.chdir(pwd)
                continue
            params = read_parameters(paramfile)
            if not os.path.exists("likelihood_results.csv"):
                print(f"likelihood_results.csv not found in {i}")
                os.chdir(pwd)
                continue
            '''
            if not os.path.exists(os.path.join(params["grid_outdir"], "grid_results.csv")):
                print(f"grid_results.csv not found in {i}")
                os.chdir(pwd)
                continue
            '''
            
            
            max_TS, center = TS_Grid(params, return_TS = True, show = True, title = i)
            
            print (f"Maximum TS for {i} is {max_TS}")
            rows.append([i, max_TS])
    print (tabulate(rows, headers = ["Nova", "Max TS"], tablefmt = "fancy_grid"))
    
def monitor_nova(name):
    result_dir = os.environ['FERMI_MONITOR']
    pwd = os.getcwd()
    os.chdir(os.path.join(result_dir, name))
    paramfile = "parameters.yaml"
    if not os.path.exists(paramfile):
        print(f"parameters.yaml not found in {name}")
        os.chdir(pwd)
        return
    params = read_parameters(paramfile)
    if not os.path.exists(os.path.join(params["grid_outdir"], "grid_results.csv")):
        print(f"grid_results.csv not found in {name}")
        os.chdir(pwd)
        return
    


    max_TS, center = TS_Grid(params, return_TS = True, show = True, title = name)
    #time.sleep(60)

    print (f"Maximum TS for {name} is {max_TS}")
if __name__ == "__main__":
    if len(sys.argv) == 1:
        check_all()
    else:
    
        monitor_nova(sys.argv[1])
    