from plot_lc import TS_Grid
from analyze_fermi import read_parameters
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

reuslt_dir = os.environ['FERMI_MONITOR']

rows = [ ]
for i in os.listdir(reuslt_dir):
    ## check if is a directory
    if os.path.isdir(os.path.join(reuslt_dir, i)):
        pwd = os.getcwd()
        os.chdir(os.path.join(reuslt_dir, i))
        paramfile = "parameters.yaml"
        if not os.path.exists(paramfile):
            print(f"parameters.yaml not found in {i}")
            os.chdir(pwd)
            continue
        
        params = read_parameters(paramfile)
        
        max_TS, center = TS_Grid(params, return_TS = True, show = True, title = i)
        
        print (f"Maximum TS for {i} is {max_TS}")
        rows.append([i, max_TS])
print (tabulate(rows, headers = ["Nova", "Max TS"], tablefmt = "fancy_grid"))