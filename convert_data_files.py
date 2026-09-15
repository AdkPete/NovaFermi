import os, sys
import analyze_fermi as af

params = af.read_parameters(sys.argv[1])

## The goal is to migrate the old TS Grid data file into the new format.

fname = params["grid_outdir"] + "/" + "grid_results.csv"
data_start = af.tpeak_to_met(-6 * 30 , params)

new_file = ""
new_fname = params["result_log"]

f = open(fname, 'r')
header = True
for i in f.readlines():
    if header:
        header = False
        continue
    sl = i.strip().split(",")
    new_file += sl[0] + "," + sl[1] + "," + sl[2] + "," + "-1" + "," + sl[4]
    new_file += "," + sl[5] + "," + "None" + "," + str(data_start) + ","
    new_file += sl[6] + "\n"
    
f.close()

nf = open(new_fname, 'w')
nf.write(new_file)
nf.close()