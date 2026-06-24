import os
import argparse, sys
parser = argparse.ArgumentParser(description='Delete all files in the current directory except for PH, SC, .yaml, .py, input_model and parameters.txt files.')
parser.add_argument('--hard', action='store_true', help='Delete result directories as well')
args = parser.parse_args()

files = []
dirs = []
for i in os.listdir("./"):
    
    if "PH" in i and ".fits" in i:
        continue
    if ".yaml" in i:
        continue
    if "SC" in i and ".fits" in i:
        continue
    if "parameters.txt" in i:
        continue
    if "input_model" in i:
        continue
    if ".py" in i:
        continue 
    if "data" in i.lower() or "figures" in i.lower():
        continue
    if args.hard and "." not in i:
        
        dirs.append(i)
    if "." not in i:
        continue
    files.append(i)
    
print (f"Warning: This script is about to delete {len(files)} files!!")
ftypes = {}
for fn in files:
    ft = fn.split(".")[-1]
    if ft in ftypes.keys():
        ftypes[ft] += 1
    else:
        ftypes[ft] = 1
        
print ("These files include:")

for ft in ftypes.keys():
    print (f"{ftypes[ft]} {ft} Files")

print ("\n\n")
check1 = input("Do you want to proceed? (Y/N) --> ")

if check1.lower() != "y":
    print ("Cancelling")
    exit()

check2 = input(f"Final Check: Delete {len(files)} files? (Y/N) --> ")

if check2.lower() != "y":
    print ("Cancelling")
    exit()
for i in files:
    os.remove(i)
    
## Same, but now for directories
if args.hard:
    print (f"Warning: This script is about to delete {len(dirs)} directories!!")
    check3 = input("Do you want to proceed? (Y/N) --> ")

    if check3.lower() != "y":
        print ("Cancelling")
        exit()

    check4 = input(f"Final Check: Delete {len(dirs)} directories? (Y/N) --> ")

    if check4.lower() != "y":
        print ("Cancelling")
        exit()
    for i in dirs:
        os.rmdir(i)