import os

files = []
for i in os.listdir("./"):
    if "PH" in i and ".fits" in i:
        continue
    if "SC" in i and ".fits" in i:
        continue
    if "parameters.txt" in i:
        continue
    if "input_model" in i:
        continue
    if ".py" in i:
        continue 
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