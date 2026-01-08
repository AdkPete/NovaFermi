import os

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
    os.remove(i)