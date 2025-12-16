'''
Script to read in Fermi Source Catalog, and return all info about a
given source. Takes in one command line argument, which should be the 
name of the source (from the Fermi catalogs).

Requires the fits version of the latest Fermi catalog; currently
available here : https://fermi.gsfc.nasa.gov/ssc/data/access/lat/

This repository includes a copy of the 14-year catalog, downloaded
on 12/5/2025


'''

from astropy.io import fits
import sys

source_name = sys.argv[1]

## Filename. Update if you download a newer source catalog
fname = "gll_psc_v35.fit"

hdu1 = fits.open(fname)

for source in hdu1[1].data:
    name = source[0]
    if name != source_name:
        continue
    for i in source:
        print (i)
