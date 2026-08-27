'''
Short script to compute the time since peak of the end of the current weekly data,
or if provided a custom time in UTC. 
'''

import sys, os
import analyze_fermi as af
from monitor_current_novae import get_current_week_number
import datetime
from astropy.io import fits
params = af.read_parameters(sys.argv[1])
if len(sys.argv) > 2:
    time = sys.argv[2]
else:
    week = get_current_week_number()
    data_dir = os.environ['LAT_DATA']
    fname = os.path.join(data_dir, f"weekly/photon/lat_photon_weekly_w{week}_p305_v001.fits")
    hdu = fits.open(fname)
    end_time = hdu[0].header['DATE-END'].split(".")[0]
    time = end_time
    
dtime = datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
dtime = dtime.replace(tzinfo=datetime.timezone.utc)
tmet = af.cal_to_met(dtime)
tpeak = af.met_to_tpeak( tmet, params)
print (tpeak)