import sys
import analyze_fermi as af
from monitor_current_novae import get_end_time
import datetime

params = af.read_parameters(sys.argv[1])
if len(sys.argv) > 2:
    dtime = datetime.datetime.strptime(sys.argv[2], "%Y-%m-%dT%H:%M:%S")
    dtime = dtime.replace(tzinfo=datetime.timezone.utc)
    tmet = af.cal_to_met(dtime)
else:
    tmet = get_end_time(params, met=True)

tpeak = af.met_to_tpeak(tmet, params)
print (tpeak)