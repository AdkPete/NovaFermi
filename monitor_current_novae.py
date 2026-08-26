'''
written by Peter Craig on 8/20/26

Designed to monitor the population of current novae in Fermi, to check
for significant detections. Will summarize with a table contanining each
system being monitored and the maximum recovered TS value.
'''

import os
import sys
import subprocess
import tabulate
import numpy as np
import shutil
import datetime
import analyze_fermi as af
import plot_lc as plc
from astropy.io import fits

## To catch file truncation warnings when downloading LAT data.
import warnings


def validate_fits(path):
    '''
    Utility function to verify our data files. Returns True if the file 
    is valid, False if not.
    '''
    try:
        with fits.open(path, memmap=False, lazy_load_hdus=False) as hdul:
            for hdu in hdul:
                if hdu.verify_datasum() != 1:
                    return False

                if hdu.verify_checksum() != 1:
                    return False

        return True

    except Exception:
        return False
    
def send_notification_email(subject, body):
    import smtplib
    from email.mime.text import MIMEText

    # Email configuration
    sender_email = os.environ['EMAIL_USER']
    receiver_email = os.environ['PRIMARY_EMAIL']
    password = os.environ['EMAIL_PSWD']
    
    ## Send email

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        
def get_current_novae(fname = "current_novae.csv"):
    '''
    Get the current list of novae from the Fermi website.
    '''
    f = open(fname, 'r')
    name = []
    for i in f.readlines():
        if i[0] == "#":
            continue
        name.append(i.strip())
         
    return name

def get_current_week_number(time=datetime.datetime.now(tz=datetime.timezone.utc)):
    '''
    Function to get the current week number, used for downloading appropriate
    Fermi weekly data files.
    '''
    #today = datetime.datetime.now(tz=datetime.timezone.utc)
    #time = datetime.datetime(2026, 8, 1, 11, 59, 59, tzinfo=datetime.timezone.utc)
    ref = datetime.datetime(2026, 8, 13, 0, 0, 0, tzinfo=datetime.timezone.utc)
    ref_week = 950
    delta_weeks = (time - ref).days // 7
    week_number = ref_week + delta_weeks
    
    return week_number
    
def update_weekly_data():
    '''
    Function to get latest Fermi weekly data files. Intended only for 
    monitoring currently active novae, so we only keep the last 26 weeks of data.
    Older files will be removed as new files are downloaded.
    '''
    
    cwd = os.getcwd()
    os.chdir(os.environ['LAT_DATA'])
    
    current_week = get_current_week_number()
    
    download_weeks = [current_week - i for i in range(26)]
    download_weeks.append(current_week + 1) # In case the file for the next week exists
    
    down_url = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/photon/'
    
    for week in download_weeks:
        print ("Downloading Data for week {}".format(week))
        url = down_url + f'lat_photon_weekly_w{week}_p305_v001.fits'
        command = f'wget -m -P . -nH --cut-dirs=4 -np -e robots=off {url}'
        fname = f'weekly/photon/lat_photon_weekly_w{week}_p305_v001.fits'
        output = subprocess.run(command, shell=True)
        
        if os.path.exists(fname):
            if validate_fits(fname):
                print (f"Successfully downloaded and validated file for week {week}.")
            else:
                print (f"Downloaded file for week {week} is corrupted.")
                os.remove(fname)
                print (f"Removed corrupted file for week {week}.")
                print ("Re-downloading...")
                output = subprocess.run(command, shell=True)
                if validate_fits(fname):
                    print (f"Successfully re-downloaded and validated file for week {week}.")
                else:
                    print (f"Re-downloaded file for week {week} is still corrupted. Skipping this week.")
                    os.remove(fname)
                    raise ValueError
        
    if len(os.listdir('weekly/photon/')) > 26:
        # Remove the oldest files
        files = sorted(os.listdir('weekly/photon/'))
        for f in files[:-25]:
            os.remove(os.path.join('weekly/photon/', f))
    
    ## Last, but not least, get updated spacecraft file.
    spacecraft_url = 'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/spacecraft/'
    for week in download_weeks:
        print ("Downloading Spacecraft Data for week {}".format(week))
        url = spacecraft_url + f'lat_spacecraft_weekly_w{week}_p310_v001.fits'
        command = f'wget -m -P . -nH --cut-dirs=4 -np -e robots=off {url}'
        
        output = subprocess.run(command, shell=True)
        fname = f'weekly/spacecraft/lat_spacecraft_weekly_w{week}_p310_v001.fits'
        if os.path.exists(fname):
            if validate_fits(fname):
                print (f"Successfully downloaded and validated spacecraft file for week {week}.")
            else:
                print (f"Downloaded spacecraft file for week {week} is corrupted.")
                os.remove(fname)
                print (f"Removed corrupted spacecraft file for week {week}.")
                print ("Re-downloading...")
                output = subprocess.run(command, shell=True)
                if validate_fits(fname):
                    print (f"Successfully re-downloaded and validated spacecraft file for week {week}.")
                else:
                    print (f"Re-downloaded spacecraft file for week {week} is still corrupted. Skipping this week.")
                    os.remove(f'weekly/spacecraft/lat_spacecraft_weekly_w{week}_p310_v001.fits')
                    raise ValueError

    files = []
    for i in os.listdir('weekly/spacecraft/'):
        if i.startswith('lat_spacecraft_weekly_w') and i.endswith('.fits'):
            files.append(i)
            
    if len(files) != 26:
        print ("Warning: Expected 26 spacecraft files, but found {}. Please check the downloads.".format(len(files)))
        
    if len(files) > 26:
        print ("Warning: More than 26 spacecraft files found. Removing oldest files.")
        files = sorted(files)
        for f in files[:-26]:
            os.remove(os.path.join('weekly/spacecraft/', f))

    ## Last step is to combine the spacecraft files into a single file for analysis.
    os.chdir('weekly/spacecraft/')
    subprocess.call('ls lat_spacecraft_weekly_w*.fits > weekly.list', shell=True)
    ft1 = "ftmerge @weekly.list lat_spacecraft_weekly_merged.fits"
    ft1 += " lastkey='TSTOP,DATE-END' clobber=yes"
    subprocess.call(ft1, shell=True)
    
    os.chdir(cwd)

    
def setup_new_nova(name):
    '''
    Set up a new nova for monitoring.
    '''
    
    print ("Setting up new analysis for nova: {}".format(name))
    result_dir = os.environ['FERMI_MONITOR']
    dirname = os.path.join(result_dir, name)
    # make a new directory for the nova
    if os.path.exists(dirname):
        return 0
    
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    
    shutil.copyfile("monitoring_parameters.yaml", os.path.join(dirname, "parameters.yaml"))
    
    
    input("Press Enter to continue after editing parameters.yaml for nova: {}".format(name))
    cwd = os.getcwd()
    os.chdir(dirname)
    
    params = af.read_parameters(os.path.join(dirname, "parameters.yaml"))
    ## Setup input model so that we can run automatically later
    
    tstart = af.tpeak_to_met(-5, params)
    tend = af.tpeak_to_met(5, params)
    os.mkdir("temp/")
    af.data_selection(params, tstart, tend, False, "", None, "temp/")
    af.bin_data(params , False , "", None, "temp/")
    af.lt_exp_maps(params , False , "", None, "temp/")
    af.gen_model(params, "", None, outdir = "temp/", skip_pruning = False)
    
    
    
    os.chdir(cwd)
    
def reset_results(name):
    '''
    Resets the results of a nova. Deletes results based on bins from the last two
    days, intended to ensure that all analysis is done with complete data sets.
    '''
    dirname = os.path.join(os.environ['FERMI_MONITOR'], name)
    params= af.read_parameters(os.path.join(dirname, "parameters.yaml"))
    cwd = os.getcwd()
    os.chdir(dirname)
    
    if not os.path.exists(params["grid_logfile"]):
        os.chdir(cwd)
        return 0
    
    max_met = 0
    f = open(params["grid_logfile"], 'r')
    for i in f.readlines():
        if "Flux" in i:
            continue
        
        sl = i.split(",")
        
        max_met = max(max_met, float(sl[5]))
    f.close()
    
    new_logfile = ""
    f = open(params["grid_logfile"], 'r')
    for i in f.readlines():
        if "Flux" in i:
            new_logfile += i
            continue
        
        sl = i.split(",")
        
        end_met = float(sl[5])
        if end_met > max_met - 2.0 * 3600 * 24: # 2 days in seconds
            continue
        new_logfile += i
        
    f.close()
    f = open(params["grid_logfile"], 'w')
    f.write(new_logfile)
    f.close()
    os.chdir(cwd)
    
    
def mark_result_as_incomplete(name):
    '''
    Marks the results of a nova as incomplete.
    '''
    dirname = os.path.join(os.environ['FERMI_MONITOR'], name)
    params= af.read_parameters(os.path.join(dirname, "parameters.yaml"))
    cwd = os.getcwd()
    os.chdir(dirname)
    
    new_logfile = ""
    f = open(params["grid_logfile"], 'r')
    
    current_met = af.cal_to_met(datetime.datetime.now(tz=datetime.timezone.utc))
    
    for i in f.readlines():
        if "Flux" in i:
            new_logfile += i
            continue
        
        sl = i.split(",")
        if abs(float(sl[5]) - current_met) > 2.0 * 3600 * 24: # 2 days in seconds
            new_logfile += i
            continue
        else:
            new_logfile += i.strip() + ",maybe_incomplete\n"
        
    f.close()
    f = open(params["grid_logfile"], 'w')
    f.write(new_logfile)
    f.close()
    os.chdir(cwd)
    
    os.chdir(cwd)

def cleanup_monitoring_data(name):
    '''
    Cleans up the monitoring data by removing old files and directories.
    '''
    dirname = os.path.join(os.environ['FERMI_MONITOR'], name)
    params= af.read_parameters(os.path.join(dirname, "parameters.yaml"))
    cwd = os.getcwd()
    os.chdir(dirname)
    os.chdir("grid_results/")
    
    for i in os.listdir("."):
        if i.endswith(".fits") or i.endswith(".log"):
            os.remove(i)
    
    os.chdir('../prune_files/')
    for i in os.listdir("."):
        if i.endswith(".fits") or i.endswith(".log"):
            os.remove(i)
    os.chdir(cwd)
    
def realtime_monitor():
    '''
    Function to monitor the current novae in real time. Will check for new data and run analysis.
    '''
    last_update = None
    while True:
        week = get_current_week_number()
        current_file = f'weekly/photon/lat_photon_weekly_w{week}_p305_v001.fits'
        breakpoint()
        update_weekly_data()
        
def main_loop(table_only=False, reset=True):
    '''
    Main function to monitor the current novae
    '''
    
    ## First, get the list of novae to analyze

    names = get_current_novae()

    ## Front-load any user setup requirements.
    if not table_only:
        update_weekly_data()
        for i in range(len(names)):
            
            setup_new_nova(names[i])
    

    ## Ok, time to do some analysis.

    novae = []
    max_test_statistics = []
    
    for i in range(len(names)):
        print ("Analyzing nova: {}".format(names[i]))
        cwd = os.getcwd()
        os.chdir(os.path.join(os.environ['FERMI_MONITOR'], names[i]))
        params = af.read_parameters("parameters.yaml")
        current_met = af.cal_to_met(datetime.datetime.now(tz=datetime.timezone.utc))
        
        time_since_peak = af.met_to_tpeak(current_met, params)
        stop = int(time_since_peak) + 2
        params['max_end'] = stop
        if not table_only:
            ## Run the analysis
            if reset:
                reset_results(names[i])
            cleanup_monitoring_data(names[i])
            af.run_analysis(params)
            cleanup_monitoring_data(names[i])
            #mark_result_as_incomplete(names[i])
            
        ## plot the results and get max TS
        max_ts , time_of_max = plc.TS_Grid(params, return_TS = True)
        print ("Maximum TS for {} is {}".format(names[i], max_ts))
        if max_ts >= 25:
            subject = "Fermi Nova Alert: {}".format(names[i])
            body = "The nova {} has been detected with a maximum TS of {} at time {}".format(names[i], max_ts, time_of_max)
            send_notification_email(subject, body)
            
        os.chdir(cwd)
        
        novae.append(names[i])
        max_test_statistics.append(max_ts)
    
    print ("Summary of current novae analysis:")
    table = tabulate.tabulate(list(zip(novae, max_test_statistics)), headers=["Nova", "Max TS"], tablefmt = 'fancy_grid')
    print (table)
    
    if table_only:
        table = tabulate.tabulate(list(zip(novae, max_test_statistics)), headers=["Nova", "Max TS"], tablefmt = 'simple')
        send_notification_email("Fermi Nova Monitoring Summary", table)
        
    
if __name__ == "__main__":
    #update_weekly_data()
    realtime_monitor()
    if len(sys.argv) > 1 and sys.argv[1] == "table_only":
        print ("Running in table only mode. No new analysis will be performed.")
        main_loop(table_only=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "noreset":
        print ("Running in no reset mode. No data will be reset.")
        main_loop(table_only=False, reset=False)
    else:
        print ("Running in full analysis mode. New analysis will be performed.")
        
        main_loop(table_only=False)
        main_loop(table_only=True)
    #get_current_week_number()
    #update_weekly_data()