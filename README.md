
This package is intended to analyze Fermi data to search for novae. Intended
to be a straightforward way to consistently approach searching for
significant detections of novae by Fermi.

Actually running this analysis is easy; just run the analyze_fermi.py script
and pass in the name of a parameter file as a command line argument. For a
suitably set up parameter file, this will run almost all the analysis 
automatically. An example parameter file is listed in this repository as
parameters.txt.

For examples of actual nova analyses, there are two example cases included
in this repository. Each of these has functional parameter files to run
all of the analysis that is currently implemented (though with parameters
chosen to keep the runtime low compared to the defaults). If you want to test
your installation, run analyze_fermi.py with the test_params.txt, then run
plot_lc.py with the same parameter file. There is a directory called test_figures
that contains the correct outputs; compare the results in your Figures/ directory
with these.

For a typical case, you will want to change the following in this file:

1. ra / dec for your source
2. name : Set the name of your target
3. peak : Set the peak time (or any other reference time. Other times are listed in days relative to this time)
4. nproc : number of processors you want to use (for functions that support multiple processes)
5. Set Yes for any analysis you want to run
6. Set parameters for any analysis that you selected above

Summary of the different analysis options:

Average Run:

This runs a single likelihood fit using the data between two times. Will
produce a model fit, flux and TS value for this data set.

Find MTS:

Runs an optimizer on the TS values to find the data window (within some
boundaries) that yields the maximum TS value. Not recommended for most cases,
instead you should typically use the grid option below. This function is
largely experimental, and ideally should work for cases where you want to search
for a large TS value without the cpu cost associated with a full grid search.
However, the grid search is great for checking how the TS is varying over time.

Generate TSMap (gen_ts):

Generates a TSMap from the average run, which you can use to find any sources
that are missing from the source list. Also should highlight the target source
if it is significantly detected. Will produce two TS maps, one with the target
included in the source model (which ideally will not have any high TS values)
and one without the source model (which may have a large TS value in the center,
if the target is detected).

Generate Light Curve (gen_lc):

Builds a light curve. You can set the size of the data windows and the 
separation between points in the parameter file, as well as the threshold
for a detection and wether you want to generate upper limits.

Generate Grid (gen_grid):

Will run a grid of likelihood analysis that vary the start and end times.
Useful for exploring the time around the peak of a nova to search for
significant detections, and it can constrain the duration of the gamma-ray emission.
This can get slow depending on the parameter space you want to search;
on my system, it takes ~6-7 minutes per likelihood run without upper limits,
and the full grid that I often generate includes 635 runs. Running on 10
cpus, this takes ~7 hours to fully run. So, you may want to reduce the number
of points from the default if you want a faster analysis.


Some notes here for best results:

First, when setting up an input model, make sure that the name of your target
in the input model matches exactly with the name listed in the parameter file.
Secondly, it is often a good idea to run a quick average analysis before
any other analysis, to provide a starting point for future model fits that will
be closer to the right solution. This can be done over a large
data range to work out the right background solution. To use this as a base,
just set the produced fit_model.xml as the input model in your parameter file.
You may want to reset the spectral model for your source.

When running a model grid, turn up_lim_lc to No. Otherwise, we end up 
spending some time computing upper limits for many of the grid cells,
which are not used for anything at the moment.


Detailed Documentation of the internal functions can be found here:

HTML format:

html_docs/index.html

PDF format:

documentation.pdf

