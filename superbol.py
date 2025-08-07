#!/usr/bin/env python

version = '2.5'

'''
    SUPERBOL: Supernova Bolometric Light Curves
    Written by Matt Nicholl, 2015-2022

    Version 2.5 : Make error penalties slightly less harsh for large extrapolations (MN)
    Version 2.4 : Can now specify individually for each band whether it's in AB or Vega mags (MN)
    Version 2.3 : Allow using more than one band at a time as the reference band (though makes colour-based extrapolations risky!) (MN)
    Version 2.2 : Prompt user for initial guess in temperature and radius for the first epoch. Can also use use T/R from previous step as the initial guess for later steps (MN)
    Version 2.1 : Allow limit to MJDs to slice up light curve after plotting. Also duplicate Gaia G == E in case computer can't differentiate _g.txt and _G.txt (MN)
    Version 2.0 : Implement Nicholl+ 2017 / Yan+ 2018 BB absorption function in UV for SED fits, removes need to fit UV/optical separately (MN)
    Version 1.12: Fix bug in default answers to absolute/apparent mags - thanks to Aysha Aamer for catching (MN)
    Version 1.11: Add NEOWISE bands (MN)
    Version 1.10: If no overlap in temporal coverage with reference band, extrapolate to nearest epoch for colour (MN)
    Version 1.9 : Add w band (MN)
    Version 1.8 : Fix bug in suppression integral - thanks Sebastian Gomez (MN)
    Version 1.7 : Fix bug introduced in 1.6 where extinction/Swift corrections not always applied (MN)
    Version 1.6 : Save interpolations before applying other corrections (MN)
    Version 1.5 : Add prompt to convert Swift AB to Vega (MN)
    Version 1.4 : Narrow date range for finding max of polynomial fit to peak (MN)
    Version 1.3 : Minor tweaks to output plots (MN)
    Version 1.2 : Add extinction correction as an option (MN)
    Version 1.1 : Add bibliographic reference, output file now includes K-correction info (MN)
    Version 1.0 : Release version, Nicholl 2018 RNAAS (MN)
    Version 0.17: Fix bug to write nans instead of blanks when BB fit fails (MN)
    Version 0.16: Correct inconsistency in x axis labels, automatically exit if <2 filters used (MN)
    Version 0.15: Plot temperature and radius, other small figure adjustments (MN)
    Version 0.14: Fixed bug where having two reference epochs the same broke manual interpolation (MN)
    Version 0.13: Give user control over whether to fit UV separately, improve commenting and output files, change min integration wavelength to 100A (MN)
    Version 0.12: Change UV suppression to power law (lambda/lambda_max)^x following Nicholl, Guillochon & Berger 2017 (MN)
    Version 0.11: Added ATLAS c and o filters (MN)
    Version 0.10: Added Gaia G filter. Automatically sort wavelength array when calculating Luminosity. Updated constants in bbody. Added option to change cosmologies with astropy. (SG)
    Version 0.9 : Only do separate UV fit if > 2 UV filters (MN)
    Version 0.8 : New feature! Can choose to shift SED to rest-frame for data with no K-correction (MN)
    Version 0.7 : Improved handling of errors (MN)
    Version 0.6 : Tidied up separate blackbody integrations in UV/NIR (MN)
    Version 0.5 : Specifying files on command line now COMMA separated to handle 2-digit indices (MN)
    Version 0.4 : Fixed bug in blackbody formula - missing factor of pi led to overestimate of radius (MN)
    Version 0.3 : Added GALEX NUV and FUV bands in AB system (MN)
    Version 0.2 : Swift effective wavelengths now match Poole et al 2008 (update by MN)
    Version 0.1 : Origin and veracity of all zeropoints checked by SJS. Comments added, with more details in superbol.man file. Archived this version in /home/sne/soft
    Version 0   : Written by Matt Nicholl (QUB), 2015

    Computes pseudobolometric light curves and estimates full bolometric with blackbody corrections

    See superbol.man for the manual file and more details.

    Requirements and usage:
    Needs numpy, scipy and matplotlib

    To-do:
        - set error floor for interpolation to ref band error
        - make compatible with other inputs (Open Supernova Catalog, output from psf.py)

    Input files should be called SNname_filters.EXT, eg PTF12dam_ugriz.txt, LSQ14bdq_JHK.dat, etc
    Can have multiple files per SN with different filters in each

    Format of files must be:
    MJD filter1 err1 filter2 err2...

    MJD can be replaced by phase or some other time parameter, but must be consistent between files.

    Important: Bands must be in their common systems -- AB mag for ugrizy and GALEX, Vega mag for UBVRIJHK and Swift (S=UVW2 D=UVM2 A=UVW1)
    Important : Order of filter magnitudes in file must match order of filters in filename.

    Output of each run of the code will contain all the filters used in the integration in the filenames


    Steps:
     - Find files associated with SN and determine available filters and data
     - Correct for time dilation, distance, and approximate K-correction if desired
     - Map light curves in each filter to a common set of times (typically defined by the filter with most observations)
        - Interpolation options: linear between neighbouring epochs or polynomial fits (user determines order of polynomial interactively)
        - Extrapolation: using polynomials or assuming constant colour with respect to reference filter. Large extrapolations = large uncertainties!
        - Save interpolated light curves for reproducability!
     - Fit blackbodies to SED at each epoch (most SNe can be reasonably approximated by blackbody above ~3000 A). In UV, user can choose to:
        - fit SED over all wavelengths with single blackbody
        - use a simple prescription for line blanketing at UV wavelengths, defined as L_uv(lambda < cutoff) = L_bb(lambda)*(lambda/cutoff)^x, where x is chosen by user. User can choose cutoff wavelength and suppression index (Nicholl+ 2017, Yan+ 2018)
    - Numerically integrate observed SEDs, and account for missing UV and NIR flux using blackbody extrapolations. NIR is easy, UV used options described above
    - Save outputs:
        - interpolated-lcs_<SN>_<filters>.txt = multicolour light curves mapped to common times. Footer gives methods of interpolation and extrapolation. If file exists, can be read in future to skip interpolating next time.
        - bol_<SN>_<filters>.txt = main output. Contains pseudobolometric light curve, integrated trapezoidally, and bolometric light curve including the additional BB corrections, and errors on each. Footer gives filters and method of UV fitting.
        - logL-obs_<SN>_<filters>.txt = same pseudobolometric (observed) light curve, in convenient log form
        - logL-bb_<SN>_<filters>.txt = light curve with the BB corrections, in convenient log form
        - BB-params_<SN>_<filters>.txt = fit parameters for blackbodies: T, R and inferred L from Stefan-Boltzmann law (can compare with direct integration method).

    Recommended practice: run once with ALL available filters, and fit missing data as best you can using light curve interpolations. Then re-run choosing only the well-observed filters for the integration. You can compare results and decide for yourself whether you have more belief in the "integrate all filters with light curve extrapolations" method or the "integrate only the well-sampled filters and account for missing flux with blackbodies" method.
    '''


import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as itg
from scipy.optimize import curve_fit
from scipy.interpolate import interpolate as interp
import glob
import sys
import os
# If you don't have astropy, can comment this out, and uncomment cosmocalc routine
from astropy.coordinates import Distance
#from lmfit import Parameters, Model
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

# print 'cool' logo
print('\n    * * * * * * * * * * * * * * * * * * * * *')
print('    *                                       *')
print('    *        Welcome to `SUPER BOL`!        *')
print('    *   SUPernova BOLometric light curves   *')
print('    *                                       *')
print('    *                ______                 *')
print('    *               {\   */}                *')
print('    *                 \__/                  *')
print('    *                  ||                   *')
print('    *                 ====                  *')
print('    *                                       *')
print('    *   Matt Nicholl (2018, RNAAS, 2, 230)  *')
print('    *                 V'+version+'                *')
print('    *                                       *')
print('    * * * * * * * * * * * * * * * * * * * * *\n\n')

# interactive plotting
plt.ion()

# Define some functions:

def easyint(x,y,err,xref,yref):
    '''
    Adapt scipy interpolation to include extrapolation for filters missing early/late data
    Originally based on `bolom.py` by Enrico Cappellaro (2008)
    Returns light curve mapped to reference epochs and errors on each point
    '''
    ir = (xref>=min(x))&(xref<=max(x))
    # for times where observed and reference band overlap, do simple interpolation
    yint = interp.interp1d(x[np.argsort(x)],y[np.argsort(x)])(xref[ir])
    yout = np.zeros(len(xref),dtype=float)
    # For times before or after observed filter has observations, use constant colour with reference band
    try:
        ylow = yint[np.argmin(xref[ir])]-yref[ir][np.argmin(xref[ir])]+yref[xref<min(x)]
    except:
        ylow = y[0]-yref[-1]+yref[xref<min(x)]
    try:
        yup  = yint[np.argmax(xref[ir])]-yref[ir][np.argmax(xref[ir])]+yref[xref>max(x)]
    except:
        yup  = y[-1]-yref[0]+yref[xref>max(x)]
    yout[ir] = yint
    yout[xref<min(x)] = ylow
    yout[xref>max(x)] = yup
    errout = np.zeros(len(xref),dtype=float)
    # put error floor of 0.1 mag on any interpolated data
    errout[ir] = max(np.mean(err),0.1)
    # Compute error as random sum of average error in band plus 0.1 (0.05) mag for every 10 days extrapolated late (early)
    errout[xref<min(x)] = np.sqrt((min(x) - xref[xref<min(x)])**2*0.005**2 + np.mean(err)**2)
    errout[xref>max(x)] = np.sqrt((xref[xref>max(x)] - max(x))**2*0.01**2 + np.mean(err)**2)
    # Limit to an error of 1.2 mag (factor 3 in flux)
    errout[errout>1.2] = 1.2

    return yout,errout


def cosmocalc(z):
    ################# cosmocalc by N. Wright ##################

    '''
    This was used in an older version of superbol, but can still
    be used in place of astropy if desired - just uncomment cosmocalc in step 3
    '''
    # initialize constants
    H0 = 70                         # Hubble constant
    WM = 0.27                        # Omega(matter)
    WV = 1.0 - WM - 0.4165/(H0*H0)  # Omega(vacuum) or lambda

    WR = 0.        # Omega(radiation)
    WK = 0.        # Omega curvaturve = 1-Omega(total)
    c = 299792.458 # velocity of light in km/sec
    Tyr = 977.8    # coefficent for converting 1/H into Gyr
    DTT = 0.0      # time from z to now in units of 1/H0
    DCMR = 0.0     # comoving radial distance in units of c/H0
    DA = 0.0       # angular size distance
    DL = 0.0       # luminosity distance
    DL_Mpc = 0.0
    a = 1.0        # 1/(1+z), the scale factor of the Universe
    az = 0.5       # 1/(1+z(object))

    h = H0/100.
    WR = 4.165E-5/(h*h)   # includes 3 massless neutrino species, T0 = 2.72528
    WK = 1-WM-WR-WV
    az = 1.0/(1+1.0*z)
    n=1000         # number of points in integrals


    for i in range(n):
        a = az+(1-az)*(i+0.5)/n
        adot = np.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
        DTT = DTT + 1./adot
        DCMR = DCMR + 1./(a*adot)

    DTT = (1.-az)*DTT/n
    DCMR = (1.-az)*DCMR/n

    ratio = 1.00
    x = np.sqrt(abs(WK))*DCMR
    if x > 0.1:
        if WK > 0:
            ratio =  0.5*(np.exp(x)-np.exp(-x))/x
        else:
            ratio = np.sin(x)/x
    else:
        y = x*x
        if WK < 0: y = -y
        ratio = 1. + y/6. + y*y/120.

    DCMT = ratio*DCMR
    DA = az*DCMT

    DL = DA/(az*az)

    DL_Mpc = (c/H0)*DL

    return DL_Mpc


# Filter information

#SDSS filters and AB mags:
#These effective wavelengths for SDSS filters are from Fukugita et al. (1996, AJ, 111, 1748) and are
#the wavelength weighted averages (effective wavelengths in their Table 2a, first row)

#Effective wavelengths (in Angs)
wle = {'u': 3560,  'g': 4830, 'r': 6260, 'i': 7670, 'z': 8890, 'y': 9600, 'w':5985, 'Y': 9600,
       'U': 3600,  'B': 4380, 'V': 5450, 'R': 6410, 'G': 6730, 'E': 6730, 'I': 7980, 'J': 12200, 'H': 16300,
       'K': 21900, 'S': 2030, 'D': 2231, 'A': 2634, 'F': 1516, 'N': 2267, 'o': 6790, 'c': 5330,
       'W': 33526, 'Q': 46028
}
# For Swift UVOT: S=UVW2, D=UVM2, A=UVW1
# For GALEX: F=FUV, N=NUV
# For NEOWISE: W=W1, Q=W2
# For Gaia: E=G


# The below zeropoints are needed to convert magnitudes to fluxes
#For AB mags,
#     m(AB) = -2.5 log(f_nu) - 48.60.
# f_nu is in units of ergs/s/cm2/Hz such that
#    m(AB) = 0 has a flux of f_nu = 3.63E-20 erg/s/cm2/Hz  = 3631 Jy
# Therefore, AB magnitudes are directly related to a physical flux.
# Working through the conversion to ergs/s/cm2/Angs, gives
# f_lam = 0.1089/(lambda_eff^2)  where lambda_eff is the effective wavelength of the filter in angstroms
# Note then that the AB flux zeropoint is defined ONLY by the choice of effective wavelength of the bandpass

# However, not all bands here are AB mag, so for consistency across all filters the zeropoints are stored in the following dictionary

# Matt originally listed the following from  Paul Martini's page : http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
# That is not an original source, for AB mags it simply uses the f_lam =0.1089/(lambda_eff^2) relation, and the effective wavelengths from Fukugita et al.

# ugriz and GALEX NUV/FUV are in AB mag system, UBVRI are Johnson-Cousins in Vega mag, JHK are Glass system Vega mags, Swift UVOT SDA and WISE WQ are in Vega mag system
#
#The values for UBVRIJHK are for the Johnson-Cousins-Glass system and are taken directly from Bessell et al. 1998, A&A, 333, 231 (Paul Martini's page lists these verbatim)
#Note that these Bessell et al. (1998) values were calculated not from the spectrum of Vega itself, but from a Kurucz model atmosphere of an AOV star.
#GALEX effective wavelengths from here: http://galex.stsci.edu/gr6/?page=faq
# WISE data from http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=WISE&asttype=

# ATLAS values taken from Tonry et al 2018

##All values in 1e-11 erg/s/cm2/Angs
#zp = {'u': 859.5, 'g': 466.9, 'r': 278.0, 'i': 185.2, 'z': 137.8, 'y': 118.2, 'w': 245.7, 'Y': 118.2,
#      'U': 417.5, 'B': 632.0, 'V': 363.1, 'R': 217.7, 'I': 112.6, 'G': 240.0, 'E': 240.0,
#      'J': 31.47, 'H': 11.38, 'K': 3.961, 'S': 536.2, 'D': 463.7, 'A': 412.3, 'F': 4801., 'N': 2119.,
#      'o': 236.2, 'c': 383.3, 'W': 0.818, 'Q': 0.242}
      
# New Sept 2022: All ZPs from SVO filter service
zp_AB = {'u': 867.6, 'g': 487.6, 'r': 282.9, 'i': 184.9, 'z': 98.6, 'y': 117.8, 'w': 303.8, 'Y': 117.8,
      'U': 847.1, 'B': 569.7, 'V': 362.8, 'R': 257.8, 'I': 169.2, 'G': 278.5, 'E': 278.5,
      'J': 72.2, 'H': 40.5, 'K': 23.5, 'S': 2502.2, 'D': 2158.3, 'A': 1510.9, 'F': 4536.6, 'N': 2049.9,
      'o': 238.9, 'c': 389.3, 'W': 9.9, 'Q': 5.2}
      
zp_Vega = {'u': 351.1, 'g': 526.6, 'r': 242.6, 'i': 127.4, 'z': 49.5, 'y': 71.5, 'w': 245.7, 'Y': 71.5,
      'U': 396.5, 'B': 613.3, 'V': 362.7, 'R': 217.0, 'I': 112.6, 'G': 249.8, 'E': 249.8,
      'J': 31.3, 'H': 11.3, 'K': 4.3, 'S': 523.7, 'D': 457.9, 'A': 408.4, 'F': 650.6, 'N': 445.0,
      'o': 193.1, 'c': 400.3, 'W': 0.818, 'Q': 0.242}

#Default system
default_sys = {'u': 'AB',  'g': 'AB', 'r': 'AB', 'i': 'AB',  'z': 'AB', 'y': 'AB', 'w': 'AB', 'Y': 'Vega',
         'U': 'Vega',  'B': 'Vega', 'V': 'Vega', 'R': 'Vega', 'G': 'AB', 'E': 'AB', 'I': 'Vega',
         'J': 'Vega', 'H': 'Vega', 'K': 'Vega', 'S': 'Vega', 'D': 'Vega', 'A': 'Vega',  'F': 'AB',
         'N': 'AB', 'o': 'AB', 'c': 'AB', 'W': 'Vega', 'Q': 'Vega'}


#Filter widths (in Angs)
width = {'u': 458,  'g': 928, 'r': 812, 'i': 894,  'z': 1183, 'y': 628, 'w': 2560, 'Y': 628,
         'U': 485,  'B': 831, 'V': 827, 'R': 1389, 'I': 899, 'G': 4203, 'E': 4203, 'J': 1759, 'H': 2041,
         'K': 2800, 'S': 671, 'D': 446, 'A': 821,  'F': 268,  'N': 732, 'o': 2580, 'c': 2280,
         'W': 6626, 'Q': 10422}

#Extinction coefficients in A_lam / E(B-V). Uses York Extinction Solver (http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/community/YorkExtinctionSolver/coefficients.cgi)
extco = {'u': 4.786,  'g': 3.587, 'r': 2.471, 'i': 1.798,  'z': 1.403, 'y': 1.228, 'w':2.762, 'Y': 1.228,
         'U': 4.744,  'B': 4.016, 'V': 3.011, 'R': 2.386, 'I': 1.684, 'G': 2.216, 'E': 2.216,
         'J': 0.813, 'H': 0.516, 'K': 0.337, 'S': 8.795, 'D': 9.270, 'A': 6.432,  'F': 8.054,  'N': 8.969,
         'o': 2.185, 'c': 3.111, 'W': 0.190, 'Q': 0.127}

# Colours for plots
cols = {'u': 'dodgerblue', 'g': 'g', 'r': 'r', 'i': 'goldenrod', 'z': 'k', 'y': '0.5', 'w': 'firebrick',
        'Y': '0.5', 'U': 'slateblue', 'B': 'b', 'V': 'yellowgreen', 'R': 'crimson', 'I': 'chocolate',
        'G': 'salmon', 'E': 'salmon', 'J': 'darkred', 'H': 'orangered', 'K': 'saddlebrown',
        'S': 'mediumorchid', 'D': 'purple', 'A': 'midnightblue',
        'F': 'hotpink', 'N': 'magenta', 'o': 'darkorange', 'c': 'cyan',
        'W': 'forestgreen', 'Q': 'peru'}

# Maintains order from blue to red effective wavelength
bandlist = 'FSDNAuUBgcVwrRoGEiIzyYJHKWQ'



# First step is to search directory for existing superbol files, or photometry files matching our naming conventions
print('\n######### Step 1: input files and filters ##########')

# keep tabs on whether interpolated LCs exist
useInt = 'n'

# SN name defines names of input and output files
sn = input('\n> Enter SN name:   ')

if not sn:
    print('\n* No name given; lets just call it `SN`...')
    sn = 'SN'

# Keep outputs in this directory
outdir = 'superbol_output_'+sn
if not os.path.exists(outdir): os.makedirs(outdir)



# Get photometry files
do1 = input('\n> Find input files automatically?[y]   ')
if not do1: do1='y'
# User will almost always want to do this automatically, if files follow naming convention!

use1 = []

if do1 == 'y':
    # first check for previous superbol interpolations
    files = glob.glob(outdir+'/interpolated-lcs_'+sn+'*.txt')
    if len(files)>0:
        print('\n* Interpolated LC(s) already available:')

        # If multiple interpolations exist, ask user which they want
        for i in range(len(files)):
            print('  ', i, ':', files[i])

        use = input('\n> Use interpolated LC? (e.g. 0,2 for files 0 and 2, or n for no) [0]\n (Warning: using multiple interpolation files can cause problems unless times match!)   ')
        # Default is to read in the first interpolation file
        # Multiple interpolations can be read using commas, BUT if time axes don't match then the phases can end up incorrectly defined for some bands!!!
        if not use: use1.append(0)

        if use!='n':
            # if previous interpolations are used, need to keep tabs so we don't interpolate again later!
            useInt = 'y'
            if len(use)>0:
                for i in use.split(','):
                    use1.append(i)
        else: print('\n* Not using interpolated data')


    if len(files)==0 or use=='n':
        # And here is if we don't have (or want) previously interpolated data
        # search for any files matching with SN name
        files = glob.glob(sn+'_*')

        if len(files)>0:
            # If files are found, print them and let the user choose which ones to read in
            print('\n* Available files:')

            for i in range(len(files)):
                print('  ', i, ':', files[i])

            use = input('\n> Specify files to use (e.g. 0,2 for files 0 and 2) [all]   ')
            if len(use)>0:
                # Include only specified files
                for i in use.split(','):
                    use1.append(i)
            else:
                # Or include all files
                for i in range(len(files)):
                    use1.append(i)

        else:
            # If no files found, keep track and print message
            do1 = 'n'
            print('* No files found for '+sn)


if do1 != 'y':
    # If we did not find any input data, you can specify files manually - BUT should still follow filter conventions and end in _<filters>.EXT
    files1 = input('\n> Enter all file names separated by commas:\n')
    if not files1:
        # But if no files specified by this point, we give up prompting!
        print('No files given - exiting!')
        sys.exit(0)

    files = []
    for i in files1.split(','):
        # If manually specified files given, add them to input list
        files.append(i)
    for i in range(len(files)):
        # Also need to keep an integer index for each file, so we can treat them the same as we would the automatically-detected files
        use1.append(i)


# This dictionary is vital, will hold all light curve data!
lc = {}

# This keeps track of filters used (don't remember why I used strings in place of lists...)
filts2 = str()

for i in use1:
    # These integers map to the list of input files
    i = int(i)
    # get filter from file name and add to list
    # filts1 keeps track of filters IN THAT FILE ONLY, filts2 is ALL filters across ALL files.
    filts1 = files[i].split('.')[0]
    filts1 = filts1.split('_')[-1]
    filts2 += filts1

    # Here we read in the files using genfromtxt. Uses try statements to catch a few common variants of the input, e.g. with csv or header rows
    try:
        d = np.genfromtxt(files[i])
        x = 1
        for j in filts1:
            # loop over filters (j) in file and add each light curve to dictionary
            # column 0 is time, odd columns (x) are magnitudes, even columns (x+2) are errors
            lc[j] = np.array(list(zip(d[:,0][~np.isnan(d[:,x])],d[:,x][~np.isnan(d[:,x])],d[:,x+1][~np.isnan(d[:,x])])))
            x+=2
    except:
        try:
            d = np.genfromtxt(files[i],skip_header=1)
            x = 1
            for j in filts1:
                lc[j] = np.array(list(zip(d[:,0][~np.isnan(d[:,x])],d[:,x][~np.isnan(d[:,x])],d[:,x+1][~np.isnan(d[:,x])])))
                x+=2
        except:
            try:
                d= np.genfromtxt(files[i],delimiter=',')
                x = 1
                for j in filts1:
                    lc[j] = np.array(list(zip(d[:,0][~np.isnan(d[:,x])],d[:,x][~np.isnan(d[:,x])],d[:,x+1][~np.isnan(d[:,x])])))
                    x+=2
            except:
                try:
                    d= np.genfromtxt(files[i],delimiter=',',skip_header=1)
                    x = 1
                    for j in filts1:
                        lc[j] = np.array(list(zip(d[:,0][~np.isnan(d[:,x])],d[:,x][~np.isnan(d[:,x])],d[:,x+1][~np.isnan(d[:,x])])))
                        x+=2
                except:
                    raise ValueError('Could not read file')


# sort list of recognised filters from filts2 into wavelength order:
filters = str()
for i in bandlist:
    if i in filts2:
        filters += i

# If a filter name is not recognised, prompt user to add its properties manually
for i in filts2:
    if not i in wle:
        print('\n* Unknown filter '+i+'!')
        print('* Please enter details for filter',i)
        wle[i] = float(input(' >Lambda_eff (angstroms):   '))
        zp[i] = float(input(' >Flux zero point (1e11 erg/cm2/s/ang):   '))
        width[i] = float(input(' >Filter width (angstroms):   '))
        ftmp = str()
        cols[i] = 'grey'
        for j in filters:
            if wle[j]<wle[i]:
                ftmp += j
        ftmp += i
        for j in filters:
            if wle[j]>wle[i]:
                ftmp += j
        filters = ftmp
        
        

plt.figure(1,(8,6))
plt.clf()

# Default time axis label
xlab = 'Time'

# Plot all light curves on same axes
for i in filters:
    plt.errorbar(lc[i][:,0],lc[i][:,1],lc[i][:,2],fmt='o',color=cols[i],label=i)

plt.gca().invert_yaxis()
plt.xlabel(xlab)
plt.ylabel('Magnitude')
plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
plt.tight_layout(pad=0.5)
plt.draw()


limitMJDs = input('\n> Limit time range to use? [n] ')
if not limitMJDs: limitMJDs = 'n'

if limitMJDs == 'y':
    
    MJDmin = input('Min time to use [0] ')
    if not MJDmin: MJDmin = 0
    MJDmin = float(MJDmin)
    
    MJDmax = input('Max time to use [1000000] ')
    if not MJDmax: MJDmax = 1000000
    MJDmax = float(MJDmax)
    
    for i in lc:
        lc[i] = lc[i][lc[i][:,0]>MJDmin]
        lc[i] = lc[i][lc[i][:,0]<MJDmax]

plt.clf()

# Plot all light curves on same axes
for i in filters:
    plt.errorbar(lc[i][:,0],lc[i][:,1],lc[i][:,2],fmt='o',color=cols[i],label=i)

plt.gca().invert_yaxis()
plt.xlabel(xlab)
plt.ylabel('Magnitude')
plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
plt.tight_layout(pad=0.5)
plt.draw()


# This ends the data import


print('\n######### Step 2: reference band for phase info ##########')



print('\n* Displaying all available photometry...')

# User can choose to include only a subset of filters, e.g. if they see that some don't have very useful data
t3 = input('\n> Enter bands to use (blue to red) ['+filters+']   ')
if not t3: t3 = filters

filters = ''

for i in t3:
    if i != ',' and i != ' ':
        filters += i

if len(filters) < 2:
    # If only one filter, no need to interpolate, and can't apply BB fits, so makes no sense to use superbol!
    print('At least two filters required - exiting!')
    sys.exit(0)


# Loop through dictionary and determine which filter has the most data
ref1 = 0
for i in filters:
    ref2 = len(lc[i])
    if ref2>ref1:
        ref1 = ref2
        ref3 = i


# If using light curves that have not yet been interpolated by a previous superbol run, we need a reference filter
if useInt!='y':
    ref = input('\n> Choose reference band(s) for sampling epochs (Comma delimited)\n   Suggested (most LC points): ['+ref3+']   ')
    # Defaults to the band with the most data
    if not ref: ref = ref3
    ref_list = ref.split(',')

# If light curves are already interpolated, reference is mainly for plotting so just pick first band
else: ref_list = filters[0]

for i in ref_list:
    print('\n* Using '+i+'-band for reference')
    
n_refs = len(ref_list)

ref_stack = np.zeros_like(lc[ref_list[0]][0])

for i in ref_list:
    ref_stack = np.vstack((ref_stack,lc[i]))
    
ref_stack = ref_stack[1:]

ref_stack = ref_stack[ref_stack[:,0].argsort()]

# User may want to have output in terms of days from maximum, so here we find max light in reference band
# Two options: fit light curve interactively, or just use brightest point. User specifies what they want to do
t1 = input('\n> Interactively find maximum?[n] ')
if not t1:
    # Default to not doing interactive fit
    t1 = 'n'

    # in this case check if user wants quick approximation
    doSh = input('\n> Shift to approx maximum?[n] ')
    # Default to not doing this either - i.e. leave light curve as it is
    if not doSh: doSh = 'n'

    if doSh=='y':
        # If approx shift wanted, find time of brightest point in ref band to set as t=0
        d = np.copy(ref_stack)
        shift = d[:,0][np.argmin(d[:,1])]
        # Loop over all bands and shift them
        for j in lc:
            lc[j][:,0]-=shift
            
        ref_stack[:,0]-=shift

        # update x-axis label
        xlab += ' from approx maximum'

        print('\n* Approx shift done')


if t1!='n':
    # Here's where date of maximum is fit interactively, if user wanted it
    # Start with approx shift of reference band
    d = np.copy(ref_stack)
    shift = d[:,0][np.argmin(d[:,1])]
    d[:,0]-=shift

    plt.clf()
    # Plot reference band centred roughly on brightest point
    plt.errorbar(d[:,0],d[:,1],d[:,2],fmt='o',color='0.5')

    plt.ylim(max(d[:,1])+0.2,min(d[:,1])-0.2)
    plt.xlabel(xlab + ' from approx maximum')
    plt.ylabel('Magnitude')
    plt.tight_layout(pad=0.5)
    plt.draw()

    # As long as happy ='n', user can keep refitting til they get a good result
    happy = 'n'

    print('\n### Begin polynomial fit to peak... ###')

    # Default polynomial order =4
    order1 = 4

    # Only fit data at times < Xup from maximum light. Default is 50 days
    Xup1 = 50

    while happy == 'n':

        print('\n### Select data range ###')

        # Interactively set upper limit on times to fit
        Xup = input('>> Cut-off phase for polynomial fit?['+str(Xup1)+']   ')
        if not Xup: Xup = Xup1
        Xup = float(Xup)
        Xup1 = Xup

        d1 = d[d[:,0]<Xup]

        plt.clf()

        # Plot only times < Xup
        plt.errorbar(d1[:,0],d1[:,1],d1[:,2],fmt='o',color='0.5')

        plt.ylim(max(d1[:,1])+0.4,min(d1[:,1])-0.2)
        plt.tight_layout(pad=0.5)
        plt.draw()

        # Interactively set polynomial order
        order = input('\n>> Order of polynomial to fit?['+str(order1)+']   ')
        if not order: order = order1
        order = int(order)
        order1 = order

        # Fit light curve with polynomial
        fit = np.polyfit(d1[:,0],d1[:,1],deg=order)

        # Plot the polynomial
        days = np.arange(min(-40,min(d[:,0]))-10,Xup)
        eq = 0
        for i in range(len(fit)):
            # Loop allows calculation for arbitrary polynomial order
            eq += fit[i]*days**(order-i)
        plt.plot(days,eq,label='Fit order = %d' %order)

        plt.ylabel('Magnitude')
        plt.xlabel(xlab + ' from approx maximum')
        plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
        plt.xlim(min(d[:,0])-5,Xup)
        plt.tight_layout(pad=0.5)
        plt.draw()

        # Check if user likes fit
        happy = input('\n> Happy with fit?(y/[n])   ')
        # Default is to try again!
        if not happy: happy = 'n'

    # After user tired/satisfied with fit, check if they want to use the peak of their most recent polynomial as t=0, or default to the brightest point
    new_peak = input('> Use [p-olynomial] or o-bserved peak date?    ')
    # Default is to use polynomial for peak date
    if not new_peak: new_peak = 'p'

    xlab += ' from maximum'

    # Plot reference band shifted to match polynomial peak
    if new_peak=='p':
        days = np.arange(d[:,0][np.argmin(d[:,1])]-10,d[:,0][np.argmin(d[:,1])]+10)
        eq = 0
        for i in range(len(fit)):
            # Loop allows calculation for arbitrary polynomial order
            eq += fit[i]*days**(order-i)
        peak = days[np.argmin(eq)]
        d[:,0] -= peak
        plt.clf()
        plt.errorbar(d[:,0],d[:,1],d[:,2],fmt='o',color='0.5')
        plt.ylabel('Magnitude')
        plt.xlabel(xlab)
        plt.ylim(max(d[:,1])+0.2,min(d[:,1])-0.2)
        plt.tight_layout(pad=0.5)
        plt.draw()

    # If user instead wants observed peak, that shift was already done!
    if new_peak == 'o':
        peak = 0

    # Shift all light curves by same amount as reference band
    for j in lc:
        lc[j][:,0]-=(shift+peak)
        
    ref_stack[:,0]-=(shift+peak)


plt.figure(1)
plt.clf()

# Re-plot the light curves after shifting
for i in filters:
    plt.errorbar(lc[i][:,0],lc[i][:,1],lc[i][:,2],fmt='o',color=cols[i],label=i)

plt.gca().invert_yaxis()
plt.xlabel(xlab)
plt.ylabel('Magnitude')
plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
plt.tight_layout(pad=0.5)
plt.draw()

# Needed for K-correction step a bit later
skipK = 'n'

# Input redshift or distance modulus, needed for flux -> luminosity
z = input('\n> Please enter SN redshift or distance modulus:[0]  ')
# Default to zero
if not z: z=0
z = float(z)

if z<10:
    # Redshift always less than 10, distance modulus always greater, so easy to distinguish
    print('Redshift entered (or DM=0)')

    t2 = ''

    # Check if user wants to correct time axis for cosmological time dilation
    if ref_stack[0,0]>25000 or useInt=='y':
        # If time is in MJD or input light curves were already interpolated, default to no
        t2 = input('\n> Correct for time-dilation?[n] ')
        if not t2: t2 = 'n'
    else:
        # Otherwise default to yes
        t2 = input('\n> Correct for time-dilation?[y] ')
        if not t2: t2 = 'y'

    if t2=='y':
        # Apply correction for time dilation
        for j in lc:
            lc[j][:,0]/=(1+z)
        print('\n* Displaying corrected phases')
        
        ref_stack[:,0]/=(1+z)

        xlab += ' (rest-frame)'
        plt.xlabel(xlab)


    plt.figure(1)
    plt.clf()

    # Re-plot light curves in rest-frame times
    for i in filters:
        plt.errorbar(lc[i][:,0],lc[i][:,1],lc[i][:,2],fmt='o',color=cols[i],label=i)

    plt.gca().invert_yaxis()
    plt.xlabel(xlab)
    plt.ylabel('Magnitude')
    plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
    plt.tight_layout(pad=0.5)
    plt.draw()


    print('\n######### Step 3: Flux scale ##########')

    # New version uses astropy coordinates.Distance
    # Old version used cosmocalc (thanks to Sebastian Gomez for change)
    # Options for cosmologies
    # WMAP9, H0 = 69.3, Om0 = 0.286, Tcmb0 = 2.725, Neff = 3.04, m_nu = 0, Ob0 = 0.0463
    # And many others...
    # from astropy.cosmology import WMAP9
    # cosmology.set_current(WMAP9)
    DL_Mpc = Distance(z = z).Mpc

    # To use cosmocalc instead, uncomment below:
    # DL_Mpc = cosmocalc(z)

    #############################################

    # Check value of first light curve point to see if likely absolute or apparent mag
    print('\n* First ref band mag = %.2f' %ref_stack[0,1])
    absol='n'
    if ref_stack[0,1] < 0:
        # If negative mag, must be absolute (but check!)
        absol = input('> Are magnitudes *absolute* mags? [y] ')
        if not absol: absol='y'
    else:
        # If positive mag, must be apparent (but check!)
        absol = input('> Are magnitudes *absolute* mags? [n] ')
        if not absol: absol ='n'

    if absol=='y':
        # If absolute mag, set distance to 10 parsecs
        DL_Mpc = 1e-5
        print('\n* Absolute mags; Luminosity distance = 10 pc')
    else:
        # Otherwise use luminosity distance from redshift
        print('\n* Luminosity distance = %.2e Mpc' %DL_Mpc)

    # convert Mpc to cm, since flux in erg/s/cm2/A
    dist = DL_Mpc*3.086e24

else:
    # If distance modulus entered, different approach needed!
    print('Distance modulus entered')

    # No k correction if no redshift!
    skipK = 'y'

    for i in lc:
        # Subtract distance modulus to convert to absolute mags (assuming no one would ever supply absolute mags and still enter a DM...)
        lc[i][:,1]-=z
        # Once absolute, distance = 10 pc
        dist = 1e-5*3.086e24




print('\n######### Step 4: Interpolate LCs to ref epochs ##########')

# If light curves are not already interpolated, now we need to do some work
if useInt!='y':
    # Sort light curves by phase (sometimes this isn't done already...)
    for i in lc:
        lc[i] = lc[i][lc[i][:,0].argsort()]

    # New dictionary for interpolated light curves
    lc_int = {}
    
    if n_refs == 1:
        lc_int[ref_list[0]] = lc[ref_list[0]]

    # User decides whether to fit each light curve
    t4 = input('\n> Interpolate light curves interactively?[y] ')
    # Default is yes
    if not t4: t4 = 'y'

    if t4=='y':
        print('\n### Begin polynomial fit... ###')

        # Interpolate / extrapolate other bands to same epochs - polynomial fits
        # - what if there are only one or two points??? Use colour?

        # Use this to keep tabs on method used, and append to output file
        intKey = '\n# Reference bands: '
        for refband in ref_list:
            intKey += refband

        for i in filters:
            # Need to loop through and interpolate every band except reference
            if n_refs == 1 and i in ref_list:
                pass
            else:
                print('\n### '+i+'-band ###')

                # Default polynomial order to fit light curves
                order1 = 4

                # Keep looping until happy
                happy = 'n'
                while happy == 'n':
                    # Plot current band and reference band
                    plt.clf()
                    plt.errorbar(lc[i][:,0],lc[i][:,1],lc[i][:,2],fmt='o',color=cols[i],label=i)
                    plt.errorbar(ref_stack[:,0],ref_stack[:,1],ref_stack[:,2],fmt='o',color='0.5',label='ref')
                    plt.gca().invert_yaxis()
                    plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
                    plt.xlabel(xlab)
                    plt.ylabel('Magnitude')
                    plt.ylim(max(max(ref_stack[:,1]),max(lc[i][:,1]))+0.5,min(min(ref_stack[:,1]),min(lc[i][:,1]))-0.5)
                    plt.tight_layout(pad=0.5)
                    plt.draw()

                    # Choose order of polynomial fit to use
                    order = input('\n>> Order of polynomial to fit?(q to quit and use constant colour)['+str(order1)+']   ')
                    # If user decides they can't get a good fit, enter q to use simple linear interpolation and constant-colour extrapolation
                    if order == 'q':
                        break
                    # Or use default order
                    if not order: order = order1

                    order = int(order)
                    # Set new default to current order
                    order1 = order

                    # Fit light curve with polynomial
                    fit = np.polyfit(lc[i][:,0],lc[i][:,1],deg=order)

                    # Plot fit
                    days = np.arange(np.min([np.min(lc[i][:,0]),np.min(ref_stack[:,0])]), np.max([np.max(lc[i][:,0]),np.max(ref_stack[:,0])]))
                    eq = 0
                    for j in range(len(fit)):
                        # Loop for arbitrary polynomial order
                        eq += fit[j]*days**(order-j)
                    plt.plot(days,eq,label='Fit order = %d' %order)
                    plt.ylabel('Magnitude')
                    plt.xlabel(xlab)
                    plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
                    plt.tight_layout(pad=0.5)
                    plt.draw()

                    # Check if happy with fit
                    happy = input('\n> Happy with fit?(y/[n])   ')
                    # Default to no
                    if not happy: happy = 'n'

                # If user quit polyfit, use easyint
                if order == 'q':
                    # This breaks if no overlap in time with ref band
                    tmp1,tmp2 = easyint(lc[i][:,0],lc[i][:,1],lc[i][:,2],ref_stack[:,0],ref_stack[:,1])
                    tmp = list(zip(ref_stack[:,0],tmp1,tmp2))
                    lc_int[i] = np.array(tmp)
                    print('\n* Interpolating linearly; extrapolating assuming constant colour...')
                    # Add method to output
                    intKey += '\n# '+i+': Linear interp; extrap=c'
                else:
                    # If user was happy with fit, add different interpolation string to output
                    intKey += '\n# '+i+': fit order='+str(order)+'; extrap method '

                    # Construct polynomial interpolation
                    # Goal: if band has point at same epoch as ref band, use point; otherwise, use polynomial prediction

                    mag_int = []

                    for k in ref_stack:
                        # Check each light curve point against each reference time
                        # If match, add that point to interpolated light curve
                        k1 = np.where(lc[i][:,0]==k[0])
                        if len(k1[0])>0:
                            mag_int.append(lc[i][k1][0])

                    # Convert matches to numpy array (just to compare with reference array)
                    tmp_arr = np.array(mag_int)

                    if tmp_arr.size:
                        # Do this loop if there were some temporal matches between current and reference band
                        for k in ref_stack:
                            # Iterate over each reference time
                            if k[0] not in tmp_arr[:,0]:
                                # If no match in current band, calculate magnitude from polynomial
                                mag = 0
                                for j in range(len(fit)):
                                    # Sum works for arbitrary polynomial order
                                    mag += fit[j]*k[0]**(order-j)
                                # Append polynomial magnitude to light curve, with an error floor of 0.1 mags
                                out = np.array([k[0],mag,max(np.mean(lc[i][:,2]),0.1)])
                                mag_int.append(out)
                    else:
                        # Do this loop if there were zero matches between current band and reference times
                        for l in ref_stack[:,0]:
                            # Construct polynomial mag as above for each reference time
                            mag = 0
                            for j in range(len(fit)):
                                mag += fit[j]*l**(order-j)
                            out = np.array([l,mag,max(np.mean(lc[i][:,2]),0.1)])
                            mag_int.append(out)

                    # Convert full interpolated light curve to np array
                    mag_int = np.array(mag_int)

                    # Sort chronologically
                    tmp = mag_int[np.argsort(mag_int[:,0])]

                    # Now need to check extrapolation to times outside observed range for current band
                    # Polynomial method already did an extrapolation, but polynomial can be bad where there is no data to constrain it!
                    # Here we apply the constant colour method too, and user can check what they prefer

                    # Earliest time in band (extrapolating if necessary to get some overlap with ref band)
                    low = min(lc[i][:,0])
                    low = min(low,max(tmp[:,0]))
                    # Latest time in band
                    up = max(lc[i][:,0])
                    up = max(up,min(tmp[:,0]))
                    # Colour wrt reference band at earliest and latest interpolated epochs
                    col1 = tmp[tmp[:,0]>=low][0,1] - ref_stack[tmp[:,0]>=low][0,1]
                    col2 = tmp[tmp[:,0]<=up][-1,1] - ref_stack[tmp[:,0]<=up][-1,1]
                     # Get extrapolated points in current band by adding colour to reference band
                    early = ref_stack[tmp[:,0]<low][:,1]+col1
                    late = ref_stack[tmp[:,0]>up][:,1]+col2
                    # Compute error as random sum of average error in band plus 0.1 (0.05) mag for every 10 days extrapolated late (early)
                    tmp[:,2][tmp[:,0]<low] = np.sqrt((low - tmp[:,0][tmp[:,0]<low])**2*0.005**2 + np.mean(lc[i][:,2])**2)
                    tmp[:,2][tmp[:,0]>up] = np.sqrt((tmp[:,0][tmp[:,0]>up] - up)**2*0.01**2 + np.mean(lc[i][:,2])**2)
                    # Limit to an error of 1.2 mag (factor 3 in flux)
                    tmp[:,2][tmp[:,2]>1.2] = 1.2

                    # Plot light curve from polynomial fit
                    plt.errorbar(tmp[:,0],tmp[:,1],fmt='s',markersize=12,mfc='none',markeredgewidth=3,markeredgecolor=cols[i],label='Polynomial')
                    # Plot constant colour extrapolation
                    plt.errorbar(tmp[tmp[:,0]<low][:,0],early,fmt='o',markersize=12,mfc='none',markeredgewidth=3,markeredgecolor=cols[i],label='Constant colour')
                    plt.errorbar(tmp[tmp[:,0]>up][:,0],late,fmt='o',markersize=12,mfc='none',markeredgewidth=3,markeredgecolor=cols[i])
                    plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
                    plt.tight_layout(pad=0.5)
                    plt.draw()

                    if len(tmp[tmp[:,0]<low])>0:
                        # If there are early extrapolated points, ask user whether they prefer polynomial, constant colour, or want to hedge their bets
                        extraptype = input('\n> Early-time extrapolation:\n  [P-olynomial], c-onstant colour, or a-verage of two methods?\n')
                        # Default to polynomial
                        if not extraptype: extraptype = 'p'
                        if extraptype == 'c':
                            # constant colour
                            tmp[:,1][tmp[:,0]<low]=early
                        if extraptype == 'a':
                            # average
                            tmp[:,1][tmp[:,0]<low]=0.5*(tmp[:,1][tmp[:,0]<low]+early)
                    # If no need to extrapolate:
                    else: extraptype = 'n'

                    # Keep tabs on which extrapolation method was used!
                    intKey += 'early='+extraptype+';'

                    # Now do same for late times
                    if len(tmp[tmp[:,0]>up])>0:
                        extraptype = input('\n> Late-time extrapolation:\n  [P-olynomial], c-onstant colour, or a-verage of two methods?\n')
                        if not extraptype: extraptype = 'p'
                        if extraptype == 'c':
                            tmp[:,1][tmp[:,0]>up]=late
                        if extraptype == 'a':
                            tmp[:,1][tmp[:,0]>up]=0.5*(tmp[:,1][tmp[:,0]>up]+late)
                    else: extraptype = 'n'

                    intKey += 'late='+extraptype

                    # Add the final interpolated and extrapolated light curve to the dictionary
                    lc_int[i] = tmp

        # Key for output file
        intKey += '\n# p = polynomial, c = constant colour, a = average'

    # If user does not want to do interpolation interactively:
    else:
        for i in filters:
            # For every band, use easyint for linear interpolation between points, and constant colour extrapolation
            tmp1,tmp2 = easyint(lc[i][:,0],lc[i][:,1],lc[i][:,2],ref_stack[:,0],ref_stack[:,1])
            tmp = list(zip(ref_stack[:,0],tmp1,tmp2))
            lc_int[i] = np.array(tmp)
        print('\n* Interpolating linearly; extrapolating assuming constant colour...')

        intKey = '\n# All light curves linearly interpolated\n# Extrapolation done by assuming constant colour with reference band'

    # Need to save interpolated light curves for future re-runs
    int_out = np.empty([len(ref_stack[:,0]),1+2*len(filters)])
    # Start with reference times
    int_out[:,0] = ref_stack[:,0]

    for i in range(len(filters)):
        # Append magnitudes and errors, in order from bluest to reddest bands
        int_out[:,2*i+1] = lc_int[filters[i]][:,1]
        int_out[:,2*i+2] = lc_int[filters[i]][:,2]

    # Open file in superbol output directory to write light curves
    int_file = open(outdir+'/interpolated-lcs_'+sn+'_'+filters+'.txt','wb')

    # Construct header
    cap = '#phase\t'
    for i in filters:
        # Add a column heading for each filter
        cap = cap+i+'\terr\t'
    cap +='\n'

    # Save to file, including header and footer containing log of interpolation methods
    np.savetxt(int_file,int_out,fmt='%.2f',delimiter='\t',header=cap,footer=intKey,comments='#')
    # Close output file
    int_file.close()

    # Plot interpolated lcs
    print('\n* Displaying all interpolated/extrapolated LCs')
    plt.figure(1)
    plt.clf()
    for i in filters:
        plt.errorbar(lc_int[i][:,0],lc_int[i][:,1],lc_int[i][:,2],fmt='o',color=cols[i],label=i)
    plt.gca().invert_yaxis()
    plt.xlabel(xlab)
    plt.ylabel('Magnitude')
    plt.legend(numpoints=1,fontsize=16,ncol=2,frameon=True)
    plt.tight_layout(pad=0.5)
    plt.draw()

# Or if light curves were already interpolated, no need for the last 250 lines!
else:
    print('\n* Interpolation already done, skipping step 4!')

    # Put pre-interpolated lcs into dictionary
    lc_int = {}
    for i in filters:
        lc_int[i] = lc[i]


print('\n######### Step 5: Extinction and K-corrections #########')

# Extinction correction
ebv = input('\n> Please enter Galactic E(B-V): \n'
                        '  (0 if data are already extinction-corrected) [0]   ')
if not ebv: ebv=0
ebv = float(ebv)

for i in lc_int:
    # Subtract foreground extinction using input E(B-V) and coefficients from YES
    lc_int[i][:,1]-=extco[i]*ebv

## If UVOT bands are in AB, need to convert to Vega
#if 'S' in lc_int or 'D' in lc_int or 'A' in lc_int:
#    shiftSwift = input('\n> UVOT bands detected. These must be in Vega mags.\n'
#                            '  Apply AB->Vega correction for these bands? [n]   ')
#    if not shiftSwift: shiftSwift = 'n'
#
#    if shiftSwift == 'y':
#        if 'S' in lc_int:
#            lc_int['S'][:,1] -= 1.51
#        if 'D' in lc_int:
#            lc_int['D'][:,1] -= 1.69
#        if 'A' in lc_int:
#            lc_int['A'][:,1] -= 1.73
#        print('\n* Converting UV bands to Vega')


print('\nDefault photometric systems:')
print(default_sys)

is_correct_system = input('\n> Are all bands in their default systems? ([y]/n)  ')
if not is_correct_system: is_correct_system = 'y'

systems = {}
if is_correct_system == 'n':
    for i in filters:
        # This loop should ask for each band if it is in A-B or V-ega, and add to a dictionary
        sys1 = input('Is '+i+'-band data in [A]-B or V-ega? ')
        if not sys1: sys1 = 'AB'
        systems[i] = sys1
else:
    for i in filters:
        systems[i] = default_sys[i]

# Whether to apply approximate K correction
doKcorr = 'n'
# i.e. if we have a redshift:
if skipK == 'n':
    # converting to rest-frame means wavelength /= 1+z and flux *= 1+z. But if input magnitudes were K-corrected, this has already been done implicitly!
    doKcorr = input('\n> Do you want to covert flux and wavelength to rest-frame?\n'
                            '  (skip this step if data are already K-corrected) [y]   ')
    if not doKcorr: doKcorr = 'y'

######### Now comes the main course - time to build SEDs and integrate luminosity

# Build list of wavelengths
wlref = []
# First wavelength is roughly blue edge of bluest band (effective wavelength + half the width)
wlref1 = [wle[filters[0]]-width[filters[0]]/2]
# wlref contains band centres only (for BB fit), whereas wlref1 also has outer band edges (for numerical integration)

# List of flux zeropoints matching wavelengths
fref = []

# List of widths for each band (needed for error estimates)
bandwidths = []

# Loop over used filters and populate lists from dictionaries of band properties
for i in filters:
    wlref.append(float(wle[i]))
    if systems[i] == 'Vega':
        fref.append(zp_Vega[i]*1e-11)
    else:
        fref.append(zp_AB[i]*1e-11)
    wlref1.append(float(wle[i]))
    bandwidths.append(float(width[i]))

# Final reference wavelength is red edge of reddest band
wlref1.append(wle[filters[-1]]+width[filters[-1]]/2)
# Flux will be set to zero at red and blue extrema of SED when integrating pseudobolometric light curve

# Make everything a numpy array
wlref1 = np.array(wlref1)
wlref = np.array(wlref)
fref = np.array(fref)
bandwidths = np.array(bandwidths)

# Get phases with photometry to loop over
phase = ref_stack[:,0]

# Correct flux and wavelength to rest-frame, if user chose that option earlier
if doKcorr == 'y':
    wlref /= (1+z)
    wlref1 /= (1+z)
    fref *= (1+z)
    bandwidths /= (1+z)
    print('\n* Converting to rest frame')

# construct some notes for output file
method = '\n# Methodology:'
method += '\n# filters used:'+filters
method += '\n# redshift used:'+str(z)
method += '\n# extinction used:'+str(ebv)

if doKcorr == 'y':
    method += '\n# Flux and wavelength converted to rest-frame'
else:
    method += '\n# Wavelengths used in observer frame (data already K-corrected?)'



print('\n######### Step 6: Fit blackbodies and integrate flux #########')

# these are needed to scale and offset SEDs when plotting, to help visibility
k = 1
fscale = 4*np.pi*dist**2*1e-9*10**(-0.4*min(ref_stack[:,1])) / 1e40

# These lists will be populated with luminosities as we loop through the data and integrate SEDs
L1arr = []
L2arr = []
L1err_arr = []
L2err_arr = []
Lbb_full_arr = []
Lbb_full_err_arr = []
Lbb_opt_arr = []
Lbb_opt_err_arr = []


# Set up some parameters for absorbed BB fits and integrations:
# Functional form comes from Nicholl, Guillochon & Berger 2017 / Yan et al 2018:
# - power law in (lambda / lambda_cutoff) joins smoothly to BB at lambda_cutoff
# These default parameters give an unattenuated blackbody
bluecut = 1
sup = 0

do_absorb = input('\n> Absorbed blackbody L_uv(lam) = L_bb(lam)*(lam/lam_max)^x\n can give better fit in UV. Apply absorption? [n]   ')
if not do_absorb: do_absorb = 'n'

if do_absorb in ('y','yes'):
    bluecut = input('\n> Absorb below which wavelength? [3000A]   ')
    if not bluecut: bluecut = 3000
    bluecut = float(bluecut)
    
    sup = input('\n> Suppression index for BB flux bluewards of '+str(bluecut)+'A? [1]   ')
    if not sup: sup = 1
    sup = float(sup)

# NOTE: at some point should give option to make these free parameters...


T_init = input('\n> Initial guess for starting temperature [10000 K]?   ')
if not T_init: T_init = 10000
T_init = float(T_init)

T_init /= 1000


R_init = input('\n> Initial guess for starting radius [1.0e15 cm]?   ')
if not R_init: R_init = 1.0e15
R_init = float(R_init)

R_init /= 1e15


def bbody_absorbed(x,T,R,lambda_cutoff=bluecut,alpha=sup):
    '''
    Calculate the blackbody radiance for a set
    of wavelengths given a temperature and radiance.
    Modified in the UV

    Parameters
    ---------------
    lam: Reference wavelengths in Angstroms
    T:   Temperature in 10^3 Kelvin
    R:   Radius in 10^15 cm

    Output
    ---------------
    Spectral radiance in units of erg/s/Angstrom

    (calculation and constants checked by Sebastian Gomez)
    '''

    T *= 1000
    R *= 1e15

    # Planck Constant in cm^2 * g / s
    h = 6.62607E-27
    # Speed of light in cm/s
    c = 2.99792458E10

    # Convert wavelength to cm
    lam_cm = x * 1E-8

    # Boltzmann Constant in cm^2 * g / s^2 / K
    k_B = 1.38064852E-16

    # Calculate Radiance B_lam, in units of (erg / s) / cm ^ 2 / cm
    exponential = (h * c) / (lam_cm * k_B * T)
    B_lam = ((2 * np.pi * h * c ** 2) / (lam_cm ** 5)) / (np.exp(exponential) - 1)
    B_lam[x <= lambda_cutoff] *= (x[x <= lambda_cutoff]/lambda_cutoff)**alpha

    # Multiply by the surface area
    A = 4*np.pi*R**2

    # Output radiance in units of (erg / s) / Angstrom
    Radiance = B_lam * A / 1E8

    return Radiance / 1e40



# Open output files for bolometric light curve and blackbody parameters
out1 = open(outdir+'/bol_'+sn+'_'+filters+'.txt','w')
out2 = open(outdir+'/BB-params_'+sn+'_'+filters+'.txt','w')

# Write header for bol file
out1.write('# ph\tLobs\terr\tL+BB\terr\t\n\n')

# Write header for BB params file - if separate UV/optical fits, need another set of columns for the optical-only filts
# T_bb etc are fits to all data, T_opt are fits to data at lambda>3000A (i.e. not affected by line blanketing)
out2.write('# ph\tT_bb\terr\tR_bb\terr\tL_bb\terr\n\n')

# Display various lines for different fitting assumptions, tell user here rather than cluttering figure legend
print('\n*** Fitting Blackbodies to SED ***')
print('\n* Solid line = blackbody fit with assumed suppression')

    # plot suppression if used, and tell output file where suppression began and what was the index
print('* Dotted lines = bbody without blanketing')

method += '\n# BB fit below '+str(bluecut)+'A suppressed by factor (lamda/'+str(bluecut)+')^'+str(sup)

# New figure to display SEDs
plt.figure(2,(8,8))
plt.clf()


## For lmfit - hasn't worked yet
#bbmod = Model(bbody_absorbed)
#fit_params = Parameters()
#fit_params.add('T', value=1, max=100, min=0.1)
#fit_params.add('R', value=1, max=10, min=0.01)
#fit_params.add('lambda_cutoff', value=bluecut, vary=False)
#fit_params.add('alpha', value=sup, vary=False)

# Loop through reference epochs
for i in range(len(phase)):
    # get date
    ph = phase[i]
    # Get list of mags and errors in all filters at each epoch - start with blank arrays to add all filters
    mags = np.zeros(len(filters))
    errs = np.zeros(len(filters))
    for j in range(len(filters)):
        # Loop through filters and populate SED tables with interpolated light curves
        mags[j] = lc_int[filters[j]][i,1]
        errs[j] = lc_int[filters[j]][i,2]
    # convert magnitudes to physical fluxes using zeropoints and distance
    flux = 4*np.pi*dist**2*fref*10**(-0.4*mags)
    # convert mag errors to flux errors
    ferr = 2.5/np.log(10) * flux * errs
    # Set flux to zero at red and blue extrema matching wlref1
    flux1 = np.insert(flux,0,0)
    flux1 = np.append(flux1,0)

    # Fit blackbody to SED (the one that is not padded with zeros)
    
    # Using scipy
    BBparams, covar = curve_fit(bbody_absorbed,wlref,flux/1e40,p0=(T_init,R_init),sigma=ferr/1e40,maxfev=5000,bounds = ((0.1, 0.01), (100,100)))
    # Get temperature and radius, with errors, from fit
    T1 = BBparams[0]
    T1_err = min(np.sqrt(np.diag(covar))[0],T1)
    R1 = np.abs(BBparams[1])
    R1_err = min(np.sqrt(np.diag(covar))[1],R1)
    
#    T_init = T1
#    R_init = R1

#    # Using lmfit
#    result = bbmod.fit(flux/1e40,x=wlref,params=fit_params,weights=1e40/ferr)
#    T1 = result.params['T'].value
#    T1_err = min(result.params['T'].stderr,T1)
#    R1 = result.params['R'].value
#    R1_err = min(result.params['R'].stderr,R1)


    # Plot SEDs, offset for clarity
    plt.figure(2)
    plt.errorbar(wlref,flux/1e40-fscale*k,ferr/1e40,fmt='o',color=cols[filters[k%len(filters)]],label='%.1f' %ph)
    plt.plot(np.arange(1,25000),bbody_absorbed(np.arange(1,25000),T1,R1,bluecut,sup)-fscale*k,color=cols[filters[k%len(filters)]],linestyle='-')
    plt.plot(np.arange(1,25000),bbody_absorbed(np.arange(1,25000),T1,R1,1,0)-fscale*k,color=cols[filters[k%len(filters)]],linestyle=':')

    # Get pseudobolometric luminosity by trapezoidal integration, with flux set to zero outside of observed bands
    L1 = itg.trapezoid(flux1[np.argsort(wlref1)],wlref1[np.argsort(wlref1)])
    # Use flux errors and bandwidths to get luminosity error
    L1_err = np.sqrt(np.sum((bandwidths*ferr)**2))
    # Add luminosity to array (i.e. pseudobolometric light curve)
    L1arr.append(L1)
    L1err_arr.append(L1_err)

    # Calculate luminosity using alternative method of Stefan-Boltzmann, and T and R from fit
    L1bb = 4*np.pi*(R1*1e15)**2*5.67e-5*(T1*1e3)**4
    L1bb_err = L1bb*np.sqrt((2*R1_err/R1)**2+(4*T1_err/T1)**2)

    # Get UV luminosity (i.e. bluewards of bluest band)
    Luv = itg.trapezoid(bbody_absorbed(np.arange(1,wlref[0]),T1,R1,bluecut,sup),np.arange(1,wlref[0])) * 1e40
    Luv_err = Luv*np.sqrt((2*R1_err/R1)**2+(4*T1_err/T1)**2)

    # NIR luminosity from integrating blackbody above reddest band
    Lnir = itg.trapezoid(bbody_absorbed(np.arange(wlref[-1],25000),T1,R1,bluecut,sup),np.arange(wlref[-1],25000)) * 1e40
    Lnir_err = Lnir*np.sqrt((2*R1_err/R1)**2+(4*T1_err/T1)**2)


    out2.write('%.2f\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\n' %(ph,T1*1e3,T1_err*1e3,R1*1e15,R1_err*1e15,L1bb,L1bb_err))

    # Estimate total bolometric luminosity as integration over observed flux, plus corrections in UV and NIR from the blackbody extrapolations
    # If separate UV fit was used, Luv comes from this fit and Lnir comes from optical-only fit
    # If no separate fits, Luv and Lnir come from the same BB (inferior fit and therefore less accurate extrapolation)
    L2 = Luv + itg.trapezoid(flux,wlref) + Lnir
    # Add errors on each part of the luminosity in quadrature
    L2_err = np.sqrt(L1_err**2 + (Luv_err)**2 + (Lnir_err)**2)
    # Append to light curve
    L2arr.append(L2)
    L2err_arr.append(L2_err)

    # Write light curve to file: L1 is pseudobolometric, L2 is full bolometric
    out1.write('%.2f\t%.2e\t%.2e\t%.2e\t%.2e\n' %(ph,L1,L1_err,L2,L2_err))

    plt.draw()
    plt.xlabel('Wavelength (Ang)')
    plt.ylabel(r'$\mathit{L}_\lambda$ + constant')

    # Counter shifts down next SED on plot for visibility
    k += 1

plt.figure(2)
plt.yticks([])
plt.xlim(min(wlref)-2000,max(wlref)+3000)
plt.ylim(min(flux)/1e40-fscale*k,max(flux)/1e40+fscale*k)
plt.tight_layout(pad=0.5)
plt.legend(numpoints=1,ncol=2,fontsize=16,frameon=True,loc='upper right')

# Add methodologies and keys to output files so user knows which approximations were made in this run
out1.write('\n#KEY\n# Lobs = integrate observed fluxes with no BB fit\n# L+BB = observed flux + BB fit extrapolation')
out1.write('\n# See logL-obs_'+sn+'_'+filters+'.txt and logL-bb_'+sn+'_'+filters+'.txt for simple LC files')
out1.write(method)

# Close output files
out1.close()
out2.close()

# Make final light curves into numpy arrays
L1arr = np.array(L1arr)
L1err_arr = np.array(L1err_arr)
L2arr = np.array(L2arr)
L2err_arr = np.array(L2err_arr)

print('\n\n*** Done! Displaying bolometric light curve ***')

# Save convenient log versions of light curves
logout = np.array(list(zip(phase,np.log10(L1arr),0.434*L1err_arr/L1arr)))
logoutBB = np.array(list(zip(phase,np.log10(L2arr),0.434*L2err_arr/L2arr)))

np.savetxt(outdir+'/logL-obs_'+sn+'_'+filters+'.txt',logout,fmt='%.3f',delimiter='\t')
np.savetxt(outdir+'/logL-bb_'+sn+'_'+filters+'.txt',logoutBB,fmt='%.3f',delimiter='\t')


# Plot final outputs
plt.figure(3,(8,8))
plt.clf()

plt.subplot(311)

# Plot pseudobolometric and bolometric (including BB) light curves (logarithmic versions)
plt.errorbar(logout[:,0],logout[:,1],logout[:,2],fmt='o',color='k',markersize=12,label='Observed flux only')
plt.errorbar(logoutBB[:,0],logoutBB[:,1],logoutBB[:,2],fmt='d',color='r',markersize=9,label='Plus BB correction')
plt.ylabel(r'$log_{10} \mathit{L}_{bol}\,(erg\,s^{-1})$')
plt.legend(numpoints=1,fontsize=16)
plt.xticks(visible=False)

# Get blackbody temperature and radius
bbresults = np.genfromtxt(outdir+'/BB-params_'+sn+'_'+filters+'.txt')

# Plot temperature in units of 10^3 K
plt.subplot(312)
plt.errorbar(bbresults[:,0],bbresults[:,1]/1e3,bbresults[:,2]/1e3,fmt='o',color='k',markersize=12,label='Fit all bands')
plt.ylabel(r'$\mathit{T}_{BB}\,(10^3K)$')
plt.xticks(visible=False)

if len(bbresults[0])==13:
    # If separate fit to optical-only, plot this too
    plt.errorbar(bbresults[:,0],bbresults[:,7]/1e3,bbresults[:,8]/1e3,fmt='s',color='c',markersize=8,label=r'Fit >3000$\AA$')
    plt.legend(numpoints=1,fontsize=16)

# Plot radius in units of 10^15 cm
plt.subplot(313)
plt.errorbar(bbresults[:,0],bbresults[:,3]/1e15,bbresults[:,4]/1e15,fmt='o',color='k',markersize=12,label='Fit all bands')
plt.ylabel(r'$\mathit{R}_{BB}\,(10^{15}cm)$')

if len(bbresults[0])==13:
    plt.errorbar(bbresults[:,0],bbresults[:,9]/1e15,bbresults[:,10]/1e15,fmt='s',color='c',markersize=8,label='Exclude UV')

# X-label for all subplots
plt.xlabel(xlab)

plt.subplots_adjust(hspace=0)
plt.tight_layout(pad=0.5)
plt.draw()
plt.show()


plt.figure(1)
plt.savefig(outdir+'/interpolated-lcs_'+sn+'_'+filters+'.pdf')

plt.figure(2)
plt.savefig(outdir+'/bb_fits_'+sn+'_'+filters+'.pdf')

plt.figure(3)
plt.savefig(outdir+'/results_'+sn+'_'+filters+'.pdf')


# Wait for key press before closing plots!
fin = input('\n\n> PRESS RETURN TO EXIT...\n')
