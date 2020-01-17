# Superbol

[![DOI](https://zenodo.org/badge/73849147.svg)](https://zenodo.org/badge/latestdoi/73849147)

Python program to calculate a bolometric luminosity from a set of input magnitudes, which can be
apparent mags or absolute mags.

Requirements and usage:
- numpy
- scipy
- matplotlib
- (astropy)

To run code:

    python superbol.py


- superbol.py  (Python 3 version, preferred)
- superbol-py2.py (Python 2 version)

Maintenance and updates will prioritise python 3 version, but python 2 version also appears to work.

Versions
------------
    Version 1.3 : Minor tweaks to output plots (MN)
    Version 1.2 : Add extinction correction as an option (MN)
    Version 1.1 : Add bibliographic reference, output file now includes K-correction info (MN)
    Version 1.0 : Release version, Nicholl 2018 RNAAS (MN)
    Version 0.17: Fix bug to write nans instead of blanks when BB fit fails
    Version 0.16: Correct inconsistency in x axis labels, automatically exit if <2 filters used
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
    (note: pre-release version numbers have been downgraded from 1.x->0.x from .14 onwards)

# Usage

The program does not do extinction corrections or K-corrections from spectra.
K-correction not strictly required, if user chooses to shift SED to rest-frame before integration.
Extinction corrections should be applied before use (for now).

Should run in directory with photometry data. User prompted at all steps
(goal: no prior knowledge of superbol required to use it!)

The user is prompted for the input files, which should be called SNname_filters.txt, e.g.

    PTF12dam_ugriz.txt, LSQ14bdq_JHK.dat

Multiple files per transient are allowed, with different filters included in each.
The format of files must be:

    MJD filter1 err1 filter2 err2...

MJD can be replaced by phase or some other time parameter, but must be consistent between files.
- Important: Bands must be in their common systems: AB mag for ugrizy, Gaia, ATLAS, GALEX; Vega mags for UBVRIJHK and Swift (S=UVW2 D=UVM2 A=UVW1)
- Important : Order of filter magnitudes in file must match order of filters in filename.

Computes pseudobolometric light curves by integrating flux over observed filters only ("Lobs") as well as full
bolometric light curves with blackbody extrapolations ("Lbb"). BB fit parameters
(temperature and radius) are also saved as output. Optical and UV can be fit
separately to mimic line blanketing.

The user will be prompted for redshift or distance modulus to allow
calculation of the luminosity. If a redshift is entered then a
standard cosmology is employed to determine distance modulus.
Default cosmology module has been updated to use astropy, but Ned Wright's
cosmocalc is also available (just uncomment that section in code)

If some filters only have a few epochs, code implements a choice of interpolation/extrapolation
based on either constant colours or interactve polynomial fitting. All
interpolated light curves are saved as output. On subsequent runs,
code will detect presence of interpolated light curves so you can
choose to skip interpolation stage. Output of each run of the code will contain all the filters used in the integration in the filenames


Steps:
-------
 - Find files associated with SN and determine available filters and data
 - Correct for time dilation, distance, and approximate K-correction if desired
 - Map light curves in each filter to a common set of times
    (typically defined by the filter with most observations)
    - Interpolation options: linear between neighbouring epochs or polynomial fits
       (user determines order of polynomial interactively)
    - Extrapolation: using polynomials or assuming constant colour with respect to reference filter.
        Large extrapolations = large uncertainties!
    - Save interpolated light curves for reproducability
 - Fit blackbodies to SED at each epoch (most SNe can be reasonably approximated by blackbody above ~3000 Angstroms). In UV, user can choose to:
    - fit SED over all wavelengths with single blackbody
    - fit separate blackbodies to optical and UV (if UV data exist).
        Optical fit gives better temperature estimate than single BB.
         UV fit used only to extrapolate flux for bolometric luminosity.
    - use a simple prescription for line blanketing at UV wavelengths,
        defined as L_uv(lambda < cutoff) = L_bb(lambda)*(lambda/cutoff)^x, where x is chosen by user.
        Cutoff is either set to bluest available band, or if bluest band is >3000A, cutoff = 3000A
- Numerically integrate observed SEDs, and account for missing UV and NIR flux using blackbody extrapolations.
    NIR is easy, UV uses options described above

Outputs
------
- interpolated_lcs_<SN>_<filters>.txt
    - multicolour light curves mapped to common times.
    - Footer gives methods of interpolation and extrapolation.
    - If file exists, can be read in future to skip interpolating next time.
- bol_<SN>_<filters>.txt
    - main output.
    - Contains pseudobolometric light curve, integrated trapezoidally,
    and bolometric light curve including the additional BB corrections, and errors on each.
    - Footer gives filters and method of UV fitting.
- logL_obs_<SN>_<filters>.txt
    - same pseudobolometric (observed) light curve, in convenient log form
- logL_obs_<SN>_<filters>.txt
    - light curve with the BB corrections, in convenient log form
- BB_params_<SN>_<filters>.txt
    - fit parameters for blackbodies: T, R and inferred L from Stefan-Boltzmann law (can compare with direct integration method).
    - If separate optical/UV fit, gives both T_bb (fit to all data) and T_opt (fit only to data >3000 A)


Recommended practice: run once with ALL available filters, and fit missing
data as best you can. Then re-run choosing only the well-observed
filters for the integration. You can compare results and decide for
yourself whether you have more belief in the "integrate all filters
with light curve extrpolations" method or the "integrate only the
well-sampled filters and account for missing flux with blackbodies"
method.

    To-do:
        - set error floor for interpolation to ref band error
        - make compatible with other inputs (Open Supernova Catalog, output from psf.py)
        - include extinction correction


# Notes on various filter systems (courtesy Stephen Smartt)

AB system flux zeropoints (used for SDSS-like filters)
-------------------------------------------------------

It is worth noting the origin of these numbers for clarifation and to avoid confusion and use of incorrect zeropoints
From the definition of AB magnitudes

     m(AB) = -2.5 log(f_nu) - 48.60.

Therefore, AB magnitudes are directly related to a physical flux.
They are defined as monocrhomatic fluxes. Bandpasses need to be defined at an effective wavelength

f_nu is recovered from the above in units of ergs/s/cm2/Hz such that
 m(AB) = 0 has a flux of f_nu = 3.63E-20 erg/s/cm2/Hz  = 3631 Jy
 1 Jy = 1 Jansky = 10E-26 W Hz-1 m-2 = 10-23 erg s-1 Hz-1 cm-2
Working through the conversion to ergs/s/cm2/Angs (and making sure all units are respected)
 f_lam = 0.1089/(lambda_eff^2)
 where lambda_eff is the effective wavelength of the filter **in angstroms**.
Note then that the AB flux zeropoint is defined ONLY by the choice of effective wavelength of the
bandpass


These effective wavelengths for SDSS filters are from Fukugita et al. (1996, AJ, 111, 1748) and are
the wavelength weighted averages (effective wavelenghts in their Table 2a, first row)

Then then lead to the following fluxes for the SDSS filters (in 10^-11 erg/s/cm2/Angs) for the
zeropoints of the system (i.e. for m_AB = 0)

SDSS filter   lambda_eff  Flux ZP (erg/s/cm2/Angs)
'u'           3560        859.5e-11
'g'           4830        466.9e-11
'r'           6260        278.0e-11
'i'           7670        185.2e-11
'z'           9100        131.5e-11

One should note that these are listed on
Paul Martini's page : http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
However that is not an original source - it simply uses the f_lam =0.1089/(lambda_eff^2) relation,
and the effective wavelengths from Fukugita et al.

Vega based system zeropoints (used for Johnson-Cousins-Glass filters and Swift)
-----------------------------------------------------------------------------------

The values for UBVRIJHK are for the Johnson-Cousins-Glass system and are taken directly
from Bessell et al. 1998, A&A, 333, 231 (Paul Martini's page also lists these verbatim)
Note that these Bessell et al. (1998) values were calculated not from the spectrum of
Vega itself, but from a Kurucz model atmosphere of an AOV star.

The central wavelengths and the zeropoint fluxes are
     lam_eff  Flux ZP (erg/s/cm2/Angs)
'U'   3600   417.5e-11
'B'   4380   632.0e-11
'V'   5450   363.1e-11
'R'   6410   217.7e-11
'I'   7980   112.6e-11
'J'   12200  31.47e-11
'H'   16300  11.38e-11
'K'   21900  3.961e-11

The R and I filters are in the Cousins system (not the Johnson system), and the JHK are in the
Glass system. The overall system above is often referred to as the Johnson-Cousins-Glass system.

The Swift UV magnitudes are expected to be Vega based mags and therefore have the following
effective wavelengths an flux zeropints.
**[WE NEED TO DOUBLE CHECK THE EXACT  ORIGINS OF THESE ZEROPINTS, BUT THEY ARE CORRECT TO ~ few %]**

Filter     lam_eff  Flux ZP (erg/s/cm2/Angs)
'S': UVW2    2030   536.2e-11
'D': UVM2    2230   463.7e-11
'A': UVW1    2590   412.3e-11



Uncertainties in the calculation of zeropoints
-------------------------------------------------

The conversion between magnitude, in any system, and physical flux is
not without significant uncertainty. For Vega based magnitudes, the
uncertainty is mainly in the measurement of Vega's flux in the
specific wavebands, or in the model used. For AB mags, the uncertainty
is in the value of effective wavelength to use since this is dependent
on the spectrum of the specific object (see Fukugita et al. Section
2.1). The effective wavelength is not simply the central or average wavelength, it depends
on the input spectrum.

As a comparison, a Space Telescope Science Institute report in 1996 by
 L. Colina, R. Bohlin, F. Castelli (Instrument Science Report
 CAL/SCS-008, "Absolute Flux Calibrated Spectrum of Vega") gives the
 *measured* UV to optical flux calibrated spectrum and flux zeropints
 of Vega,and the NIR model magnitudes. These are below :


   CBC96 obs  CBC model      B98 model   percentage difference between CBC96 and B98
 U 4.34e-9    ...              4.18e-9         3.7
 B 6.40e-9    ...              6.32e-9         1.3
 V 3.67e-9    ...              3.63e-9         1.1
 J 3.06e-10   3.05e-10         3.15e-10        2.9
 H 1.24e-10   1.10e-10         1.14e-10        8.1
 K 3.92e-11*  3.82e-10         3.96e-11        1.0
*Although one set of measurements from CBC96 gives 4.19e-11 for K

Note that R and I are not included here for comparison, since CBC96 provides them in the
Johnson filters, but B98 provides them in Cousins and they are not the same.
This program assumes that input values are in the Cousins system


 This superbol programme does not propogate the errors above through
 the filter calculations and include them in the error budget of the
 object.  Future versions should aim to do this. Although the H-band
 uncertainty is as high as 8.1%, this will not propogate to an 8%
 error in L_bol, since the the total would uncertainty contribution
 will be a flux weighted average over all these uncertainties. To
 first order, a straight average of the above is 3%, which is a
 reasonabe estimate of error contribution from the Vega flux
 calibration


For the AB mags, the range of effective wavelengths calculated for
ugriz by Fukugita et al. (no atmospheric extinction and airmass = 1.2,
compared with weighting by wavelength/frequency) gives maximum deviations of
4%,6%,3%,2%,2% or a straight average of 3%.
Therefore an added uncertainty of 3% to the L_bol uncertainties produced
by this program should be considered.
