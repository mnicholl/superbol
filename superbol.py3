#!/usr/bin/env python

'''
    SUPERBOL: Supernova Bolometric Light Curves
    Written by Matt Nicholl, 2015

    Version 1.9 : Only do separate UV fit if > 2 UV filters (MN)
    Version 1.8 : New feature! Can choose to shift SED to rest-frame for data with no K-correction (MN)
    Version 1.7 : Improved handling of errors (MN)
    Version 1.6 : Tidied up separate blackbody integrations in UV/NIR (MN)
    Version 1.5 : Specifying files on command line now COMMA separated to handle 2-digit indices (MN)
    Version 1.4 : Fixed bug in blackbody formula - missing factor of pi led to overestimate of radius (MN)
    Version 1.3 : Added GALEX NUV and FUV bands in AB system (MN)
    Version 1.2 : Swift effective wavelengths now match Poole et al 2008 (update by MN)
    Version 1.1 : Origin and veracity of all zeropoints checked by SJS. Comments added, with more details in superbol.man file. Archived this version in /home/sne/soft
    Version 1.0 : Written by Matt Nicholl (QUB), 2015

    See superbol.man for the manual file and more details.

    Requirements and usage:

    Needs numpy, scipy and matplotlib

    Input files should be called SNname_filters.EXT, eg PTF12dam_ugriz.txt, LSQ14bdq_JHK.dat, etc

    Can have multiple files per SN with different filters in each

    Format of files must be:

    MJD filter1 err1 filter2 err2...

    MJD can be replaced by phase or some other time parameter, but must be consistent between files.

    Important : Order of filter magnitudes in file must match order of filters in filename.

    Output of each run of the code will contain all the filters used in the integration in the filenames

    Computes pseudobolometric light curves "Lobs" as well as full bolometric with blackbody extrapolations "Lbb" - BB fit parameters (temperature and radius) saved as output.
    Optical and UV fit separately to mimic line blanketing

    If some filters only have a few epochs, implements constant-colour and/or interactve polynomial fitting to interpolate/extrapolate - all interpolated light curves also saved as output. On subsequent runs, code will detect presence of interpolated light curves so you can choose to skip polynomial fits if you want to rerun with fewer filters in the integration

    Best practice: run once with ALL available filters, and fit missing data as best you can. Then re-run choosing only the well-observed filters for the integration. You can compare results and decide for yourself whether you have more belief in the "integrate all filters with light curve extrpolations" method or the "integrate only the well-sampled filters and account for missing flux with blackbodies" method.

    '''


import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as itg
from scipy.optimize import curve_fit
from scipy.interpolate import interpolate as interp
import glob
import sys
import os
#import time


print('\n    * * * * * * * * * * * * * * * * * * * * *\n    *                                       *\n    *        Welcome to `SUPER BOL`!        *\n    *   SUPernova BOLometric light curves   *\n    *                                       *\n    *                ______                 *\n    *               {\   */}                *\n    *                 \__/                  *\n    *                  ||                   *\n    *                 ====                  *\n    *                                       *\n    *           M. Nicholl (V1.8)           *\n    *                                       *\n    * * * * * * * * * * * * * * * * * * * * *\n\n')


plt.ion()

def bbody(lam,T,R):
    A = 4*np.pi*R**2
    Blam = A*(2*np.pi*6.626e-27*(3e10)**2/(lam*1e-8)**5)/(np.exp(6.627e-27*3e10/(lam*1e-8*1.38e-16*T))-1)/1e8
    return Blam


def easyint(x,y,err,xref,yref):
    ir = (xref>=min(x))&(xref<=max(x))
    yint = interp.interp1d(x[np.argsort(x)],y[np.argsort(x)])(xref[ir])
    yout = np.zeros(len(xref),dtype=float)
    ylow = yint[np.argmin(xref[ir])]-yref[ir][np.argmin(xref[ir])]+yref[xref<min(x)]
    yup  = yint[np.argmax(xref[ir])]-yref[ir][np.argmax(xref[ir])]+yref[xref>max(x)]
    yout[ir] = yint
    yout[xref<min(x)] = ylow
    yout[xref>max(x)] = yup
    errout = np.zeros(len(xref),dtype=float)
    errout[ir] = max(np.mean(err),0.1)
    errout[xref<min(x)] = np.sqrt((min(x) - xref[xref<min(x)])**2/1.e4 + np.mean(err)**2)
    errout[xref>max(x)] = np.sqrt((xref[xref>max(x)] - max(x))**2/1.e4 + np.mean(err)**2)

    return yout,errout


# Filter information

#SDSS filters and AB mags:
#These effective wavelengths for SDSS filters are from Fukugita et al. (1996, AJ, 111, 1748) and are
#the wavelength weighted averages (effective wavelenghts in their Table 2a, first row)

#See superbol.mann for more details
#For AB mags,
#     m(AB) = -2.5 log(f_nu) - 48.60.
# f_nu is in units of ergs/s/cm2/Hz such that
#    m(AB) = 0 has a flux of f_nu = 3.63E-20 erg/s/cm2/Hz  = 3631 Jy
# Therefore, AB magnitudes are directly related to a physical flux.
# Working through the conversion to ergs/s/cm2/Angs, gives
# f_lam = 0.1089/(lambda_eff^2)  where lambda_eff is the effective wavelength of the filter in angstroms
# Note then that the AB flux zeropoint is defined ONLY by the choice of effective wavelength of the bandpass

# Matt originally listed th following from  Paul Martini's page : http://www.astron##omy.ohio-state.edu/~martini/usefuldata.html
# That is not an original source, for AB mags it simply uses the f_lam =0.1089/(lambda_eff^2) relation, and the effective wavelengths from Fukugita et al.

#Effective wavelengths (in Angs)
wle = {'u': 3560, 'g': 4830, 'r': 6260, 'i': 7670, 'z': 9100, 'U': 3600,
        'B': 4380, 'V': 5450, 'R': 6410, 'I': 7980, 'J': 12200, 'H': 16300,
        'K': 21900, 'S': 2030, 'D': 2231, 'A': 2634, 'F': 1516, 'N': 2267}

#Zeropoints (ugriz and GALEX NUV/FUV are in AB mag system, UBVRI are Johnson-Cousins in Vega mag, JHK are Glass system Vega mags, and Swift UVOT SDA are in Vega mag system)
#
#The values for UBVRIJHK are for the Johnson-Cousins-Glass system and are taken directly from Bessell et al. 1998, A&A, 333, 231 (Paul Martini's page lists these verbatim)
#Note that these Bessell et al. (1998) values were calculated not from the spectrum of Vega itself, but from a Kurucz model atmosphere of an AOV star.
#GALEX effective wavelengths from here: http://galex.stsci.edu/gr6/?page=faq

#All values in 1e-11 erg/s/cm2/Angs
zp = {'u': 859.5, 'g': 466.9, 'r': 278.0, 'i': 185.2, 'z': 131.5, 'U': 417.5,
        'B': 632, 'V': 363.1, 'R': 217.7, 'I': 112.6, 'J': 31.47, 'H': 11.38,
        'K': 3.961, 'S': 536.2, 'D': 463.7, 'A': 412.3, 'F': 4801., 'N': 2119.}

#Filter widths (in Angs)
width = {'u': 458, 'g': 928, 'r': 812, 'i': 894, 'z': 1183, 'U': 485,
            'B': 831, 'V': 827, 'R': 1389, 'I': 899, 'J': 1759, 'H': 2041,
            'K': 2800, 'S': 671, 'D': 446, 'A': 821, 'F': 268, 'N': 732}


cols = {'u': 'dodgerblue', 'g': 'g', 'r': 'r', 'i': 'goldenrod', 'z': 'k',
        'U': 'slateblue', 'B': 'b', 'V': 'yellowgreen', 'R': 'crimson',
        'I': 'chocolate', 'J': 'darkred', 'H': 'orangered', 'K': 'saddlebrown',
        'S': 'mediumorchid', 'D': 'purple', 'A': 'midnightblue',
        'F': 'hotpink', 'N': 'cyan'}

bandlist = 'FSDNAuUBgVrRiIzJHK'


print('\n######### Step 1: input files and filters ##########')

useInt = 'n'

sn = input('\n> Enter SN name:   ')

if not sn:
    print('\n* No name given; lets just call it `SN`...')
    sn = 'SN'

outdir = 'superbol_output_'+sn
if not os.path.exists(outdir): os.makedirs(outdir)



# Get photometry files
do1 = input('\n> Find input files automatically?[y]   ')
if not do1: do1='y'

use1 = []

if do1 == 'y':
    files = glob.glob(outdir+'/interpolated_lcs_'+sn+'*')

    if len(files)>0:
        print('\n* Interpolated LC(s) already available:')

        for i in range(len(files)):
            print('  ', i, ':', files[i])

        use = input('\n> Use interpolated LC? (e.g. 0 for file 0, or n for no) [0]   ')

        if not use: use1.append(0)

        if use!='n':
            useInt = 'y'
            if len(use)>0:
                for i in use.split(','):
                    use1.append(i)
            else:
                for i in range(len(files)):
                    use1.append(i)
        else: print('\n* Not using interpolated data')


    if len(files)==0 or use=='n':
        files = glob.glob(sn+'_*')

        if len(files)>0:
            print('\n* Available files:')

            for i in range(len(files)):
                print('  ', i, ':', files[i])

            use = input('\n> Specify files to use (e.g. 0,2 for files 0 and 2) [all]   ')
            if len(use)>0:
                for i in use.split(','):
                    use1.append(i)
            else:
                for i in range(len(files)):
                    use1.append(i)

        else:
            do1 = 'n'

            print('* No files found for '+sn)


if do1 != 'y':
    files1 = input('\n> Enter all file names separated by commas:\n')
    if not files1:
        print('No files given - exiting!')
        sys.exit(0)

    files = []
    for i in files1.split(','):
        files.append(i)
    for i in range(len(files)):
        use1.append(i)


# compile filters and LCs from input
lc = {}

filts2 = str()

for i in use1:
    i = int(i)
    filts1 = files[i].split('.')[0]
    filts1 = filts1.split('_')[-1]
    filts2 += filts1

    d = np.genfromtxt(files[i])
    x = 1
    for j in filts1:
        lc[j] = np.array(list(zip(d[:,0][~np.isnan(d[:,x])],d[:,x][~np.isnan(d[:,x])],d[:,x+1][~np.isnan(d[:,x])])))
        x+=2


# sort known filters:
filters = str()

for i in bandlist:
    if i in filts2:
        filters += i

# add unknown filters manually
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



print('\n######### Step 2: reference band for phase info ##########')



# Choose reference band
plt.figure(1,(10,8))
plt.clf()

for i in filters:
    plt.errorbar(lc[i][:,0],lc[i][:,1],lc[i][:,2],fmt='o',color=cols[i],label=i)

plt.gca().invert_yaxis()

plt.xlabel('Days')

plt.ylabel('Magnitude')

plt.legend(numpoints=1)

plt.draw()

ref1 = 0

for i in filters:
    ref2 = len(lc[i])
    if ref2>ref1:
        ref1 = ref2
        ref3 = i



print('\n* Displaying all available photometry...')

t3 = input('\n> Enter bands to use (blue to red) ['+filters+']   ')
if not t3: t3 = filters

filters = t3


if useInt!='y':
    ref = input('\n> Choose reference filter for sampling epochs\n   Suggested (most LC points): ['+ref3+']   ')

    if not ref: ref = ref3

else: ref = filters[0]

print('\n* Using '+ref+'-band for reference')


# Convert MJD to rest-frame phase
t1 = input('\n> Interactively find '+ref+'-band maximum?[n] ')

if not t1:
    t1 = 'n'

    doSh = input('\n> Shift to approx maximum?[n] ')

    if not doSh: doSh = 'n'

    if doSh=='y':
        d = lc[ref]

        shift = d[:,0][np.argmin(d[:,1])]

        for j in lc:
            lc[j][:,0]-=shift

        print('\n* Approx shift done')


if t1!='n':

    d = lc[ref]

    shift = d[:,0][np.argmin(d[:,1])]

    d[:,0]-=shift


    plt.clf()

    plt.errorbar(d[:,0],d[:,1],d[:,2],fmt='o',color=cols[ref])

    plt.ylim(max(d[:,1])+0.2,min(d[:,1])-0.2)

    plt.xlabel('Days from approx maximum')

    plt.ylabel('Magnitude')

    plt.draw()


    happy = 'n'

    print('\n### Begin polynomial fit to peak... ###')

    order1 = 4

    Xup1 = 50

    while happy == 'n':

        print('\n### Select data range ###')

        Xup = input('>> Cut-off phase for polynomial fit?['+str(Xup1)+']   ')

        if not Xup: Xup = Xup1

        Xup = float(Xup)

        Xup1 = Xup

        d1 = d[d[:,0]<Xup]

        plt.clf()

        plt.errorbar(d1[:,0],d1[:,1],d1[:,2],fmt='o',color=cols[ref])

        plt.ylim(max(d1[:,1])+0.4,min(d1[:,1])-0.2)

        plt.draw()


        order = input('\n>> Order of polynomial to fit?['+str(order1)+']   ')

        if not order: order = order1

        order = int(order)

        order1 = order

        fit = np.polyfit(d1[:,0],d1[:,1],deg=order)

        days = np.arange(min(-40,min(d[:,0]))-10,Xup)

        eq = 0

        for i in range(len(fit)):
            eq += fit[i]*days**(order-i)


        plt.plot(days,eq,label='Fit order = %d' %order)


        plt.ylabel('Magnitude')

        plt.xlabel('Days from approx maximum')

        plt.legend(numpoints=1)

        plt.xlim(min(d[:,0])-5,Xup)

        plt.draw()

        happy = input('\n> Happy with fit?(y/[n])   ')

        if not happy: happy = 'n'


    new_peak = input('> Use [p-olynomial] or o-bserved peak date?    ')


    if not new_peak: new_peak = 'p'

    if new_peak=='p':
        peak = days[np.argmin(eq)]
        d[:,0] -= peak
        plt.clf()
        plt.errorbar(d[:,0],d[:,1],d[:,2],fmt='o',color=cols[ref])
        plt.ylabel('Magnitude')
        plt.xlabel('Days from maximum')
        plt.ylim(max(d[:,1])+0.2,min(d[:,1])-0.2)
        plt.draw()

    if new_peak == 'o':
        peak = 0

    for j in lc:
        lc[j][:,0]-=(shift+peak)

    lc[ref][:,0]+=(shift+peak)


plt.figure(1)
plt.clf()

for i in filters:
    plt.errorbar(lc[i][:,0],lc[i][:,1],lc[i][:,2],fmt='o',color=cols[i],label=i)

plt.gca().invert_yaxis()

plt.xlabel('Days from '+ref+'-band maximum')

plt.ylabel('Magnitude')

plt.legend(numpoints=1)

plt.draw()




skipK = 'n'

# Get SN distance
z = input('\n> Please enter SN redshift or distance modulus:[0]  ')

if not z: z=0

z = float(z)

if z<10:

    print('Redshift entered (or DM=0)')

    t2 = ''

    if lc[ref][0,0]>25000 or useInt=='y':
        t2 = input('\n> Correct for time-dilation?[n] ')

        if not t2: t2 = 'n'

    else:
        t2 = input('\n> Correct for time-dilation?[y] ')

        if not t2: t2 = 'y'


    if t2=='y':
        for j in lc:
            lc[j][:,0]/=(1+z)
        print('\n* Displaying corrected phases')


    plt.figure(1)
    plt.clf()

    for i in filters:
        plt.errorbar(lc[i][:,0],lc[i][:,1],lc[i][:,2],fmt='o',color=cols[i],label=i)

    plt.gca().invert_yaxis()

    plt.xlabel('Days from '+ref+'-band maximum')

    plt.ylabel('Magnitude')

    plt.legend(numpoints=1)

    plt.draw()



    print('\n######### Step 3: Flux scale ##########')



    ################# cosmocalc by N. Wright ##################

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

    #############################################

    print('\n* First '+ref+'-band mag = %.2f' %lc[ref][0,1])
    absol='n'
    if lc[ref][0,1] < 0:
        absol = input('> Magnitudes are in Absolute mags, correct?[y] ')
        if not absol: absol='y'
    else:
        absol = input('> Magnitudes are in Apparent mags, correct?[y] ')
        if not absol: absol ='n'

    if absol=='y':
        DL_Mpc = 1e-5
        print('\n* Absolute mags; Luminosity distance = 10 pc')
    else:
        print('\n* Luminosity distance = %.2e Mpc' %DL_Mpc)


    dist = DL_Mpc*3.086e24 # convert Mpc to cm, since flux in erg/s/cm2/A

else:
    print('Distance modulus entered')

    skipK = 'y'

    for i in lc:
        lc[i][:,1]-=z
        dist = 1e-5*3.086e24


doKcorr = 'n'

if skipK == 'n':
    doKcorr = input('\n> Do you want to covert flux and wavelength to rest-frame?\n'
                            '  (skip this step if data are already K-corrected) [n]   ')



if useInt!='y':

    for i in lc:
        lc[i] = lc[i][lc[i][:,0].argsort()]

    print('\n######### Step 4: Interpolate LCs to ref epochs ##########')


    lc_int = {}

    lc_int[ref] = lc[ref]



    t4 = input('\n> Interpolate light curves interactively?[y] ')

    if not t4: t4 = 'y'

    if t4=='y':
        print('\n### Begin polynomial fit... ###')

        # Interpolate / extrapolate other bands to same epochs - polynomial fits
        # - what if there are only one or two points??? Use colour?
        # Include error in extrapolation

        intKey = '\n# Reference band was '+ref

        for i in filters:

            if i!=ref:

                print('\n### '+i+'-band ###')


                order1 = 4

                happy = 'n'

                while happy == 'n':


                    plt.clf()
                    plt.errorbar(lc[i][:,0],lc[i][:,1],lc[i][:,2],fmt='o',color=cols[i],label=i)
                    plt.errorbar(lc[ref][:,0],lc[ref][:,1],lc[ref][:,2],fmt='o',color=cols[ref],label=ref)
                    plt.gca().invert_yaxis()
                    plt.legend(numpoints=1)
                    plt.xlabel('Days from '+ref+'-band maximum')
                    plt.ylabel('Magnitude')
                    plt.ylim(max(max(lc[ref][:,1]),max(lc[i][:,1]))+0.5,min(min(lc[ref][:,1]),min(lc[i][:,1]))-0.5)
                    plt.draw()


                    order = input('\n>> Order of polynomial to fit?(q to quit and use constant colour)['+str(order1)+']   ')

                    if order == 'q':
                        break

                    if not order: order = order1

                    order = int(order)

                    order1 = order

                    fit = np.polyfit(lc[i][:,0],lc[i][:,1],deg=order)

                    days = np.arange(np.min(lc[ref][:,0]),np.max(lc[ref][:,0]))

                    eq = 0

                    for j in range(len(fit)):
                        eq += fit[j]*days**(order-j)


                    plt.plot(days,eq,label='Fit order = %d' %order)

                    plt.ylabel('Magnitude')

                    plt.xlabel('Days from '+ref+'-band maximum')

                    plt.legend(numpoints=1)

                    plt.draw()

                    happy = input('\n> Happy with fit?(y/[n])   ')

                    if not happy: happy = 'n'

                if order == 'q':
                    tmp1,tmp2 = easyint(lc[i][:,0],lc[i][:,1],lc[i][:,2],lc[ref][:,0],lc[ref][:,1])
                    # ir = (lc[ref][:,0]>=min(lc[i][:,0]))&(lc[ref][:,0]<=max(lc[i][:,0]))
                    # tmp2 = max(np.mean(lc[i][:,2]),0.1)*ir + *~ir
                    tmp = list(zip(lc[ref][:,0],tmp1,tmp2))
                    lc_int[i] = np.array(tmp)
                    print('\n* Interpolating linearly; extrapolating assuming constant colour...')
                    intKey += '\n# '+i+': Linear interp; extrap=c'


                else:

                    intKey += '\n# '+i+': fit order='+str(order)+'; extrap method '


            ##########################
            # Goal here: if band has point at same epoch as ref band, use point; otherwise, interpolate
                    mag_int = []

                    for k in lc[i]:
                        if k[0] in lc[ref][:,0]:
                            mag_int.append(k)

                    tmp_arr = np.array(mag_int)

                    if tmp_arr.size:
                        for k in lc[ref]:
                            if k[0] not in tmp_arr[:,0]:
                                mag = 0
                                for j in range(len(fit)):
                                    mag += fit[j]*k[0]**(order-j)
                                out = np.array([k[0],mag,max(np.mean(lc[i][:,2]),0.1)])
                                mag_int.append(out)
                    else:
                        for l in lc[ref][:,0]:
                            mag = 0
                            for j in range(len(fit)):
                                mag += fit[j]*l**(order-j)
                            out = np.array([l,mag,max(np.mean(lc[i][:,2]),0.1)])
                            mag_int.append(out)


                    mag_int = np.array(mag_int)

                    tmp = mag_int[np.argsort(mag_int[:,0])]



                    ####################
                    try:
                        low = min(lc[i][:,0])
                        up = max(lc[i][:,0])
                        col1 = tmp[tmp[:,0]>low][0,1] - lc[ref][tmp[:,0]>low][0,1]
                        col2 = tmp[tmp[:,0]<up][-1,1] - lc[ref][tmp[:,0]<up][-1,1]
                        early = lc[ref][tmp[:,0]<low][:,1]+col1
                        late = lc[ref][tmp[:,0]>up][:,1]+col2
                        # Compute error as random sum of average error in band plus 0.1 mag for every 10 days extrapolated
                        tmp[:,2][tmp[:,0]<low] = np.sqrt((low - tmp[:,0][tmp[:,0]<low])**2/1.e4 + np.mean(lc[i][:,2])**2)
                        tmp[:,2][tmp[:,0]>up] = np.sqrt((tmp[:,0][tmp[:,0]>up] - up)**2/1.e4 + np.mean(lc[i][:,2])**2)
                    except:
                        raise ValueError('Error likely due to having two points in same band with identical MJD '
                                            '- this breaks interpolator if there is only one point with this MJD in reference band. '
                                            'Try manually removing point or changing MJD by tiny amount!')

                    plt.errorbar(tmp[:,0],tmp[:,1],fmt='s',markersize=12,mfc='none',markeredgewidth=3,markeredgecolor=cols[i],label='Polynomial')
                    plt.errorbar(tmp[tmp[:,0]<low][:,0],early,fmt='o',markersize=12,mfc='none',markeredgewidth=3,markeredgecolor=cols[i],label='Constant colour')
                    plt.errorbar(tmp[tmp[:,0]>up][:,0],late,fmt='o',markersize=12,mfc='none',markeredgewidth=3,markeredgecolor=cols[i])
                    plt.legend(numpoints=1)
                    plt.draw()

                    if len(tmp[tmp[:,0]<low])>0:
                        type = input('\n> Early-time extrapolation:\n  [P-olynomial], c-onstant colour, or a-verage of two methods?\n')
                        if not type: type = 'p'
                        if type == 'c':
                            tmp[:,1][tmp[:,0]<low]=early
                        if type == 'a':
                            tmp[:,1][tmp[:,0]<low]=0.5*(tmp[:,1][tmp[:,0]<low]+early)
                    else: type = 'n'

                    intKey += 'early='+type+';'


                    if len(tmp[tmp[:,0]>up])>0:
                        type = input('\n> Late-time extrapolation:\n  [P-olynomial], c-onstant colour, or a-verage of two methods?\n')
                        if not type: type = 'p'
                        if type == 'c':
                            tmp[:,1][tmp[:,0]>up]=late
                        if type == 'a':
                            tmp[:,1][tmp[:,0]>up]=0.5*(tmp[:,1][tmp[:,0]>up]+late)
                    else: type = 'n'

                    intKey += 'late='+type



                    lc_int[i] = tmp

        intKey += '\n# p = polynomial, c = constant colour, a = average'


    else:
        for i in filters:

            if i!=ref:
                tmp1,tmp2 = easyint(lc[i][:,0],lc[i][:,1],lc[i][:,2],lc[ref][:,0],lc[ref][:,1])
                # ir = (lc[ref][:,0]>=min(lc[i][:,0]))&(lc[ref][:,0]<=max(lc[i][:,0]))
                # tmp2 = 0.1*ir+0.15*~ir
                tmp = list(zip(lc[ref][:,0],tmp1,tmp2))
                lc_int[i] = np.array(tmp)
        print('\n* Interpolating linearly; extrapolating assuming constant colour...')

        intKey = '\n# All light curves linearly interpolated\n# Extrapolation done by assuming constant colour with reference band ('+ref+')'




    int_out = [lc[ref][:,0]]

    for i in filters:
        int_out.append(lc_int[i][:,1])
        int_out.append(lc_int[i][:,2])

    int_out = np.swapaxes(np.array(int_out),0,1)

    int_file = open(outdir+'/interpolated_lcs_'+sn+'_'+filters+'.txt','wb')

    cap = '#phase\t'

    for i in filters:
        cap = cap+i+'\terr\t'

    cap +='\n'

    # int_file.write(cap)

    np.savetxt(int_file,int_out,fmt='%.2f',delimiter='\t',header=cap,footer=intKey,comments='#')


    # int_file.write(intKey)

    int_file.close()

    print('\n* Displaying all interpolated/extrapolated LCs')

    plt.figure(1)
    plt.clf()

    for i in filters:
        plt.errorbar(lc_int[i][:,0],lc_int[i][:,1],lc_int[i][:,2],fmt='o',color=cols[i],label=i)

    plt.gca().invert_yaxis()

    plt.xlabel('Days from '+ref+'-band maximum')

    plt.ylabel('Magnitude')

    plt.legend(numpoints=1)

    plt.ylim(max(max(lc_int[ref][:,1]),max(lc_int[i][:,1]))+0.5,min(min(lc_int[ref][:,1]),min(lc_int[i][:,1]))-0.5)

    plt.draw()

else:
    print('\n* Interpolation already done, skipping step 4!')

    lc_int = {}

    for i in filters:
        lc_int[i] = lc[i]




# Convert mags to flux

# Fit blackbody - separate fit below g band? suppression factor? Stefan-Boltzman or integrate?
# Depends on whether we have UV?
# Errors?

# File output - first column should always be RF phase!


wlref = []
wlref1 = [wle[filters[0]]-width[filters[0]]/2]

fref = []

bandwidths = []

for i in filters:
    wlref.append(float(wle[i]))
    fref.append(zp[i]*1e-11)
    wlref1.append(float(wle[i]))
    bandwidths.append(float(width[i]))

wlref1.append(wle[filters[-1]]+width[filters[-1]]/2)

wlref1 = np.array(wlref1)
wlref = np.array(wlref)
fref = np.array(fref)
bandwidths = np.array(bandwidths)

phase = lc_int[ref][:,0]


# Correct flux and wavelength to rest-frame
if doKcorr == 'y':
    wlref /= (1+z)
    wlref1 /= (1+z)
    fref *= (1+z)
    bandwidths /= (1+z)


k = 1

L1arr = []
L2arr = []
L1err_arr = []
L2err_arr = []
test_arr = []

out1 = open(outdir+'/bol_'+sn+'_'+filters+'.txt','w')
out2 = open(outdir+'/BB_params_'+sn+'_'+filters+'.txt','w')

out1.write('# ph\tLobs\terr\tL+BB\terr\t\n\n')


if np.min(wlref)<3500:
    out2.write('# ph\tT_bb\terr\tR_bb\terr\tT_opt\terr\tR_opt\terr\n\n')
else:
    out2.write('# ph\tT_bb\terr\tR_bb\terr\n\n')


#Find luminosity by integrating blackbody fits. If UV coverage, fit separate BB below 3500A
#Flux errors used as weights. L_err from T_err and R_err in curve_fit
#Also do straight integration of observed fluxes for comparison

print('\n######### Step 5: Fit blackbodies and integrate flux #########')


sup = input('\n> Suppression factor for BB flux bluewards of '+filters[0]+'-band?\n  i.e. L_uv = L_uv(BB)/x [x=1] ')
if not sup: sup = 1

sup = 1/np.float(sup)

print('\n*** Fitting Blackbodies to SED ***')

print('\n* Solid line = blackbody fit for flux extrapolation')

if np.min(wlref)<3500:
    print('* Dashed lines = separate fit to optical and UV for T and R estimates')

if sup!=1:
    print('* Dotted lines = UV flux with assumed blanketing')

fscale = 4*np.pi*dist**2*zp[ref]*1e-11*10**(-0.4*min(lc[ref][:,1]))



plt.figure(2,(12,8))
plt.clf()


for i in range(len(phase)):
    ph = phase[i]
    mags = np.zeros(len(lc_int))
    errs = np.zeros(len(lc_int))
    for j in range(len(filters)):
        mags[j] = lc_int[filters[j]][i,1]
        errs[j] = lc_int[filters[j]][i,2]
    flux = 4*np.pi*dist**2*fref*10**(-0.4*mags)
    ferr = 2.5/np.log(10) * flux * errs
    flux1 = np.insert(flux,0,0)
    flux1 = np.append(flux1,0)


    BBparams, covar = curve_fit(bbody,wlref,flux,p0=(10000,1e15),sigma=ferr)
    T1 = BBparams[0]
    T1_err = np.sqrt(np.diag(covar))[0]
    R1 = BBparams[1]
    R1_err = np.sqrt(np.diag(covar))[1]
#    co = np.abs(covar[0,1])


    plt.figure(2)
    plt.errorbar(wlref,flux-fscale*k,ferr,fmt='o',color=cols[filters[k%len(filters)]],label='%.1f' %ph)
    plt.plot(np.arange(1000,25000),bbody(np.arange(1000,25000),T1,R1)-fscale*k,color=cols[filters[k%len(filters)]],linestyle='-')
    plt.plot(np.arange(1000,wlref[0]),bbody(np.arange(1000,wlref[0]),T1,R1)*sup-fscale*k,color=cols[filters[k%len(filters)]],linestyle=':')

    L1 = itg.trapz(flux1,wlref1)

    L1_err = np.sqrt(np.sum((bandwidths*ferr)**2))

    L1arr.append(L1)
    L1err_arr.append(L1_err)

    test = 4*np.pi*R1**2*5.67e-5*T1**4
    test_arr.append(test)

    Luv = itg.trapz(bbody(np.arange(1000,wlref[0]),T1,R1),np.arange(1000,wlref[0]))*sup
    Luv_err = Luv*np.sqrt((2*R1_err/R1)**2+(4*T1_err/T1)**2)

    Lnir = itg.trapz(bbody(np.arange(wlref[-1],25000),T1,R1),np.arange(wlref[-1],25000))
    Lnir_err = Lnir*np.sqrt((2*R1_err/R1)**2+(4*T1_err/T1)**2)


    if len(wlref[wlref<3000])>2:
        try:
            BBparams, covar = curve_fit(bbody,wlref[wlref>3200],flux[wlref>3200],p0=(10000,1e15),sigma=ferr[wlref>3200])
            Topt = BBparams[0]
            Topt_err = np.sqrt(np.diag(covar))[0]
            Ropt = BBparams[1]
            Ropt_err = np.sqrt(np.diag(covar))[1]

            Lnir = itg.trapz(bbody(np.arange(wlref[-1],25000),Topt,Ropt),np.arange(wlref[-1],25000))
            Lnir_err = Lnir*np.sqrt((2*Ropt_err/Ropt)**2+(4*Topt_err/Topt)**2)

            BBparams, covar = curve_fit(bbody,wlref[wlref<4000],flux[wlref<4000],p0=(10000,1e15),sigma=ferr[wlref<4000])
            Tuv = BBparams[0]
            Tuv_err = np.sqrt(np.diag(covar))[0]
            Ruv = BBparams[1]
            Ruv_err = np.sqrt(np.diag(covar))[1]

            Luv = itg.trapz(bbody(np.arange(1000,wlref[0]),Tuv,Ruv),np.arange(1000,wlref[0]))*sup
            Luv_err = Luv*np.sqrt((2*Ruv_err/Ruv)**2+(4*Tuv_err/Tuv)**2)

            out2.write('%.2f\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\n' %(ph,T1,T1_err,R1,R1_err,Topt,Topt_err,Ropt,Ropt_err))

    #            print '*** Fitting UV and opt-NIR separately'

            plt.figure(2)
            plt.plot(np.arange(3600,25000),bbody(np.arange(3600,25000),Topt,Ropt)-fscale*k,color=cols[filters[k%len(filters)]],linestyle='--',linewidth=1.5)
            plt.plot(np.arange(1000,3600),bbody(np.arange(1000,3600),Tuv,Ruv)-fscale*k,color=cols[filters[k%len(filters)]],linestyle='-.',linewidth=1.5)

        except:
            out2.write('%.2f\t%.2e\t%.2e\t%.2e\t%.2e\n' %(ph,T1,T1_err,R1,R1_err))

    else:
        out2.write('%.2f\t%.2e\t%.2e\t%.2e\t%.2e\n' %(ph,T1,T1_err,R1,R1_err))


    L2 = Luv + itg.trapz(flux,wlref) + Lnir

    L2_err = np.sqrt(L1_err**2 + (Luv_err)**2 + (Lnir_err)**2)

    L2arr.append(L2)
    L2err_arr.append(L2_err)


    out1.write('%.2f\t%.2e\t%.2e\t%.2e\t%.2e\n' %(ph,L1,L1_err,L2,L2_err))



    plt.draw()
    plt.xlabel('Wavelength (Ang)')
    plt.ylabel(r'$\mathit{L}_\lambda$ + constant')
    plt.legend(numpoints=1,ncol=2)

    k += 1


print('\n\n*** Done! Displaying bolometric light curve ***')

L1arr = np.array(L1arr)
L1err_arr = np.array(L1err_arr)
L2arr = np.array(L2arr)
L2err_arr = np.array(L2err_arr)

plt.figure(3,(10,8))
plt.clf()

plt.errorbar(phase,np.log10(L1arr),0.434*L1err_arr/L1arr,fmt='o',color='k',markersize=12,label='Observed flux only')
plt.errorbar(phase,np.log10(L2arr),0.434*L2err_arr/L2arr,fmt='d',color='r',markersize=9,label='Plus BB correction')

plt.xlabel('Days from '+ref+'-band maximum')
plt.ylabel(r'$log_{10} \mathit{L}_{bol}\,(erg\,s^{-1})$')


logout = list(zip(phase,np.log10(L1arr),0.434*L1err_arr/L1arr))
logoutBB = list(zip(phase,np.log10(L2arr),0.434*L2err_arr/L2arr))

#logfile1 = open(outdir+'/logL_obs_'+sn+'_'+filters+'.txt','w')
#logfile1.write('#logL for model fits\n')
#logfile1.write('#Phase\tlogL\terr')
np.savetxt(outdir+'/logL_obs_'+sn+'_'+filters+'.txt',logout,fmt='%.3f',delimiter='\t')
#logfile1.close()

#logfile2 = open(outdir+'/logL_bb_'+sn+'_'+filters+'.txt','w')
#logfile2.write('#logL for model fits (inc BB extrapolation)\n')
#logfile2.write('#Phase\tlogL\terr')
np.savetxt(outdir+'/logL_bb_'+sn+'_'+filters+'.txt',logoutBB,fmt='%.3f',delimiter='\t')
#logfile2.close()


#plt.gca().set_yscale('log')

plt.legend(numpoints=1)

plt.draw()

plt.show()


out1.write('\n#KEY\n#Lobs = integrate observed fluxes with no BB fit\n#L+BB = observed flux + BB fit extrapolation')

out1.close()
out2.close()


fin = input('\n\n> PRESS RETURN TO EXIT...\n')
