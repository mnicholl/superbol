#!/usr/bin/env python

version = '0.1'

'''
    PSF: PHOTOMETRY SANS FRUSTRATION

    Written by Matt Nicholl, 2015

    Requirements:

    Needs pyraf, pyfits, numpy, and matplotlib (tested with Ureka and
    Anaconda python installations)

    Currently only available in python 2 for iraf compatibility

    Run in directory with image files in FITS format.

    Use -i flag to specify image names separated by spaces. If no names given,
    assumes you want to do photometry on every image in directory.

    File with SN coordinates must exist in same directory or parent
    directory, and should be named *_coords.txt
    Format:
    RA_value  DEC_value

    File with local sequence star coordinates (J2000) and magnitudes
    must exist in same directory or parent directory, and should be
    named *_seq.txt
    Format:
    RA_value DEC_value  MAGBAND1_value    MAGBAND2_value    ...

    Given this list of field star magnitudes and coordinates, PSF.py will
    compute the zeropoint of the image, construct a point spread function
    from these stars, fit this to the target of interest, show the resulting
    subtraction, and return the apparent magnitude from both PSF and aperture
    photometry

    Run with python psf.py <flags> (see help message with psf.py --help)

    Outputs:

    PSF_phot_X.txt (where X is an integer that increases every time code is run,
        to avoid overwriting results)
        This is the primary output of PSF
        Format of text file is:
        image  filter  mjd  PSFmag  err  APmag  err  comments
        - Row exists for each input image used in run
        - PSFmag is from PSF fitting, APmag is from simple aperture photometry
        using aperture size specified with --ap (default 10 pixel radius)
        - comment allows user to specify if e.g. PSF fit looked unreliable
    PSF_output_X/ (where X is same integer as for text file) has additional
        info useful for double-checking results
        For each input file, directory contains:
        - IMAGENAME_psf_stars.txt : list of stars used to build PSF
            Number corresponds to position in input file (_seq.txt)
        - IMAGENAME_seqMags.txt : instrumental magnitudes of sequence stars
        - zeropoints.txt : instrumental zeropoint for each input image
        - zp_list.txt : zeropoints inferred from all sequence stars
            (only for last image processed)
        - IMAGENAME_SN_ap.txt : aperture instrumental mag of target
        - IMAGENAME_SN_dao.txt : Daophot PSF-fitting instrumental mag of target
        - IMAGENAME_psf.fits : Postage stamp image of PSF from iraf task seepsf
        - IMAGENAME.psf.?.fits : Full results of PSF fits. May be more than
            one per image if multiple iterations of PSF fitting were needed
        - IMAGENAME.mag.? : Iraf phot logs for sequence stars and for target

    NEED TO FULLY DOCUMENT

    '''

import numpy as np
import glob
from pyraf import iraf
from iraf import daophot
try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits
import sys
import shutil
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D
import argparse
from matplotlib.patches import Circle

####### Parameters to vary if things aren't going well: #####################
#
# aprad = 10      # aperture radius for phot
# nPSF = 10              # number of PSF stars to try
# recen_rad_stars = 10     # recentering radius: increase if centroid not found; decrease if wrong centroid found!
# recen_rad_sn = 5
# varOrd = 0             # PSF model order: -1 = pure analytic, 2 = high-order empirical corrections, 0/1 intermediate
# sigClip = 1         # Reject sequence stars if calculated ZP differs by this may sigma from the mean
#
#############


parser = argparse.ArgumentParser()

parser.add_argument('--ap', dest='aprad', default=10, type=int,
                    help='Radius for aperture/PSF phot.')

parser.add_argument('--sky', dest='skyrad', default=10, type=int,
                    help='Width of annulus for sky background.')

parser.add_argument('--npsf', dest='nPSF', default=10, type=int,
                    help='Number of PSF stars.')

parser.add_argument('--re', dest='recen_rad_stars', default=10, type=int,
                    help='Radius for recentering on stars.')

parser.add_argument('--resn', dest='recen_rad_sn', default=5, type=int,
                    help='Radius for recentering on SN.')

parser.add_argument('--var', dest='varOrd', default=0, type=int,
                    help='Order for PSF model.')

parser.add_argument('--sig', dest='sigClip', default=1, type=int,
                    help='Sigma clipping for rejecting sequence stars.')

parser.add_argument('--ims','-i', dest='file_to_reduce', default='', nargs='+',
                    help='List of files to reduce. Accepts wildcards or '
                    'space-delimited list.')

parser.add_argument('--high', dest='z2', default=1, type=float,
                    help='Colour scaling for zoomed images; upper bound is '
                    'this value times the standard deviation of the counts.')

parser.add_argument('--low', dest='z1', default=1, type=float,
                    help='Colour scaling for zoomed images; lower bound is '
                    'this value times the standard deviation of the counts.')

parser.add_argument('--keepsub', dest='keep_sub', default=False, action='store_true',
                    help='Do not delete residual images during clean-up ')


args = parser.parse_args()


aprad = args.aprad
skyrad = args.skyrad
nPSF = args.nPSF
recen_rad_stars = args.recen_rad_stars
recen_rad_sn = args.recen_rad_sn
varOrd = args.varOrd
sigClip = args.sigClip
z1 = args.z1
z2 = args.z2

ims = [i for i in args.file_to_reduce]


if len(ims) == 0:
    ims = glob.glob('*.fits')


##################################################

iraf.centerpars.calgo='centroid'
iraf.centerpars.cmaxiter=50
iraf.fitskypars.salgo='centroid'
iraf.fitskypars.annulus=aprad
iraf.fitskypars.dannulus=skyrad
iraf.photpars.apertures=aprad
iraf.photpars.zmag=0
iraf.datapars.sigma='INDEF'
iraf.datapars.datamin='INDEF'
iraf.datapars.datamax='INDEF'

daophot.daopars.function='gauss'
daophot.daopars.psfrad=aprad
daophot.daopars.fitrad=10
daophot.daopars.matchrad=3
daophot.daopars.sannu=aprad
daophot.daopars.wsann=skyrad
daophot.daopars.varorder=varOrd
daophot.datapars.datamin='INDEF'
daophot.datapars.datamax='INDEF'
daophot.daopars.recenter='yes'
daophot.daopars.groupsky='yes'
daophot.daopars.fitsky='yes'


##################################################



# Try to match header keyword to a known filter automatically:

filtSyn = {'u':['u','SDSS-U','up','up1','U640','F336W','Sloan_u','u_Sloan'],
           'g':['g','SDSS-G','gp','gp1','g782','F475W','g.00000','Sloan_g','g_Sloan'],
           'r':['r','SDSS-R','rp','rp1','r784','F625W','r.00000','Sloan_r','r_Sloan'],
           'i':['i','SDSS-I','ip','ip1','i705','F775W','i.00000','Sloan_i','i_Sloan'],
           'z':['z','SDSS-Z','zp','zp1','z623','zs', 'F850LP','z.00000','Sloan_z','z_Sloan'],
           'J':['J'],
           'H':['H'],
           'K':['K','Ks'],
           'U':['U','U_32363A'],
           'B':['B','B_39692','BH'],
           'V':['V','V_36330','VH'],
           'R':['R','R_30508'],
           'I':['I','I_36283']}

# UPDATE WITH QUIRKS OF MORE TELESCOPES...

filtAll = 'ugrizUBVRIJHK'



for i in glob.glob('*.mag.*'):
    os.remove(i)

for i in glob.glob('*.als.*'):
    os.remove(i)

for i in glob.glob('*.arj.*'):
    os.remove(i)

for i in glob.glob('*.sub.*'):
    os.remove(i)

for i in glob.glob('*.pst.*'):
    os.remove(i)

for i in glob.glob('*psf.*'):
    os.remove(i)

for i in glob.glob('*.psg.*'):
    os.remove(i)

for i in glob.glob('*_seqMags.txt'):
    os.remove(i)

for i in glob.glob('*pix*fits'):
    os.remove(i)

for i in glob.glob('*_psf_stars.txt'):
    os.remove(i)



print('#################################################\n#                                               #\n#  Welcome to PSF: Photometry Sans Frustration  #\n#                    (V'+version+')                     #\n#        Written by Matt Nicholl (2015)         #\n#                                               #\n#################################################')



outdir = 'PSF_output_'+str(len(glob.glob('PSF_phot_*')))

if not os.path.exists(outdir): os.makedirs(outdir)


outFile = open('PSF_phot_'+str(len(glob.glob('PSF_phot_*')))+'.txt','w')

ZPfile = open(os.path.join(outdir,'zeropoints.txt'),'w')


outFile.write('#image\tfilter\tmjd\tPSFmag\terr\tAPmag\terr\tcomments')



plt.figure(1,(15,8))

plt.ion()

plt.show()



################################
# Part one: get sequence stars
################################


suggSn = glob.glob('*coords.txt')

if len(suggSn)==0:
    suggSn = glob.glob('../*coords.txt')

if len(suggSn)==0:
    suggSn = glob.glob('../../*coords.txt')

if len(suggSn)>0:
    snFile = suggSn[0]
else:
    sys.exit('Error: no SN coordinates (*_coords.txt) found')

print '\n####################\n\nSN coordinates found: '+snFile

#RAdec = np.genfromtxt(snFile)



suggSeq = glob.glob('*seq.txt')

if len(suggSeq)==0:
    suggSeq = glob.glob('../*seq.txt')

if len(suggSeq)==0:
    suggSeq = glob.glob('../../*seq.txt')

if len(suggSeq)>0:
    seqFile = suggSeq[0]
else:
    sys.exit('Error: no sequence stars (*_seq.txt) found')

print '\n####################\n\nSequence star magnitudes found: '+seqFile


seqDat = np.genfromtxt(seqFile,skip_header=1)

seqHead = np.genfromtxt(seqFile,skip_footer=len(seqDat),dtype=str)

np.savetxt('coords',seqDat[:,:2],fmt='%e',delimiter='  ')

seqMags = {}

for i in range(len(seqHead)-2):
    seqMags[seqHead[i+2]] = seqDat[:,i+2]


#### BEGIN LOOP OVER IMAGES ####


x_sh_1 = 0
y_sh_1 = 0

for image in ims:

    comment1 = ''

    iraf.centerpars.cbox=recen_rad_stars



################################
# Part two: image header info
################################



    print '\nFile: ' + image
    try:
        filtername = pyfits.getval(image,'FILTER')
    except:
        try:
            filtername = pyfits.getval(image,'FILTER1')
            if filtername == ('air' or 'none' or 'clear'):
                filtername = pyfits.getval(image,'FILTER2')
            if filtername == ('air' or 'none' or 'clear'):
                filtername = pyfits.getval(image,'FILTER3')
        except:
            try:
                filtername = pyfits.getval(image,'NCFLTNM2')
            except:
                filtername = 'none'
    print 'Filter found in header: ' + filtername
    if filtername=='none':
        filtername = raw_input('Please enter filter ('+filtAll+') ')

    for j in filtSyn:
        if filtername in filtSyn[j]:
            filtername = j
            print 'Standard filter = ' + filtername

    if filtername not in filtAll:
        filtername = raw_input('Please enter filter ('+filtAll+') ')




    try:
        mjd = pyfits.getval(image,'MJD')
    except:
        try:
            mjd = pyfits.getval(image,'MJD-OBS')
        except:
            try:
                jd = pyfits.getval(image,'JD')
                mjd = jd - 2400000
            except:
                mjd = 99999

        mjd = float(mjd)



#################################
# Part three: do some photometry
#################################



    print '\n#########\n'+filtername+'-band\n#########\n'



    plt.clf()

    plt.subplots_adjust(left=0.05,right=0.99,top=0.99,bottom=-0.05)

########## plot data

    ax1 = plt.subplot2grid((2,4),(0,0),colspan=2,rowspan=2)

    try:
        im = pyfits.open(image)

        im[0].verify('fix')

        data = im[0].data

        header = im[0].header

        ax1.imshow(data, origin='lower',cmap='gray',
                            vmin=np.median(data[len(data)/2/2:3*len(data)/2/2,
                                len(data)/2/2:3*len(data)/2/2])-
                                np.std(data[len(data)/2/2:3*len(data)/2/2,
                                len(data)/2/2:3*len(data)/2/2])*0.5,
                            vmax=np.median(data[len(data)/2/2:3*len(data)/2/2,
                                len(data)/2/2:3*len(data)/2/2])+
                                np.std(data[len(data)/2/2:3*len(data)/2/2,
                                len(data)/2/2:3*len(data)/2/2]))

    except:
        im = pyfits.open(image)

        im[1].verify('fix')

        data = im[1].data

        header = im[1].header

        ax1.imshow(data, origin='lower',cmap='gray',
                            vmin=np.mean(data[len(data)/2/2:3*len(data)/2/2,
                                len(data)/2/2:3*len(data)/2/2])-
                                np.std(data[len(data)/2/2:3*len(data)/2/2,
                                len(data)/2/2:3*len(data)/2/2])*0.5,
                            vmax=np.mean(data[len(data)/2/2:3*len(data)/2/2,
                                len(data)/2/2:3*len(data)/2/2])+
                                np.std(data[len(data)/2/2:3*len(data)/2/2,
                                len(data)/2/2:3*len(data)/2/2]))


    ax1.set_title(image)


    ax1.set_xlim(0,len(data))

    ax1.set_ylim(0,len(data))

    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)

    plt.draw()

    iraf.wcsctran(input='coords',output=image+'_pix.fits',image=image,
                    inwcs='world',outwcs='logical')

    co = np.genfromtxt(image+'_pix.fits')

    ax1.errorbar(co[:,0],co[:,1],fmt='o',mfc='none',markeredgewidth=3,
                    markersize=20,label='Sequence stars')


    for j in range(len(co)):
       ax1.text(co[j,0]+20,co[j,1]-20,str(j+1),color='cornflowerblue')



    iraf.wcsctran(input=snFile,output=image+'_SNpix.fits',image=image,
                    inwcs='world',outwcs='logical')

    SNco = np.genfromtxt(image+'_SNpix.fits')


    ax1.errorbar(SNco[0],SNco[1],fmt='o',markeredgecolor='r',mfc='none',
                    markeredgewidth=5,markersize=25)

    ax1.text(SNco[0]+30,SNco[1]+30,'SN',color='r')



########### Centering

    # Manual shifts:
    x_sh = raw_input('\n> Add approx pixel shift in x? ['+str(x_sh_1)+']  ')
    if not x_sh: x_sh = x_sh_1
    x_sh = int(x_sh)
    x_sh_1 = x_sh

    y_sh = raw_input('\n> Add approx pixel shift in y? ['+str(y_sh_1)+']  ')
    if not y_sh: y_sh = y_sh_1
    y_sh = int(y_sh)
    y_sh_1 = y_sh

    shutil.copy(image+'_pix.fits',image+'_orig_pix.fits')

    pix_coords = np.genfromtxt(image+'_pix.fits')
    pix_coords[:,0] += x_sh
    pix_coords[:,1] += y_sh

    np.savetxt(image+'_pix.fits',pix_coords)

    # recenter on seq stars and generate star list for daophot:
    iraf.phot(image=image,coords=image+'_pix.fits',output='default',
                    interactive='no',verify='no',wcsin='logical',verbose='yes',
                    Stdout='phot_out.txt')


    cent = np.genfromtxt('phot_out.txt')

    ax1.errorbar(cent[:,1],cent[:,2],fmt='s',mfc='none',markeredgecolor='b',
                    markersize=10,markeredgewidth=1.5,label='Recentered',
                    zorder=6)


    orig_co = np.genfromtxt(image+'_orig_pix.fits')

    del_x = np.mean(cent[:,1]-orig_co[:,0])
    del_y = np.mean(cent[:,2]-orig_co[:,1])


########### Build PSF

    daophot.pstselect(image=image,photfile='default',pstfile='default',
                        maxnpsf=nPSF,verify='no')

    ax2 = plt.subplot2grid((2,4),(0,2))


    happy = 'n'
    j = 1
    rmStar = 999
    while happy!='y':

        daophot.pselect(infiles=image+'.pst.'+str(j),
                        outfiles=image+'.pst.'+str(j+1),
                        expr='ID!='+str(rmStar))

        daophot.psf(image=image,photfile='default',pstfile='default',
                    psfimage='default',opstfile='default',groupfil='default',
                    verify='no',interactive='no')

        iraf.txdump(textfile=image+'.pst.'+str(j+1),fields='ID',expr='yes',
                    Stdout=os.path.join(outdir,image+'_psf_stars.txt'))

        psfList = np.genfromtxt(os.path.join(outdir,image+'_psf_stars.txt'),
                                dtype='int')

        for k in psfList:
            ax1.errorbar(co[k-1,0],co[k-1,1],fmt='*',mfc='none',
                            markeredgecolor='lime',markeredgewidth=2,
                            markersize=30,label='Used in PSF fit')

        # This doesn't remove stars from plot - need to give some thought!!!

        handles, labels = ax1.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(),
                    numpoints=1,prop={'size':16}, handletextpad=0.5,
                    labelspacing=0.5, borderaxespad=1, ncol=3,
                    bbox_to_anchor=(1, 1.2))

        daophot.seepsf(psfimage=image+'.psf.'+str(j)+'.fits',
                        image=os.path.join(outdir,image+'_psf.fits'))

        psfIm = pyfits.open(os.path.join(outdir,image+'_psf.fits'))

        psfIm[0].verify('fix')

        psf = psfIm[0].data


#        plt.clf()


        ax2.imshow(psf, origin='lower',cmap='gray',
                    vmin=np.mean(psf)-np.std(psf)*1.,
                    vmax=np.mean(psf)+np.std(psf)*3.)

        ax2.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)

        ax2.set_title('PSF')


        ax3 = plt.subplot2grid((2,4),(0,3),projection='3d')

        tmpArr = range(len(psf))

        X, Y = np.meshgrid(tmpArr,tmpArr)

        ax3.plot_surface(X,Y,psf,rstride=1,cstride=1,cmap='hot',alpha=0.5)

        ax3.set_zlim(np.min(psf),np.max(psf)*1.1)

        ax3.set_axis_off()

        plt.draw()


        happy = raw_input('\n> Happy with PSF? [y]')

        if not happy: happy = 'y'

        if happy != 'y':
            rmStar = raw_input('Star to remove: ')
            j+=1
            os.remove(os.path.join(outdir,image+'_psf.fits'))


    daophot.allstar(image=image,photfile='default',psfimage='default',
                    allstarfile='default',rejfile='default',subimage='default',
                    verify='no',verbose='yes',fitsky='yes')

    sub1 = pyfits.open(image+'.sub.1.fits')

    sub1[0].verify('fix')

    sub0 = sub1[0].data

    ax1.imshow(sub0, origin='lower',cmap='gray',
                vmin=np.mean(data[len(data)/2/2:3*len(data)/2/2,
                    len(data)/2/2:3*len(data)/2/2])-
                    np.std(data[len(data)/2/2:3*len(data)/2/2,
                    len(data)/2/2:3*len(data)/2/2])*0.5,
                vmax=np.mean(data[len(data)/2/2:3*len(data)/2/2,
                    len(data)/2/2:3*len(data)/2/2])+
                    np.std(data[len(data)/2/2:3*len(data)/2/2,
                    len(data)/2/2:3*len(data)/2/2]))


########## Zero point from seq stars

    if filtername in seqMags:

        iraf.txdump(textfile=image+'.als.1',fields='ID,MAG,MERR',expr='yes',
                    Stdout=os.path.join(outdir,image+'_seqMags.txt'))

        seqIm1 = np.genfromtxt(os.path.join(outdir,image+'_seqMags.txt'))
        seqIm1 = seqIm1[seqIm1[:,0].argsort()]
        mask = np.array(seqIm1[:,0],dtype=int)[~np.isnan(seqIm1[:,1])]-1
        seqIm = seqIm1[:,1]
        seqErr = seqIm1[:,2]

        zpList1 = seqMags[filtername][mask]-seqIm

        zp1 = np.mean(zpList1)
        errzp1 = np.std(zpList1)

        print 'Initial zeropoint =  %.3f +/- %.3f\n' %(zp1,errzp1)

        np.savetxt(os.path.join(outdir,'zp_list.txt'),zip(mask+1,zpList1),
        fmt='%d\t%.3f')

        checkMags = np.abs(seqIm+zp1-seqMags[filtername][mask])<errzp1*sigClip

        print 'Rejecting stars from ZP: '
        print seqIm1[:,0][~checkMags]
        print 'ZPs:'
        print seqMags[filtername][mask][~checkMags]-seqIm[~checkMags]

        zpList = seqMags[filtername][mask][checkMags]-seqIm[checkMags]

        ZP = np.mean(zpList)
        errZP = np.std(zpList)

        print 'Zeropoint = %.3f +/- %.3f\n' %(ZP,errZP)

        ZPfile.write(image+'\t'+filtername+'\t%.2f\t%.2f\t%.2f\n' %(mjd,ZP,errZP))




########### SN photometry


    SNco[0] += del_x
    SNco[1] += del_y

    iraf.centerpars.cbox = recen_rad_sn

    recen_rad_1 = recen_rad_sn

    np.savetxt(image+'_SNpix_sh.fits',SNco.reshape(1,2),fmt='%.6f')

    recen = 'y'
    j = 1
    while recen!='n':

        iraf.phot(image=image,coords=image+'_SNpix_sh.fits',output='default',
                    interactive='no',verify='no',wcsin='logical',verbose='yes')

        daophot.allstar(image=image,photfile=image+'.mag.'+str(j+1),
                        psfimage='default',allstarfile='default',
                        rejfile='default',subimage='default',verify='no',
                        verbose='yes',fitsky='yes',recenter='no')

        sub1 = pyfits.open(image+'.sub.'+str(j+1)+'.fits')

        sub1[0].verify('fix')

        sub = sub1[0].data

        ax4 = plt.subplot2grid((2,4),(1,2))


        ax4.imshow(data, origin='lower',cmap='gray',
                            vmin=np.median(data[len(data)/2/2:3*len(data)/2/2,
                                len(data)/2/2:3*len(data)/2/2])-
                                np.std(data[len(data)/2/2:3*len(data)/2/2,
                                len(data)/2/2:3*len(data)/2/2])*0.5,
                            vmax=np.median(data[len(data)/2/2:3*len(data)/2/2,
                                len(data)/2/2:3*len(data)/2/2])+
                                np.std(data[len(data)/2/2:3*len(data)/2/2,
                                len(data)/2/2:3*len(data)/2/2]))


        ax4.set_xlim(SNco[0]-(aprad+skyrad),SNco[0]+(aprad+skyrad))
        ax4.set_ylim(SNco[1]-(aprad+skyrad),SNco[1]+(aprad+skyrad))

        ax4.get_yaxis().set_visible(False)
        ax4.get_xaxis().set_visible(False)

        ax4.set_title('Supernova')

        apcircle = Circle((SNco[0], SNco[1]), aprad, facecolor='none',
                edgecolor='r', linewidth=3, alpha=1)
        ax4.add_patch(apcircle)

        skycircle = Circle((SNco[0], SNco[1]), aprad, facecolor='none',
                edgecolor='r', linewidth=3, alpha=1)
        ax4.add_patch(skycircle)


        ax5 = plt.subplot2grid((2,4),(1,3))

        ax5.imshow(sub, origin='lower',cmap='gray',
                            vmin=np.median(data[len(data)/2/2:3*len(data)/2/2,
                                len(data)/2/2:3*len(data)/2/2])-
                                np.std(data[len(data)/2/2:3*len(data)/2/2,
                                len(data)/2/2:3*len(data)/2/2])*0.5,
                            vmax=np.median(data[len(data)/2/2:3*len(data)/2/2,
                                len(data)/2/2:3*len(data)/2/2])+
                                np.std(data[len(data)/2/2:3*len(data)/2/2,
                                len(data)/2/2:3*len(data)/2/2]))


        ax5.set_xlim(SNco[0]-(aprad+skyrad),SNco[0]+(aprad+skyrad))
        ax5.set_ylim(SNco[1]-(aprad+skyrad),SNco[1]+(aprad+skyrad))

        ax5.get_yaxis().set_visible(False)
        ax5.get_xaxis().set_visible(False)

        ax5.set_title('Subtracted image')

        apcircle = Circle((SNco[0], SNco[1]), aprad, facecolor='none',
                edgecolor='r', linewidth=3, alpha=1)
        ax5.add_patch(apcircle)

        skycircle = Circle((SNco[0], SNco[1]), aprad, facecolor='none',
                edgecolor='r', linewidth=3, alpha=1)
        ax5.add_patch(skycircle)

        plt.draw()


        recen = raw_input('\n> Adjust recentering radius? [n]  ')

        if not recen: recen = 'n'

        if recen!= 'n':
            recen_rad = raw_input('\n> Enter radius ['+str(recen_rad_1)+']  ')
            if not recen_rad: recen_rad = recen_rad_1
            recen_rad = int(recen_rad)
            iraf.centerpars.cbox = recen_rad
            recen_rad_1 = recen_rad
            j += 1



    iraf.txdump(textfile=image+'.mag.'+str(j+1),fields='MAG,MERR',expr='yes',
                Stdout=os.path.join(outdir,image+'_SN_ap.txt'))

    iraf.txdump(textfile=image+'.als.'+str(j+1),fields='MAG,MERR',expr='yes',
                Stdout=os.path.join(outdir,image+'_SN_dao.txt'))


    apmag = np.genfromtxt(os.path.join(outdir,image+'_SN_ap.txt'))

    SNap = apmag[0]
    errSNap = apmag[1]

    daomag = np.genfromtxt(os.path.join(outdir,image+'_SN_dao.txt'))

    try:
        SNdao = daomag[0]
        errSNdao = daomag[1]
    except:
        print 'PSF could not be fit, using aperture mag'
        SNdao = np.nan
        errSNdao = np.nan


    print '\n'

    if filtername in seqMags:

        calMagsDao = SNdao + ZP

        errMagDao = np.sqrt(errSNdao**2 + errZP**2)


        calMagsAp = SNap + ZP

        errMagAp = np.sqrt(errSNap**2 + errZP**2)

    else:
        calMagsDao = SNdao

        errMagDao = errSNdao


        calMagsAp = SNap

        errMagAp = errSNap

        comment1 = 'No ZP - instrumental mag only'

        print '> No ZP - instrumental mag only!!!'



    print '> PSF mag = '+'%.2f +/- %.2f' %(calMagsDao,errMagDao)
    print '> Aperture mag = '+'%.2f +/- %.2f' %(calMagsAp,errMagAp)


    comment = raw_input('\n> Add comment to output file: ')

    if comment1:
        comment += (' // '+comment1)

    outFile.write('\n'+image+'\t'+filtername+'\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t'
                    %(mjd,calMagsDao,errMagDao,calMagsAp,errMagAp))
    outFile.write(comment)


outFile.close()

ZPfile.close()


os.remove('phot_out.txt')


for i in glob.glob('*.mag.*'):
    shutil.copy(i,outdir)

for i in glob.glob('*.psf.*'):
    shutil.copy(i,outdir)

# for i in glob.glob('*.sub.*'):
#     shutil.copy(i,outdir)

for i in glob.glob('*.mag.*'):
    os.remove(i)

for i in glob.glob('*.psf.*'):
    os.remove(i)

if not args.keep_sub:
    for i in glob.glob('*.sub.*'):
        os.remove(i)


for i in glob.glob('*.als.*'):
    os.remove(i)

for i in glob.glob('*.arj.*'):
    os.remove(i)

for i in glob.glob('*.pst.*'):
    os.remove(i)

for i in glob.glob('*.psg.*'):
    os.remove(i)

for i in glob.glob('*pix*fits'):
    os.remove(i)

for i in glob.glob('*_psf_stars.txt'):
    os.remove(i)

for i in glob.glob('coords'):
    os.remove(i)


print '\n##########################################\nFinished!\nCalibrated PSF phot saved to ./PSF_phot.txt\nAperture photometry saved to ./ap_phot.txt\nCheck PSF_output/ for additional info\n##########################################'
