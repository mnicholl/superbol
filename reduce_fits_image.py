#!/usr/bin/env python

import astropy
from astropy.io import fits
import ccdproc
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os
import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter('ignore', category=AstropyWarning)

###############

parser = argparse.ArgumentParser()

parser.add_argument('-e', dest='fits_extension', default=1, type=int,
                    help='Specify fits extension for data (default=1).')

parser.add_argument('-i', dest='file_to_reduce', default='', nargs='+',
                    help='List of files to reduce. Accepts wildcards or '
                    'space-delimited list.')


args = parser.parse_args()

files_to_red = [i for i in args.file_to_reduce]

AMP = args.fits_extension     # Set this to the fits extension with the data!

###############

outdir = 'reduced_'+str(len(glob.glob('reduced_*')))

if not os.path.exists(outdir): os.makedirs(outdir)


all_files = ccdproc.ImageFileCollection('.', keywords=['imagetyp','filter'])

if os.path.exists('bias.fits'):
    print('\n> Bias found\n')

    bias = ccdproc.CCDData.read('bias.fits',unit='adu')
else:
    try:
        biaslist = all_files.files_filtered(imagetyp='zero')

        biasproc = []

        print('\n> Making bias\n')

        for i in biaslist:
            ccd = ccdproc.CCDData.read(i, hdu=AMP, unit = 'adu')
            # ccd = ccdproc.subtract_overscan(ccd, median=True,
            #                                 fits_section=ccd.header['BIASSEC'])
            ccd = ccdproc.trim_image(ccd, fits_section=ccd.header['TRIMSEC'] )
            biasproc.append(ccd)


        bias = ccdproc.combine(biasproc, method='average', weights=None, scale=None,
                                clip_extrema=True, nlow=1, nhigh=1)

        bias.write('bias.fits', clobber=True)
    except:
        print('\n> No biases found! Exiting.')
        sys.exit()

fig = plt.figure(1,(8,8))

plt.ion()

plt.show()


for image in files_to_red:

    print('\n>>> '+image+':\n')

    plt.clf()

    filt = fits.getval(image,'filter')

    print('\n> Filter: '+filt+'\n')

    flatlist = all_files.files_filtered(imagetyp='flat', filter=filt)

    if os.path.exists('flat'+filt+'.fits'):
        print('\n> Flat found\n')

        flat = ccdproc.CCDData.read('flat'+filt+'.fits',unit='adu')
    else:
        try:
            print('\n> Making flat\n')

            flatproc = []

            for i in flatlist:
                ccd = ccdproc.CCDData.read(i, hdu=AMP, unit = 'adu')
                ccd = ccdproc.trim_image(ccd, fits_section=ccd.header['TRIMSEC'] )
                ccd = ccdproc.subtract_bias(ccd, bias)
                flatproc.append(ccd)


            flat = ccdproc.combine(flatproc, method='average', weights=None, scale=None,
                                    sigma_clip=True, sigma_clip_low_thresh=3,
                                    sigma_clip_high_thresh=3)

            flat.write('flat'+filt+'.fits', clobber=True)
        except:
            print('\n> No suitable flats - skipping image! \n')
            continue


    print('\n> Reducing image\n')

    hdu = fits.open(image)

    obs = ccdproc.CCDData(hdu[AMP].data, header=hdu[0].header+hdu[AMP].header,
                    unit = 'adu')

    obs = ccdproc.trim_image(obs, fits_section=obs.header['TRIMSEC'] )
    obs = ccdproc.subtract_bias(obs, bias)

    obs = ccdproc.flat_correct(obs, flat)

    out = obs.to_hdu()

    out[0].header = obs.header

    out.writeto(os.path.join(outdir,filt+'_redu_'+image), output_verify='ignore')

    plt.imshow(obs,vmin=np.mean(obs)-np.std(obs)/2,
                vmax=np.mean(obs)+np.std(obs)/2,
                cmap='gray',origin='lower')

    plt.draw()

    plt.show()

    print('\n> Done\n')


print('\n>>> Finished\n')
