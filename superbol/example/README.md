# Superbol input data

This directory contains real supernova data demonstrating the input format for superbol.

For *all* Superbol input: 
- Sloan, PanSTARRS, Gaia, ATLAS, GALEX in AB mags; 
- Johnson, NIR and Swift in Vega mags

# Example 1: SN2015bn (Nicholl et al. 2016, ApJ, 826, 39)

- Input has multiple filters per file
- Already in rest-frame absolute magnitudes (no cosmological corrections required)
- Covers UV-NIR (blackbody correction negligible if run with all filters)
- Play with effects of leaving out different filters to see which are most important and test accuracy of blackbody corrections

# Example 2: Gaia16apd (Nicholl et al. 2017, ApJL, 835, 8)

- One filter per input file
- MJD, apparent magnitudes (only extinction correction applied)
- Some mismatch in timing of different filters: useful to experiment with interpolations/extrapolations
- Can run multiple times with different reference bands to stitch together light curve
- No NIR so BB correction important
