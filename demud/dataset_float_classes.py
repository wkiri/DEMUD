#!/usr/bin/env python
# File: dataset_uci_classes.py
# Author: James Bedell, 8/20/13
#
# Specific details for floating point, unlabeled, CSV data sets
#
# Copyright 2013-2015, by the California Institute of Technology. ALL
# RIGHTS RESERVED.  United States Government Sponsorship
# acknowledged. Any commercial use must be negotiated with the Office
# of Technology Transfer at the California Institute of Technology.
#
# This software may be subject to U.S. export control laws and
# regulations.  By accepting this document, the user agrees to comply
# with all applicable U.S. export laws and regulations.  User has the
# responsibility to obtain export licenses, or other export authority
# as may be required before exporting such information to foreign
# countries or providing access to foreign persons.

import os, sys
import csv, numpy, pylab
from dataset_float import FloatDataset

###############################################################################
#
#                                PANCAM SPECTRA
#
###############################################################################
class PancamSpectra(FloatDataset):
  # Contains code needed to load, plot, and interpret Pancam spectra (CSV) data.

  def  __init__(self, filename=None):
    """PancamSpectra(filename="")

    Read in Pancam spectra in CSV format from filename.
    """

    FloatDataset.__init__(self, filename, "pancam_spectra")
    
    self.readin(1)
    self.update_features()

  def  update_features(self):

    # Feature names are in the file, but default reader can't read them
    # so add manually here (microns)
    self.xvals = numpy.array(['0.432',
                              '0.482',
                              '0.535',
                              '0.601',
                              '0.673',
                              '0.754',
                              '0.803',
                              '0.864',
                              '0.904',
                              '0.934',
                              '1.009'])

    self.xlabel = 'Wavelength (um)'
    self.ylabel = 'Reflectance'


###############################################################################
#
#                 AUTOMATED PLANET FINDER (APF) SPECTRA
#
###############################################################################
class APFSpectra(FloatDataset):
  # Contains code needed to load, plot, and interpret APF spectra (CSV) data.

  def  __init__(self, filename=None):
    """APFSpectra(filename="")

    Read in APF spectra in CSV format from filename.
    """

    FloatDataset.__init__(self, filename, "apf_spectra")

    # readin(1) means that the first entry on each line is an item name.  
    # readin(0) means that the first entry on each line is the first feature.
    self.readin(1)

    # Feature names (wavelengths in Angstroms) are in the data file
    # on the first line (starts with #).
    # This is read in by the FloatDataset class.

    self.xlabel = 'Wavelength (A)'
    self.ylabel = 'Flux'


  def  plot_item(self, m, ind, x, r, k, label, U,
                 rerr, feature_weights):

    # Select which residuals to highlight
    frac_annotate = 0.004  # top 0.4%, modify to change how many display
    band_ind = self.select_bands(x, r, frac_annotate)

    # Call the plot_item_triangles() method from dataset_float.py
    self.plot_item_triangles(m, ind, x, r, k, label, U,
                             rerr, feature_weights, band_ind)

    # Save out top hits file
    outdir   = os.path.join('results', self.name)
    hitsfile = os.path.join(outdir, 'hits-%s.txt' % self.name)
    # First item gets to create (and clear) the file
    if m == 0:
      with open(hitsfile, 'w') as f:
        f.close()

    # Write out a line for this selection
    with open(hitsfile, 'a') as f:
      # Write out the name/label of the selected item
      f.write(label)
      # Write out a comma-separated list of selected wavelengths
      for band in band_ind:
        f.write(',%f' % float(self.xvals[band]))
      f.write('\n')


###############################################################################
#
#              GBT SPECTRA (FILTERBANK DATA INTEGRATED OVER TIME)
#
###############################################################################
class GBTSpectra(FloatDataset):
  # Contains code needed to load, plot, and interpret GBT spectra (CSV) data.

  def  __init__(self, filename=None):
    """GBTSpectra(filename="")

    Read in GBT spectra in CSV format from filename.
    """

    FloatDataset.__init__(self, filename, "gbt_spectra")

    # readin(1) means that the first entry on each line is an item name.  
    # readin(0) means that the first entry on each line is the first feature.
    self.readin(1)

    # Feature names (frequencies in MHz) are in the data file
    # on the first line (starts with #).
    # This is read in by the FloatDataset class.

    self.xlabel = 'Frequency (MHz)'
    self.ylabel = 'Flux'


  # This is currently identical to the version in APFSpectra,
  # but can be specialized for different outputs or
  # different frac_annotate.
  def  plot_item(self, m, ind, x, r, k, label, U,
                 rerr, feature_weights):

    # Select which residuals to highlight
    frac_annotate = 0.004  # top 0.4%, modify to change how many display
    band_ind = self.select_bands(x, r, frac_annotate)

    # Call the plot_item_triangles() method from dataset_float.py
    self.plot_item_triangles(m, ind, x, r, k, label, U,
                             rerr, feature_weights, band_ind)

    # Save out top hits file
    outdir   = os.path.join('results', self.name)
    hitsfile = os.path.join(outdir, 'hits-%s.txt' % self.name)
    # First item gets to create (and clear) the file
    if m == 0:
      with open(hitsfile, 'w') as f:
        f.close()

    # Write out a line for this selection
    with open(hitsfile, 'a') as f:
      # Write out the name/label of the selected item
      f.write(label)
      # Write out a comma-separated list of selected wavelengths
      for band in band_ind:
        f.write(',%f' % float(self.xvals[band]))
      f.write('\n')


################################################################################
#
#                                    MAIN
#
################################################################################
if __name__ == "__main__":
  import doctest

  (num_failed, num_tests) = doctest.testmod()
  filename                = os.path.basename(__file__)

  if num_failed == 0:
    print "%-20s All %3d tests passed!" % (filename, num_tests)

