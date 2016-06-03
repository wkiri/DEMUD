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
    
    self.readin(0)

    self.update_features()

  def  update_features(self):

    # Feature names (wavelengths in Angstroms) are in the data file
    # on the first line (starts with #)

    with open(self.filename, 'r') as csvfile:
      header = csvfile.readlines()[0]

      if header[0] != '#':
        printt('Error: Did not find wavelength header line (must start with #) in %s.' % self.filename)
        sys.exit(1)
    
      # Strip off the #
      header = header[1:].strip()
      self.xvals = numpy.array(map(float,header.split(',')))

    self.xlabel = 'Wavelength (A)'
    self.ylabel = 'Flux'


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

