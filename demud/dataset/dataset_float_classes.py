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
import csv,  pylab
import numpy as np
from .dataset_float import FloatDataset

###############################################################################
#
#                                FLOAT VALUES
#
###############################################################################
class Floats(FloatDataset):
  # Contains code needed to load, plot, and interpret float data (CSV)

  def  __init__(self, filename=None, initfilename=None):
    """Floats(filename="")

    Read in floating point data in CSV format from filename.
    """

    FloatDataset.__init__(self, filename, "floats", initfilename)
    
    try:
      self.readin(0)
    except:
      print
      print('This class assumes no identifiers in the data, i.e., just numeric values.')
      sys.exit(1)



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
    self.xvals = np.array(['0.432',
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
#                 CNN FEATURE VECTORS (EXPERIMENTAL)
#
###############################################################################
class CNNFeat(FloatDataset):
  # Contains code needed to load, plot, and interpret CNN vector data.
  # Initially copied from APF spectra (CSV) data implementation.

  def  __init__(self, filename=None, initfilename=None):
    """CNNFeat(filename="")

    Read in CNN feature vectors in CSV format from filename.
    """

    FloatDataset.__init__(self, filename, "cnn", initfilename)

    # readin(1) means that the first entry on each line is an item name.  
    # readin(0) means that the first entry on each line is the first feature.
    self.readin(1)

    # Feature names (wavelengths in Angstroms) are in the data file
    # on the first line (starts with #).
    # This is read in by the FloatDataset class.

    self.xlabel = 'x'
    self.ylabel = 'y'

   # Plotting doesn't make sense for this dataset, so export original, 
   # reconstructed, and residual feature vectors to a csv for interpretation.
  def  plot_item(self, m, ind, x, r, k, label, U,
                 rerr, feature_weights):

    # This should probably be somewhere else in the code but
    reload(sys)
    sys.setdefaultencoding("UTF-8")

    # Save out top hits file
    outdir   = os.path.join('results', self.name)
    #hitsfile = os.path.join(outdir, 'hits-%s.txt' % self.name)
    selectfile = os.path.join(outdir, 'select-%s.csv' % self.name)
    reconfile = os.path.join(outdir, 'recon-%s.csv' % self.name)
    residfile = os.path.join(outdir, 'resid-%s.csv' % self.name)

    # Check validity
    if len(x) == 0 or len(r) == 0:
      print("Error: No data in x and/or r.")
      return

    # First item gets to create (and clear) the file 
    if m == 0:
      f = open(selectfile, 'w')
      f.write('')
      f.close()
      f2 = open(reconfile, 'w')
      f2.write('')
      f2.close()
      f3 = open(residfile, 'w')
      f3.write('')
      f3.close()

    # Had ndarrays converting to lists before, not necessary
    inecho = x
    recon = r
    # Residual calculation - element-wise subtraction.
    resid = (x - r)

    # Write original, reconstructed, and residual vectors of selections 
    # into a csv
    with open(selectfile, 'a') as f:
      csvwriter = csv.writer(f, dialect='excel')
      csvwriter.writerow(inecho)

    with open(reconfile, 'a') as f:
      csvwriter = csv.writer(f, dialect='excel')
      csvwriter.writerow(recon)

    with open(residfile, 'a') as f:
      csvwriter = csv.writer(f, dialect='excel')
      csvwriter.writerow(resid)

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


###############################################################################
#
#                                DAN SPECTRA
#
###############################################################################
class DANSpectra(FloatDataset):
  # Contains code needed to load, plot, and interpret PAN spectra (CSV) data.

  def  __init__(self, filename=None):
    """DANSpectra(filename="")

    Read in DAN spectra in CSV format from filename.
    """

    FloatDataset.__init__(self, filename, "dan_spectra")
    
    self.readin(1)
    self.update_features()


  def  update_features(self):

    self.xvals = [5.000, 10.6250, 16.9375, 24.0000, 31.9375,
                  40.8125, 50.7500, 61.8750, 74.3750, 88.4375,
                  104.250, 122.000, 141.938, 164.312, 189.438,
                  217.688, 249.438, 285.125, 325.250, 370.375,
                  421.125, 478.188, 542.375, 614.562, 695.750,
                  787.062, 889.750, 1005.25, 1135.19, 1281.31,
                  1445.69, 1630.56, 1838.50, 2072.38, 2335.44,
                  2631.38, 2964.25, 3338.69, 3759.88, 4233.69,
                  4766.69, 5366.31, 6040.88, 6799.75, 7653.44,
                  8611.94, 9692.31, 10907.7, 12274.9, 13813.1,
                  15543.4, 17490.1, 19680.0, 22143.6, 24915.2,
                  28033.2, 31540.9, 35487.1, 39926.6, 44920.9,
                  50539.4, 56860.3, 63971.2]
    # xvals represent bin end points, so we need to duplicate most of them
    self.xvals = [0] + sorted(self.xvals + self.xvals) + [100000.0]
    self.xvals = np.array([str(x) for x in self.xvals])

    self.xlabel = 'Time (ms)'
    self.ylabel = 'Counts'


  # DAN x values are bins with an extent.  
  # Counts are for the width of the bin
  def  plot_item(self, m, ind, x, r, k, label, U,
                 rerr, feature_weights):

    if len(x) == 0 or len(r) == 0:
      print("Error: No data in x and/or r.")
      return
  
    pylab.clf()
    # xvals, x, and r need to be column vectors
    # xvals represent bin end points, so we need to duplicate most of them
    x = np.repeat(x, 2, axis=0)
    r = np.repeat(r, 2, axis=0)

    pylab.subplot(2,1,1)
    pylab.semilogx(self.xvals, r[0:128], 'r-', label='Expected')
    pylab.semilogx(self.xvals, x[0:128], 'b.-', label='Observations')
    pylab.xlabel('CTN: ' + self.xlabel)
    pylab.ylabel(self.ylabel)
    pylab.legend(loc='upper left', fontsize=10)

    pylab.subplot(2,1,2)
    pylab.semilogx(self.xvals, r[128:], 'r-', label='Expected')
    pylab.semilogx(self.xvals, x[128:], 'b.-', label='Observations')
    pylab.xlabel('CETN: ' + self.xlabel)
    pylab.ylabel(self.ylabel)
    pylab.legend(loc='upper left', fontsize=10)

    pylab.suptitle('DEMUD selection %d (%s), item %d, using K=%d' % \
                (m, label, ind, k))
  
    outdir = os.path.join('results', self.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    figfile = os.path.join(outdir, 'sel-%d-k-%d-(%s).pdf' % (m, k, label))
    pylab.savefig(figfile)
    print('Wrote plot to %s' % figfile)
    pylab.close()


  def plot_pcs(self, m, U, mu, k, S):
    """plot_pcs(m, U, mu, k, S)
    Plot the principal components in U, after DEMUD iteration m, 
        by adding back in the mean in mu.
    Ensure that there are k of them, 
        and list the corresponding singular values from S.
    """

    colors = ['b','g','r','c','m','y','k','#666666','DarkGreen', 'Orange']
    while len(colors) < k: colors.extend(colors)
  
    pylab.clf()

    if m == 0:
      max_num_pcs = k
    else:
      cur_pcs = U.shape[1]
      max_num_pcs = min(min(cur_pcs,k), 4)
  
    umu = np.zeros_like(U)
    for i in range(max_num_pcs):
      umu[:,i] = U[:,i] + mu[:,0] #[i]
      
    for i in range(max_num_pcs):
      vector = umu[:,i]
      if i == 0 and m == 1:
        vector[0] -= 1
      vector = np.repeat(vector, 2, axis=0)
      label = 'PC %d, SV %.2e' % (i, S[i])

      pylab.subplot(2,1,1)
      pylab.semilogx(self.xvals, vector[0:128], color=colors[i], label=label)
      pylab.subplot(2,1,2)
      pylab.semilogx(self.xvals, vector[128:], color=colors[i], label=label)

    pylab.subplot(2,1,1)
    pylab.xlabel('CTN: ' + self.xlabel)
    pylab.ylabel(self.ylabel)
    pylab.legend(loc='upper left', fontsize=10)

    pylab.subplot(2,1,2)
    pylab.xlabel('CETN: ' + self.xlabel)
    pylab.ylabel(self.ylabel)
    pylab.legend(loc='upper left', fontsize=10)
      
    pylab.suptitle('SVD of dataset ' + self.name + ' after selection ' + str(m))
    
    outdir = os.path.join('results', self.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    figfile = os.path.join(outdir, 'PCs-sel-%d-k-%d-(%s).pdf' % (m, k, label))
    pylab.savefig(figfile)
    print('Wrote SVD to %s' % figfile)
    pylab.close()



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
    print("%-20s All %3d tests passed!" % (filename, num_tests))

