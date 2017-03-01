#!/usr/bin/env python
# File: dataset_decals.py
# Author: Kiri Wagstaff, 3/1/17
#
# Class for reading in and plotting DECaLS data in FITS format.
# DECaLS: Dark Energy Camera Legacy Survey
# Inspired by Arjun Dey
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

import os, sys, re
import numpy as np
import pyfits, pylab
from dataset import Dataset

class DECaLSData(Dataset):

  def __init__(self, decalsfilename=None):
    """DECaLSData(decalsfilename=None)

    Read in DECaLS catalog data from FITS file.
    """

    Dataset.__init__(self, decalsfilename, "DECaLSData", '')

    self.readin()


  @classmethod
  def  flux_to_magnitude(cls, flux):
    """flux_to_magnitude(cls, flux)

    Convert flux(es) into magnitude(s).

    Examples:

    >>> DECaLSData.flux_to_magnitude(20)
    19.247425010840047

    >>> DECaLSData.flux_to_magnitude(-20)
    -19.247425010840047

    >>> DECaLSData.flux_to_magnitude([20, -20])
    array([ 19.24742501, -19.24742501])
    """

    s = np.sign(flux)

    magnitude = s * (-2.5 * np.log10(np.abs(flux)) + 22.5)

    return magnitude
    

  def  readin(self):
    """readin()

    Read in DECaLS table data from FITS file.
    """

    datafile = pyfits.open(self.filename)

    # Read in the desired columns.
    '''
    # First the easy ones: we'll index them with [1,2,4] to get G,R,Z bands
    columns = ['DECAM_FLUX',
               'DECAM_ANYMASK',
               'DECAM_FRACFLUX',
               'DECAM_FRACMASKED',
               'DECAM_RCHI2']

    # Use the G,R,Z bands for several features
    self.data     = datafile[1].data[:].field(columns[0])[:,[1,2,4]]
    self.features = ['%s %s' % (columns[0], b) for b in ['G','R','Z']]

    for c in columns[1:]:
      self.data      = np.hstack([self.data,
                               datafile[1].data[:].field(c)[:,[1,2,4]]])
      self.features += ['%s %s' % (c, b) for b in ['G','R','Z']]
    '''

    # Compute the color ratios
    model_type = datafile[1].data[:].field('TYPE')
    #model_type_to_use = 'PSF'
    #model_type_to_use = 'SIMP'
    #model_type_to_use = 'EXP'
    model_type_to_use = 'DEV'
    use_data   = datafile[1].data[model_type == model_type_to_use]
    G = use_data.field('DECAM_FLUX')[:,1]
    R = use_data.field('DECAM_FLUX')[:,2]
    Z = use_data.field('DECAM_FLUX')[:,4]
    # ratio G/R
    self.data = DECaLSData.flux_to_magnitude(G) / DECaLSData.flux_to_magnitude(R)
    self.features = ['G/R']
    # ratio R/Z
    self.data = np.vstack([self.data,
                           DECaLSData.flux_to_magnitude(R) / 
                           DECaLSData.flux_to_magnitude(Z)])
    self.features += ['R/Z']

    G = use_data.field('DECAM_APFLUX')[:,1,2]
    R = use_data.field('DECAM_APFLUX')[:,2,2]
    Z = use_data.field('DECAM_APFLUX')[:,4,2]
    # aperture ratio G/R
    self.data = np.vstack([self.data,
                           DECaLSData.flux_to_magnitude(G) / 
                           DECaLSData.flux_to_magnitude(R)])
    self.features += ['AP G/R']
    # aperture ratio R/Z
    self.data = np.vstack([self.data,
                           DECaLSData.flux_to_magnitude(R) / 
                           DECaLSData.flux_to_magnitude(Z)])
    self.features += ['AP R/Z']

    #self.data   = self.data.T  # features x samples
    self.labels = ['%s_%d_%.6f_%.6f' % (b,id,ra,dec) for (b,id,ra,dec) in \
                     zip(use_data.field('BRICKNAME'),
                         use_data.field('OBJID'),
                         use_data.field('RA'),
                         use_data.field('DEC'))]

    datafile.close()

    self.xvals    = np.arange(self.data.shape[0]).reshape(-1,1)
    self.features = np.array(self.features)


  def  plot_item(self, m, ind, x, r, k, label, U, rerr, feature_weights):
    """plot_item(self, m, ind, x, r, k, label, U, rerr, feature_weights)

    Borrowed from UCIDataset.

    Plot selection m (index ind, data in x) and its reconstruction r,
    with k and label to annotate of the plot.

    U and rerr are here ignored.  Could use them to plot a projection
    into the first two PCs' space (see dataset_libs.py).

    If feature_weights are specified, omit any 0-weighted features 
    from the plot.
    """
    
    if x == [] or r == []: 
      print "Error: No data in x and/or r."
      return
   
    # Select the features to plot
    if feature_weights != []:
      goodfeat = [f for f in range(len(feature_weights)) \
                    if feature_weights[f] > 0]
    else:
      goodfeat = range(len(self.xvals))

    # Make a dual bar graph of the original and reconstructed features
    width = 0.35
    offset = (1 - 2*width) / 2
  
    fig = pylab.figure()
    ax = fig.add_subplot(1, 1, 1)

    x = np.array(x)
    
    xvals = [self.xvals[z][0] for z in range(self.xvals.shape[0])]
    x = [x[z] for z in range(x.shape[0])]
    
    bars1 = ax.bar([xvals[i] + offset for i in goodfeat], 
                   x, width, color='b', label='Observations')
    bars2 = ax.bar([xvals[i] + width + offset for i in goodfeat], 
                   r, width, color='r', label='Expected')

    # dashed line to show 0
    pylab.plot([0, len(self.features)], [0, 0], '--')
  
    pylab.xlabel(self.xlabel)
    pylab.ylabel(self.ylabel)
    pylab.title('DEMUD selection %d (%s), item %d, using K=%d' % \
                (m, label, ind, k))
    pylab.legend(fontsize=10)
    
    padding = 1.19
    pylab.ylim([min(0, float(min(min(x), min(r))) * padding),
                max(0, float(max(max(x), max(r))) * padding)])
    
    if len(self.features) == 0:
        pylab.xticks(pylab.arange(len(x)) + width + offset, range(len(x)))
    else:
        pylab.xticks(pylab.arange(len(x)) + width + offset, self.features,
                     rotation=-30, ha='left')
    pylab.tight_layout()
    
    if not os.path.exists('results'):
      os.mkdir('results')
    outdir = os.path.join('results', self.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    figfile = os.path.join(outdir, 'sel-%d-k-%d-(%s).pdf' % (m, k, label))
    pylab.savefig(figfile)
    print 'Wrote plot to %s' % figfile
  

if __name__ == "__main__":
  # Run inline tests    
  import doctest                                                              

  (num_failed, num_tests) = doctest.testmod()
  filename                = os.path.basename(__file__)

  if num_failed == 0:
    print "%-20s All %3d tests passed!" % (filename, num_tests)
  else:
    sys.exit(1)
