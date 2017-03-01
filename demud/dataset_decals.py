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
import pyfits
import numpy as np
import pylab
from dataset import Dataset

class DECaLSData(Dataset):

  def __init__(self, decalsfilename=None):
    """DECaLSData(decalsfilename=None)

    Read in DECaLS catalog data from FITS file.
    """

    Dataset.__init__(self, decalsfilename, "DECaLSData", '')

    self.readin()


  def  readin(self):
    """readin()

    Read in DECaLS table data from FITS file.
    """

    datafile = pyfits.open(self.filename)

    # Read in the desired columns.
    # First the easy ones: we'll index them with [1,2,4] to get G,R,Z bands
    columns = ['DECAM_FLUX',
               'DECAM_ANYMASK',
               'DECAM_FRACFLUX',
               'DECAM_FRACMASKED',
               'DECAM_RCHI2']

    self.data     = datafile[1].data[:].field(columns[0])[:,[1,2,4]]
    self.features = ['%s %s' % (columns[0], b) for b in ['G','R','Z']]

    for c in columns[1:]:
      self.data      = np.hstack([self.data,
                               datafile[1].data[:].field(c)[:,[1,2,4]]])
      self.features += ['%s %s' % (c, b) for b in ['G','R','Z']]
    self.data   = self.data.T  # features x samples
    self.labels = ['%s_%d' % (b,id) for (b,id) in \
                     zip(datafile[1].data[:].field('BRICKNAME'),
                         datafile[1].data[:].field('OBJID'))]

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
  
    pylab.xlabel(self.xlabel)
    pylab.ylabel(self.ylabel)
    pylab.title('DEMUD selection %d (%s), item %d, using K=%d' % \
                (m, label, ind, k))
    pylab.legend(fontsize=10)
    
    padding = 1.19
    pylab.ylim([float(min(min(x), min(r))), max(float(max(max(x), max(r)))
                 * padding, float(max(max(x), max(r))))])
    
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
  

