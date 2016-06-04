#!/usr/bin/env python
# File: dataset_gbtfil.py
# Author: Kiri Wagstaff, 6/3/16
#
# Readers and plotters for GBT Filterbank data sets
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

import os, fnmatch, sys, pickle
import numpy as np
import pylab
from scipy.stats import nanmean
from dataset import Dataset
import matplotlib
import datetime

############# GBT Filterbank ##############
class GBTFilterbankData(Dataset):
  # Contains code needed to load, plot, and interpret GBT filterbank data.

  def  __init__(self, gbtdirname=None, catalogfile=None):
    """GBTFilterbankData(gbtdirname=None, catalogfile=None)
    
    Read in GBT filterbank data by reference to the catalog file.
    """

    Dataset.__init__(self, gbtdirname, "gbtfil", '')

    self.read_gbt_dir(gbtdirname, catalogfile)

    self.xlabel = 'Frequency'
    self.ylabel = 'Time'
    #self.xvals  = np.arange(self.data.shape[1]).reshape(-1,1)


  def  read_gbt_dir(self, gbtdirname, catalogfile):
    """read_gbt_dir(gbtdirname, catalogfile)

    Read in GBT filterbank data from .npy files in gbtdirname,
    using catalogfile to decide which ones to use.
    """

    self.data = []
    self.nfreq = 0
    self.ntime = 0
    i = 0

    # Open the catalog file
    with open(catalogfile, 'r') as f:
      lines = f.readlines()

      for line in lines:
        # Skip comments
        if line[0] == '#':
          continue

        (fname, freq, ra, dec) = line.split()
        freq = float(freq)

        # Restrict analysis to events close to 1420 MHz
        if np.abs(freq - 1420) < 2:
          d = np.load(os.path.join(gbtdirname, fname))

          # If it's the first item, set data dimensions
          # Skip the first row of frequencies.
          if self.data == []:
            (self.ntime, self.nfreq) = d[1:,:].shape
          
          # First row is frequencies.  This is different
          # for each data item, so we'll ignore it for now.
          data_item = d[1:,:]  # freq (across) x time (down)
          
          self.data.append(data_item.reshape(-1,1))

          # Label is the first part of the filename
          self.labels.append('_'.join(fname.split('_')[0:2]))

    self.data = np.asarray(self.data).squeeze().T
    print self.data.shape
    
    print
    # Data is now d x n, where d = #freq x #times and n = #files
    print 'Read data set with %d features, %d files.' % self.data.shape


  def  plot_item(self, m, ind, x, r, k, label, U, scores, feature_weights):
    """
    plot_item(self, m, ind, x, r, k, label, U, scores, feature_weights):
    
    Plot selection m (index ind, data in x) and its reconstruction r,
    with k and label to annotate of the plot.

    Also show the residual and the RGB visualization of the selected image.

    U, scores, and feature_weights are optional; ignored in this method,
    used in some classes' submethods.
    """
    
    if x == [] or r == []: 
      print "Error: No data in x and/or r."
      return

    vmin = min(np.nanmin(x), np.nanmin(r))
    vmax = max(np.nanmax(x), np.nanmax(r))

    # Create my own color map; middle is neutral/gray, high is red, low is blue.
    cdict = {
      'red':   ((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)),
      'green': ((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)),
      'blue':  ((0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0))
      }
    cmap = matplotlib.colors.LinearSegmentedColormap('res', cdict, 256)
   
    matplotlib.rc('axes', edgecolor = 'w')
   
    pylab.figure()
    pylab.subplots_adjust(wspace=0.1,left=0)

    # FIRST SUBPLOT: filterbank data
    
    pylab.subplot(2,2,1)
    im = pylab.imshow(x.reshape((self.nfreq,
                                 self.ntime)),
                      vmin=vmin, vmax=vmax)
    pylab.tick_params(\
      axis='both',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom='off',      # ticks along the bottom edge are off
      left='off',      # ticks along the left edge are off
      right='off',      # ticks along the right edge are off
      top='off',         # ticks along the top edge are off
      labelbottom='off', # labels along the bottom edge are off
      labelleft='off')   # labels along the left edge are off?
    pylab.xlabel('Original Data')

    # Voodoo required to get colorbar to be the right height.
    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    #div = make_axes_locatable(pylab.gca())
    #cax = div.append_axes('right','5%',pad='3%')
    #pylab.colorbar(im, cax=cax)
    pylab.colorbar(im)

    # SECOND SUBPLOT: reconstructed data

    pylab.subplot(2,2,2)
    im = pylab.imshow(r.reshape((self.nfreq,
                                 self.ntime)),
                      vmin=vmin, vmax=vmax)
    pylab.tick_params(\
      axis='both',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom='off',      # ticks along the bottom edge are off
      left='off',      # ticks along the left edge are off
      right='off',      # ticks along the right edge are off
      top='off',         # ticks along the top edge are off
      labelbottom='off', # labels along the bottom edge are off
      labelleft='off')   # labels along the left edge are off?
    pylab.xlabel('Reconstructed Data')
  #  div = make_axes_locatable(pylab.gca())
  #  cax = div.append_axes('right','5%',pad='3%')
  #  pylab.colorbar(im, cax=cax)
    pylab.colorbar(im)
    
    # THIRD SUBPLOT: residual data
    
    pylab.subplot(2,2,3)
    resid = x - r
    #print "Residual Min: %5.3f, Avg: %5.3f, Max: %5.3f" % (np.nanmin(resid),
    #                                                       nanmean(resid),
    #                                                       np.nanmax(resid))
    
    im = pylab.imshow(resid.reshape((self.nfreq,
                                     self.ntime)),
                      cmap=cmap) #, vmin=-1, vmax=1) 
    pylab.tick_params(\
      axis='both',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom='off',      # ticks along the bottom edge are off
      left='off',      # ticks along the left edge are off
      right='off',      # ticks along the right edge are off
      top='off',         # ticks along the top edge are off
      labelbottom='off', # labels along the bottom edge are off
      labelleft='off')   # labels along the left edge are off?
    pylab.xlabel('Residual')

    cbar = pylab.colorbar(im)
    '''
    tickvals = np.arange(-1,1.1,0.5)
    cbar.set_ticks(tickvals)
    cbar.set_ticklabels(tickvals)
    '''

    pylab.suptitle('DEMUD selection %d (%s), item %d, using K=%d' % \
                   (m, label, ind, k))
    
    outdir = os.path.join('results', self.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    figfile = os.path.join(outdir, '%s-sel-%d-k-%d.pdf' % (self.name, m, k))
    pylab.savefig(figfile, bbox_inches='tight', pad_inches=0.1)
    
