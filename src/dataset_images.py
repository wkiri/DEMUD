#!/usr/bin/env python
# File: dataset_images.py
# Author: Kiri Wagstaff, 5/7/13
#
# Readers and plotters for image data sets.
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

import os, sys, fnmatch
from PIL import Image
import numpy as np
from dataset import *
import matplotlib
import matplotlib.pyplot as plt
from log import printt

################### Image data ##############
class ImageData(Dataset):
  # Contains code needed to load, plot, and interpret image data.

  def  __init__(self, dirname=None, initdirname=''):
    """ImageData(dirname="")

    Read in image data from dirname.

    Optionally, specify a directory in initdirname that contains
    data to initialize the model with.
    """

    Dataset.__init__(self, dirname,
                     'img-' + os.path.splitext(os.path.basename(dirname))[0], 
                     initdirname)

    self.readin()


  def  readin(self):
    """readin()

    Read in image data from a directory.
    """
    
    # Read in the initialization data (images) from initdirname, if present.
    # This variable is called 'initfilename', but it's a directory here.
    if self.initfilename != '':
      printt('Reading initialization data set from %s' % self.initfilename)
      (self.initdata, unused_labels, width, height) = ImageData.read_image_dir(self.initfilename)
      self.initdata = np.asarray(self.initdata)
      self.initdata = self.initdata.T
      print self.initdata.shape

    ########## Read in the data to analyze
    # Labels are individual filenames
    (self.data, self.labels, self.width, self.height) = \
        ImageData.read_image_dir(self.filename)
      
    self.data = np.asarray(self.data)
    print 'Read %d image files with %d pixels each.' % self.data.shape
    self.data = self.data.T
    print self.data.shape
        
    print ' Dimensions: %d width, %d height.' % (self.width, self.height)


  def  plot_item(self, m, ind, x, r, k, label, U, scores, feature_weights):
    """
    plot_item(self, m, ind, x, r, k, label, U, scores, feature_weights):
    
    Plot selection m (index ind, data in x) and its reconstruction r,
    with k and label to annotate of the plot.

    Also show the residual.

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

    # FIRST SUBPLOT: original image
    pylab.subplot(2,2,1)
    
    im = pylab.imshow(x.reshape((self.width,
                                 self.height)),
                      cmap='gray')
    pylab.tick_params(\
      axis='both',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom='off',      # ticks along the bottom edge are off
      left='off',      # ticks along the left edge are off
      right='off',      # ticks along the right edge are off
      top='off',         # ticks along the top edge are off
      labelbottom='off', # labels along the bottom edge are off
      labelleft='off')   # labels along the left edge are off?
    pylab.xlabel('Image')

    # SECOND SUBPLOT: reconstructed data

    pylab.subplot(2,2,2)
    im = pylab.imshow(r.reshape((self.width,
                                 self.height)),
                      cmap='gray')
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
    
    im = pylab.imshow(resid.reshape((self.width,
                                     self.height)),
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

    # Voodoo required to get colorbar to be the right height.
#    div = make_axes_locatable(pylab.gca())
#    cax = div.append_axes('right','5%',pad='3%')
#    cbar = pylab.colorbar(im, cax=cax)
    cbar = pylab.colorbar(im)
    #tickvals = np.arange(-1,1.1,0.5)
    #cbar.set_ticks(tickvals)
    #cbar.set_ticklabels(tickvals)

    pylab.suptitle('DEMUD selection %d (%s), item %d, using K=%d' % \
                   (m, label, ind, k))
    
    outdir = os.path.join('results', self.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    figfile = os.path.join(outdir, '%s-sel-%d-k-%d.pdf' % (self.name, m, k))
    plt.savefig(figfile, bbox_inches='tight', pad_inches=0.1)
    #print 'Wrote plot to %s' % figfile
    

  @classmethod
  def  read_image_dir(cls, dirname):
    """read_image_dir(dirname)

    Read in all of the images in dirname and return
    - a list of data
    - a list of labels
    - image width
    - image height
    """

    data   = []
    labels = []  # Save the individual file names

    (width, height) = (-1, -1)

    # Read in the image data
    files = sorted(os.listdir(dirname))
    for f in files:
      # Unix-style wildcards. 
      if (fnmatch.fnmatch(f, '*.jpg') or
          fnmatch.fnmatch(f, '*.png')):
        # Read in the image
        filename = dirname + '/' + f
        im = Image.open(filename)

        if width == -1:
          (width, height) = im.size
        else:
          # Ensure that all images are the same dimensions
          (w, h) = im.size
          if w != width or h != height:
            raise ValueError('Images must all have the same dimensions.')
        
        data.append(list(im.getdata()))

        labels.append(f)

        # Close the file
        im.close()

    return (data, labels, width, height)



    
