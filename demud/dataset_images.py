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
import numpy as np
from dataset import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import glob
from log import printt

def progbar(current, to, width=40, show=True):
    percent = float(current) / float(to)
    length = int( width * percent)
    if show:
        count = " (%d/%d)    " % (current, to)
    else:
        count = ""
    sys.stdout.write(("\r[" + ("#" * length) + " "*(width-length) + "] %0d" % (percent*100)) + "%" + count)
    sys.stdout.flush()

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
      (self.initdata, unused_labels, imshape) = \
          ImageData.read_image_dir(self.initfilename)
      self.initdata = np.asarray(self.initdata)
      self.initdata = self.initdata.T
      print('Initializing with %d images (%s).' % \
            (self.initdata.shape[1], str(imshape)))
      print(self.initdata.shape)

    ########## Read in the data to analyze
    # Labels are individual filenames
    (self.data, self.labels, self.imshape) = \
        ImageData.read_image_dir(self.filename)
      
    self.data = np.asarray(self.data)
    print(self.data.shape)

    if len(self.data) == 0: 
      print('Error: no image files found.')
      sys.exit(1)

    self.data = self.data.T
    print('Read %d images (%s).' % \
          (self.data.shape[1], str(self.imshape)))


  def  plot_item(self, m, ind, x, r, k, label, U, scores, feature_weights):
    """
    plot_item(self, m, ind, x, r, k, label, U, scores, feature_weights):
    
    Plot selection m (index ind, data in x) and its reconstruction r,
    with k and label to annotate of the plot.

    Also show the residual.

    U, scores, and feature_weights are optional; ignored in this method, 
    used in some classes' submethods.
    """
    print("Plotting...")
    if x == [] or r == []: 
      print("Error: No data in x and/or r.")
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
    
    if len(self.imshape) == 2:
      im = pylab.imshow(np.uint8(x.reshape(self.imshape)), cmap='gray')
    else:
      im = pylab.imshow(np.uint8(x.reshape(self.imshape)))
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
    # Clip reconstruction
    r[r>255] = 255
    r[r<0]   = 0
    if len(self.imshape) == 2:
      im = pylab.imshow(np.uint8(r.reshape(self.imshape)), cmap='gray')
    else:
      im = pylab.imshow(np.uint8(r.reshape(self.imshape)))
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
    
    if m > 0:
      # THIRD SUBPLOT: residual data
    
      pylab.subplot(2,2,3)
      resid = x - r
    
      if len(self.imshape) == 2:
        # Grayscale: plot so high values are red and low are blue.
        # Tweak vmin and vmax so 0 is always in the middle (white)
        absmax = max(abs(vmin), abs(vmax))
        im = pylab.imshow(resid.reshape(self.imshape),
                          cmap=cmap, vmin=-absmax, vmax=absmax) 
      else: 
        # Color: plot actual differences.
        # Scale to fill the range -127 to +127
        # Shift so 0 is at 127,127,127.
        minres = np.min(resid)
        maxres = np.max(resid)
        range_val  = max(abs(minres), maxres)
        im = pylab.imshow(np.uint8(resid*127./range_val+127).reshape(self.imshape))

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

    pylab.suptitle('DEMUD selection %d (%s), item %d, using K=%d' % \
                   (m, label, ind, k))
    
    outdir = os.path.join('results', self.name)
    if not os.path.exists('results'):
      os.mkdir('results')
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    figfile = os.path.join(outdir, 'sel-%d-k-%d.pdf' % (m, k))
    plt.savefig(figfile, bbox_inches='tight', pad_inches=0.1)
    plt.cla()
    plt.clf()
    print("done.")
    plt.close()
    pylab.close()
    #print('Wrote plot to %s' % figfile)
    

  def plot_pcs(self, m, U, mu, k, S):
    """plot_pcs(m, U, mu, k, S)
    Plot the principal components in U, after DEMUD iteration m, 
        by adding back in the mean in mu.
    Ensure that there are k of them, 
        and list the corresponding singular values from S.
    """

    #assert (k == U.shape[1])
  
    cur_pcs = U.shape[1]
    max_num_pcs = min(min(cur_pcs,k), 9)

    pylab.figure()
    pylab.subplots_adjust(wspace=0.1, left=0)

    # Display each image in its own subplot
    for i in range(max_num_pcs):
      pylab.subplot(3,3,i+1)

      #im = pylab.imshow((U[:,i] + mu[:,0]).reshape((self.width,
      im = pylab.imshow(np.uint8(U[:,i].reshape(self.imshape)))
#                        cmap='gray' if len(self.imshape) == 2 else 'rgb')
      pylab.tick_params(\
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        left='off',      # ticks along the left edge are off
        right='off',      # ticks along the right edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off', # labels along the bottom edge are off
        labelleft='off')   # labels along the left edge are off?
      pylab.xlabel('PC %d' % i)
  
    shortname = self.name[:self.name.find('-k=')]
    pylab.suptitle('SVD of dataset ' + shortname + 
                   ' after selection ' + str(m))
    
    outdir = os.path.join('results', self.name)
    if not os.path.exists('results'):
      os.mkdir('results')
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    figfile = os.path.join(outdir, 'PCs-sel-%d-k-%d.pdf' % (m, k))
    pylab.savefig(figfile)
    print('Wrote SVD to %s' % figfile)


  @classmethod
  def  read_image_dir(cls, dirname):
    """read_image_dir(dirname)

    Read in all of the images in dirname and return
    - a list of data
    - a list of labels
    - imshape: (width, height) or (width, height, depth) tuple
    """

    data   = []
    labels = []  # Save the individual file names

    imshape = (-1, -1, -1)

    # Read in the image data
    files = sorted(os.listdir(dirname))
    numimages = len(os.listdir(dirname))
    print(numimages)
    printt("Loading files:")
    counter = 0
    for idx,f in enumerate(files):
      # Unix-style wildcards. 
      if (fnmatch.fnmatch(f, '*.jpg') or
          fnmatch.fnmatch(f, '*.JPG') or
          fnmatch.fnmatch(f, '*.png')):
        # Read in the image
        filename = os.path.join(dirname, f)
        im = imread(filename)

        if imshape[0] == -1:
          #data = np.zeros([], dtype=np.float32).reshape(numimages, np.prod(im.shape))
          data = np.zeros([numimages, np.prod(im.shape)], dtype=np.float32)
          #data = np.array([], dtype=np.float32).reshape(0,np.prod(im.shape))
          imshape = im.shape
        else:
          # Ensure that all images are the same dimensions
          if imshape != im.shape:
            if len(im.shape) == 2:
              # Convert grayscale to rgb
              im = np.dstack((im, im, im))
            else:
              raise ValueError('Images must all have the same dimensions.')

        #data = np.vstack([data, im.reshape(1,np.prod(im.shape))])
        data[counter] = im.reshape(1, np.prod(im.shape))

        labels.append(f)
        progbar(idx, len(files))
        counter += 1

    return (data, labels, imshape)



    
