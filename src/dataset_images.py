#!/usr/bin/env python
# File: dataset_envi.py
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
import pickle
from dataset import *
from log import printt

################### Image data ##############
class ImageData(Dataset):
  # Contains code needed to load, plot, and interpret image data.

  def  __init__(self, dirname=None):
    """ImageData(dirname="")

    Read in image data from dirname.
    """

    Dataset.__init__(self, dirname,
                     'img-' + os.path.splitext(os.path.basename(dirname))[0], '')

    self.readin()


  def  readin(self):
    """readin()

    Read in image data from a directory.
    """
    
    dirname = self.filename

    data   = []
    labels = []  # Save the individual file names

    (width, height) = (-1, -1)

    # Read in the image data
    files = sorted(os.listdir(dirname))
    for f in files:
      # Unix-style wildcards. 
      if fnmatch.fnmatch(f, '*.jpg'):
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

    data = np.asarray(data)
    print 'Read %d image files with %d pixels each.' % data.shape
    data = data.T
        
    # Labels are individual filenames
    (self.width, self.height, self.data, self.labels) = \
        (width, height, data, labels)

    print ' Dimensions: %d width, %d height.' % (self.width, self.height)





    
