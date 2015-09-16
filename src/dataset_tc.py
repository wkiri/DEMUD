#!/usr/bin/env python
# File: dataset_tc.py
# Author: Kiri Wagstaff, 7/1/13
#
# Readers and plotters for TextureCam (image) data sets
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

import os, sys, pickle
import numpy as np
import pylab, math
import Image
from dataset import *

################### TEXTURECAM ##############
class TCData(Dataset):
  # Contains code needed to load, plot, and interpret 
  # TextureCam image (.pgm) data file

  def  __init__(self, rawfilename=None, filename=None):
    """TCData(rawfilename="", filename="")

    Read in TextureCam (image) data from (pickled) filename.
    If it doesn't exist, read in the source .ppm file (rawfilename)
    and construct the data set, save it out, and proceed.
    """

    Dataset.__init__(self, filename,
                     'tc-' + \
                       os.path.splitext(os.path.basename(filename))[0], '')
                     #filename[filename.rfind('/')+1:filename.rfind('_')], '')

    if not os.path.exists(filename):
      TCData.read_ppm(rawfilename, filename)
    
    self.readin()


  def  readin(self):
    """readin()
    """

    inf = open(self.filename, 'r')
    (self.data, self.labels, self.width, self.height,
     self.winsize, self.nbins, self.image) = pickle.load(inf)
    self.npixels = self.width * self.height
    inf.close()

    self.xlabel = 'Grayscale intensity'
    self.ylabel = 'Probability'
    self.xvals  = np.arange(self.data.shape[0]).reshape(-1,1)


  @classmethod
  def  read_ppm(cls, rawfilename, filename):
    """read_ppm(rawfilename, filename)

    Read in raw pixel data from rawfilename (.ppm).
    Create a histogram around each pixel to become
    the feature vector for that obsevation (pixel).
    Pickle the result and save it to filename.
    Note: does NOT update object fields.
    Follow this with a call to readin().
    """

    im  = Image.open(rawfilename)
    (width, height) = im.size
    npixels = width * height
    pix = np.array(im)

    # Generate one feature vector (histogram) per pixel
    #winsize = 20  # for test.pgm
    winsize = 100
    #winsize = 0  # for RGB
    halfwin = winsize/2

    nbins   = 101
    bins    = np.linspace(0, 255, nbins)

    # Only use windows that are fully populated
    mywidth  = width-winsize
    myheight = height-winsize
    data    = []
    labels  = []

    # Pick up all windows, stepping by half of the window size
    for y in range(halfwin, height-halfwin, halfwin/2):
      for x in range(halfwin, width-halfwin, halfwin/2):
        # Read in data in row-major order
        ind = (y-halfwin)*mywidth + (x-halfwin)
        #data[:,ind] = \
        #    np.histogram(pix[y-halfwin:y+halfwin,
        #                        x-halfwin:x+halfwin],
        #                        bins)[0]
        # Just RGB
        #data[:,ind] = pix[y,x]
        # RGB window
        #data[:,ind] = pix[y-halfwin:y+halfwin,x-halfwin:x+halfwin].flat
        if data == []:
          data = pix[y-halfwin:y+halfwin,x-halfwin:x+halfwin].reshape(-1,1)
        else:
          data = np.concatenate((data,
                                 pix[y-halfwin:y+halfwin,x-halfwin:x+halfwin].reshape(-1,1)),1)
        labels    += ['(%d,%d)' % (y,x)]

    outf = open(filename, 'w')
    pickle.dump((data, labels, width, height, winsize, nbins, pix), outf)
    outf.close()
    print 'Saved data to %s.' % filename
    

  def  plot_item(self, m, ind, x, r, k, label, U, scores):
    """plot_item(self, m, ind, x, r, k, label, U, scores)

    Plot selection m (index ind, data in x) and its reconstruction r,
    with k and label to annotate the plot.

    U and scores are optional; ignored in this method, used in some
    classes' submethods.
    """

    if x == [] or r == []: 
      print "Error: No data in x and/or r."
      return
  
    im = Image.fromarray(x.reshape(self.winsize, self.winsize, 3))
    outdir  = os.path.join('results', self.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    figfile = os.path.join(outdir, '%s-sel-%d-k-%d.pdf' % (self.name, m, k))
    im.save(figfile)
    print 'Wrote plot to %s' % figfile

    # record the selections in order, at their x,y coords
    # subtract selection number from n so first sels have high values
    mywidth  = self.width - self.winsize
    myheight = self.height - self.winsize
    # set all unselected items to a value 1 less than the latest
    priority = mywidth*myheight - m
    if priority < 2:
      priority = 2
    self.selections[np.where(self.selections < priority)] = priority-2
    (y,x) = map(int, label.strip('()').split(','))
    #self.selections[ind/mywidth, ind%myheight] = priority
    qtrwin = self.winsize/8
    self.selections[y-qtrwin:y+qtrwin, x-qtrwin:x+qtrwin] = priority
    
    pylab.clf()
    pylab.imshow(self.image)
    pylab.hold(True)
    #pylab.imshow(self.selections)
    masked_sels = np.ma.masked_where(self.selections < priority, self.selections)
    pylab.imshow(masked_sels, interpolation='none', alpha=0.5)
    #figfile = '%s/%s-priority-%d-k-%d.pdf' % (outdir, self.name, m, k)
    # Has to be .png or the alpha transparency doesn't work! (pdf)
    figfile = os.path.join(outdir, '%s-priority-k-%d.png' % (self.name, k))
    pylab.savefig(figfile)
    print 'Wrote selection priority plot to %s' % figfile


