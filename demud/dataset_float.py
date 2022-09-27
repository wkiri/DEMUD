#!/usr/bin/env python
# File: dataset_float.py
# Author: Kiri Wagstaff, 9/24/14
#
# Superclass for unlabeled data sets with floating point values:
# readers and plotters
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
import csv, numpy, pylab, math
from dataset import Dataset
from log import printt


class FloatDataset(Dataset):
  # Supersubclass for data sets

  def __init__(self, filename=None, name='Floatdata', initfilename=None):
    """Dataset(filename="", name="", initfilename="") -> Dataset
    
    Creates a new Dataset based on the data in filename
    and with the name given.
    Name is used in output filename prefixes.
    Assumes CSV, floating point values, and no class labels.
    Commented lines (with #) are ignored.

    This top-level class can't do anything itself,
    but needs to be instantiated with one of its subclasses.
    """

    Dataset.__init__(self, filename, name, initfilename)


  @classmethod
  def  read_csv(cls, filename, nskip):
    """read_csv(filename, nskip)

    Read in the contents of the (CSV formatted) file
    and return values to can populate fields of a FloatDataset object.
    nskip specifies how many columns to skip to get to real data.
    """

    data   = []
    labels = []
    xvals  = []
    with open(filename, 'r') as csvfile:
      lines = csvfile.readlines()

      # If there's a header (begins with #), use it to
      # populate the feature names
      if lines[0][0] == '#':
        print('Populating xvals from data file header.')
        header = lines[0][1:].strip()
        #xvals = numpy.array(map(float,header.split(',')))
        xvals = numpy.array(header.split(','))

      for line in lines:
        # Skip over empty or commented lines
        if line.strip() == '' or line[0] == '#':
          continue
        attributes = re.split(',', line.strip())
        # replace '' with NaN
        i = 0
        for i in range(len(attributes)):
          if len(attributes[i]) == 0:
            attributes[i] = 'nan'

        data += [[float(x) for x in attributes[nskip:]]]
        if nskip > 0: # Use the first column as a label
          labels.append(attributes[0])
        else:  # fake labels
          labels.append('None')

    data = numpy.array(data)


    return (xvals, data, labels)


  def  readin(self, nskip):
    """readin()
    """
    
    (self.xvals, self.data, self.labels) = FloatDataset.read_csv(self.filename,
                                                                 nskip)

    self.data = self.data.T  # features x samples

    # If there was no header with feature names, just create an empty xvals
    if self.xvals == []:
      self.xvals = numpy.arange(self.data.shape[0]).reshape(-1,1)

    # Read in the init data file, if present
    if self.initfilename != '':
      printt('Reading initialization data set from %s' % self.initfilename)
      (_, self.initdata, _) = FloatDataset.read_csv(self.initfilename, nskip)
      self.initdata = self.initdata.T  # features x samples


  def  plot_item_triangles(self, m, ind, x, r, k, label, U,
                           rerr, feature_weights, band_ind):
    """plot_item_triangles(self, m, ind, x, r, k, label, U,
       rerr, feature_weights, band_ind)

    Plot selection m (index ind, data in x) with triangles to
    mark the largest residual values on band_ind indices
    (i.e., you need to pick them before calling this).

    To use this, define plot_item() in your data set's class
    to call this function instead.
    """

    if x == [] or r == []: 
      print("Error: No data in x and/or r.")
      return
  
    pylab.clf()
    # xvals, x, and r need to be column vectors
    pylab.plot(self.xvals, r, 'r-',  linewidth=0.5, label='Expected')
    pylab.plot(self.xvals, x, 'b.-', linewidth=1,   label='Observations')
    # Boost font sizes for axis and tick labels
    pylab.xlabel(self.xlabel) #, fontsize=16)
    pylab.ylabel(self.ylabel) #, fontsize=16)
    '''
    pylab.xticks(fontsize=16)
    pylab.yticks(fontsize=16)
    '''
    pylab.title('DEMUD selection %d (%s), item %d, using K=%d' % \
                (m, label, ind, k))
    pylab.legend(fontsize=10)

    # width of triangles to plot
    width = (self.xvals.max() - self.xvals.min()) / 100.0
    
    for band in band_ind:
      w = float(self.xvals[band])
      reproj = r[band]
      # Draw a triangle that points up if r > x
      # or down if r < x
      pylab.fill([w-width, w+width, w],
                 [reproj,  reproj,  x[band]],
                 '0.6', zorder=1)

    outdir  = os.path.join('results', self.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    figfile = os.path.join(outdir, 'sel-%d-k-%d-(%s).pdf' % (m, k, label))
    pylab.savefig(figfile)
    print('Wrote plot to %s' % figfile)
    pylab.close()

    
  def  select_bands(self, x, r, frac_annotate):
    """select_bands(self, x, r, frac_annotate)

    Select which bands to highlight (largest magnitude residual).
    """
    
    res = x - r
    abs_res = numpy.absolute(res)
    mx = abs_res.max()
    mn = abs_res.min()
    print('Absolute residuals: min %2.g, max %.2g.\n' % (mn, mx))
    if mn == mx and mx == 0:
      return

    sorted_abs_res = numpy.sort(abs_res,0)
    num_annotate = int(math.floor(frac_annotate * len(abs_res)))
    thresh = sorted_abs_res[-num_annotate]
    
    print('Marking top %.3f%% of residuals (%d above %.2g).' % \
        (frac_annotate * 100, num_annotate, thresh))

    band_ind = (numpy.where(abs_res >= thresh)[0]).tolist()

    return band_ind


