#!/usr/bin/env python
# File: dataset_uci.py
# Author: James Bedell, 8/21/13
#
# Superclass for UCI data sets: readers and plotters
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
import csv, numpy, pylab
from dataset import Dataset

class UCIDataset(Dataset):
  # Supersubclass for data sets

  def __init__(self, filename=None, name='UCI dataset'):
    """Dataset(filename="", name="") -> Dataset
    
    Creates a new Dataset based on the data in filename
    and with the name given.
    Name is used in output filename prefixes.

    This top-level class can't do anything itself,
    but needs to be instantiated with one of its subclasses.
    """

    Dataset.__init__(self, filename, name, '')

  def  readin(self, nskip):
    """readin()
    """

    with open(self.filename, 'r') as csvfile:
      lines = csvfile.readlines()
      for line in lines:
        # Skip over empty lines
        if line.strip() == '' or line[0] == '#':
          continue
        attributes = re.split(r',|\s+', line.strip())

        self.data += [[float(x) for x in attributes[nskip:-1]]]
        self.samples.append(attributes[0])
        self.labels.append(attributes[-1])

    self.data = numpy.array(self.data)

    self.data   = self.data.T  # features x samples
    
    self.xvals  = numpy.arange(self.data.shape[0]).reshape(-1,1)
  
  def  plot_item(self, m, ind, x, r, k, label, U, rerr, feature_weights):
    """plot_item(self, m, ind, x, r, k, label, U, rerr, feature_weights)

    Plot selection m (index ind, data in x) and its reconstruction r,
    with k and label to annotate of the plot.

    U and rerr are here ignored.  Could use them to plot a projection
    into the first two PCs' space (see dataset_libs.py).

    If feature_weights are specified, omit any 0-weighted features 
    from the plot.
    """
    
    if len(x) == 0 or len(r) == 0:
      print("Error: No data in x and/or r.")
      return
   
    # Select the features to plot
    if len(feature_weights) > 0:
      goodfeat = [f for f in range(len(feature_weights)) \
                    if feature_weights[f] > 0]
    else:
      goodfeat = list(range(len(self.xvals)))

    # Make a dual bar graph of the original and reconstructed features
    width = 0.35
    offset = (1 - 2*width) // 2
  
    fig = pylab.figure()
    ax = fig.add_subplot(1, 1, 1)

    x = numpy.array(x)
    
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
    pylab.legend() #fontsize=10)
    
    #padding = 1.19
    #pylab.ylim([float(min(min(x), min(r))), max(float(max(max(x), max(r)))
    #             * padding, float(max(max(x), max(r))))])
    
    if len(self.features) == 0:
        pylab.xticks(pylab.arange(len(x)) + width + offset, list(range(len(x))))
    else:
        pylab.xticks(pylab.arange(len(x)) + width + offset, self.features)
    
    if not os.path.exists('results'):
      os.mkdir('results')
    outdir = os.path.join('results', self.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    figfile = os.path.join(outdir, 'sel-%d-k-%d-(%s).pdf' % (m, k, label))
    pylab.savefig(figfile)
    print('Wrote plot to %s' % figfile)
    pylab.close()
  
    



