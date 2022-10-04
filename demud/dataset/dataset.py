#!/usr/bin/env python
# File: dataset.py
# Author: Kiri Wagstaff, 3/6/13
#
# Readers and plotters for various data sets
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

class Dataset(object):
  # Superclass for data sets

  def __init__(self, filename=None, name=None, initfilename=None):
    """Dataset(filename="", name="") -> Dataset
    
    Creates a new Dataset based on the data in filename
    and with the name given.
    Name is used in output filename prefixes.
    initfilename specifies the file with data to initialize the model.

    This top-level class can't do anything itself,
    but needs to be instantiated with one of its subclasses.
    """

    self.filename = filename
    self.name     = name
    self.initfilename = initfilename
    self.xlabel   = ''
    self.ylabel   = ''
    self.data     = []
    self.initdata = [] # Data set used to initialize the model
    self.labels   = []
    self.features = []
    self.xvals    = []
    self.samples  = []


  def  plot_item(self, m, ind, x, r, k, label, U, scores, feature_weights):
    """plot_item(self, m, ind, x, r, k, label, U, scores, feature_weights)

    Plot selection m (index ind, data in x) and its reconstruction r,
    with k and label to annotate of the plot.

    U, scores, and feature_weights  are optional; ignored in this method, u
    sed in some classes' submethods.
    """
    
    if len(x) == 0 or len(r) == 0:
      print("Error: No data in x and/or r.")
      return
  
    pylab.clf()
    # xvals, x, and r need to be column vectors
    pylab.plot(self.xvals, r, 'r-', label='Expected')
    pylab.plot(self.xvals, x, 'b.-', label='Observations')

    pylab.xlabel(self.xlabel)
    pylab.ylabel(self.ylabel)
    pylab.title('DEMUD selection %d (%s), item %d, using K=%d' % \
                (m, label, ind, k))
    pylab.legend(fontsize=10)
  
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

    #assert (k == U.shape[1])
  
    colors = ['b','g','r','c','m','y','k','#666666','DarkGreen', 'Orange']
    while len(colors) < k: colors.extend(colors)
  
    pylab.clf()

    if m == 0:
      max_num_pcs = k
    else:
      cur_pcs = U.shape[1]
      max_num_pcs = min(min(cur_pcs,k), 4)
  
    umu = numpy.zeros_like(U)
    for i in range(max_num_pcs):
      umu[:,i] = U[:,i] + mu[:,0] #[i]
      
    for i in range(max_num_pcs):
      vector = umu[:,i]
      if i == 0 and m == 1:
        vector[0] -= 1
      label = 'PC %d, SV %.2e' % (i, S[i])
      pylab.plot(self.xvals, vector, color=colors[i], label=label)
      
    pylab.xlabel(self.xlabel)
    pylab.ylabel(self.ylabel)
    pylab.title('SVD of dataset ' + self.name + ' after selection ' + str(m))
    xvals = [self.xvals[z] for z in range(self.xvals.shape[0])]
    diff = pylab.mean([xvals[i] - xvals[i-1] for i in range(1, len(xvals))])
    pylab.xlim([float(xvals[0]) - diff / 6.0, float(xvals[-1]) + diff / 6.0])
    #pylab.xticks(xvals, self.features)
    pylab.legend()
    
    outdir = os.path.join('results', self.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    figfile = os.path.join(outdir, 'PCs-sel-%d-k-%d-(%s).pdf' % (m, k, label))
    pylab.savefig(figfile)
    print('Wrote SVD to %s' % figfile)
    pylab.close()


  # Write a list of the selections in CSV format
  def write_selections_csv(self, i, k, orig_ind, label, ind, scores):
    outdir = os.path.join('results', self.name)
    selfile = os.path.join(outdir, 'selections-k%d.csv' % k)
    # If this is the first selection, open for write
    # to clear out previous run.
    if i == 0:
      fid = open(selfile, 'w')
      # Output a header.  For some data sets, the label is a class;
      # for others it is an object identifier.  To be generic,
      # here we call this 'Name'.
      fid.write('# Selection, Index, Name, Score\n')

      # If scores is empty, the (first) selection was pre-specified,
      # so there are no scores.  Output 0 for this item.
      if len(scores) == 0:
        fid.write('%d,%d,%s,0.0\n' % (i, orig_ind, label))
      else:
        fid.write('%d,%d,%s,%g\n' % (i, orig_ind, label, scores[ind]))
    else:
      fid = open(selfile, 'a')
      fid.write('%d,%d,%s,%g\n' % (i, orig_ind, label, scores[ind]))

    fid.close()

  
  # Write a list of n selections that are similar to selection i (index ind)
  # using scores (with respect to selection i).
  def write_similar_html(self, n, i, k, ind, scores):
    outdir = os.path.join('results', self.name)
    simfile = os.path.join(outdir, 'similar-k%d-i%d.html' % (k, i))

    print("%d: Scores range from %f to %f." % (i, scores.min(), scores.max()))

    # Not clear how to select a threshold for similarity; 
    # no good histogram-analysis methods so far.
    # 1. Otsu's method doesn't seem to work well here; 
    # it assumes a bimodal distribution which isn't necessarily the case.
    # 2. Looking for a 0-valued bin is too brittle.

    # For now, try 1.0.  But maybe this should be scaled by
    # the number of features? max possible score for this data set? etc.?
    best_thresh = 1.0

    # Write out everything that is less than best_thresh
    fid = open(simfile, 'w')
    fid.write('<html><head><title>DEMUD: %s, k=%d</title></head>\n' % (self.name, k))
    fid.write('<body>\n')
    fid.write('<h1>DEMUD experiments on %s with k=%d:<br> Items similar to selection %d (%s)</h1>\n' % (self.name, k, i, self.labels[ind]))
    #fid.write('<p>%d similar items (those with reconstruction error at most %f) of %d total:\n' % (numpy.sum(scores <= best_thresh)-1, best_thresh, len(scores)))
    fid.write('<p>Top %d most similar items:\n' % n)
    fid.write('<ul>\n')

    # Sort objects by increasing score (difference)
    sel_sort = sorted(zip(range(len(scores)), scores), key=lambda s: s[1]) 
    n_printed = 0
    j_ind = 0
    while n_printed < n:
      (j, sc) = sel_sort[j_ind]
      print(j, sc)
      j_ind += 1
      if j == ind: # skip this item
        continue
      fid.write('<li>Item %d (%s): score %f</li>\n' % (j,
                                                       self.labels[j],
                                                       sc))
      n_printed += 1
  
    fid.write('</ul>\n')
    fid.write('</body></html>\n')
    fid.close()

    print('%d similar items (those with reconstruction error at most %f) of %d total.' % (numpy.sum(scores <= best_thresh), best_thresh, len(scores)))

    input()


if __name__ == "__main__":
  import doctest

  (num_failed, num_tests) = doctest.testmod()
  filename                = os.path.basename(__file__)

  if num_failed == 0:
    print("%-20s All %3d tests passed!" % (filename, num_tests))

