#!/usr/bin/env python
# File: dataset_des.py
# Author: Kiri Wagstaff, 8/28/17
#
# Class for reading in DES data in FITS format.
# DES: Dark Energy Survey
# Collaboration with Tim Eifler and Eric Huff
#
# Copyright 2017, by the California Institute of Technology. ALL
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
import pylab
from dataset import Dataset

class DESData(Dataset):

  def __init__(self, desfilename=None):
    """DESData(desfilename=None)

    Read in DES catalog data.
    """

    Dataset.__init__(self, desfilename, "DESData", '')
    
    self.readin()


  def  readin(self):
    """readin()

    Read in DES catalog data from FITS or .npy files.
    """
    
    if self.filename.endswith('.fits'):
      # Assumes Science Verification data
      self.read_SV_fits()
    elif self.filename.endswith('.npz'): 
      # Assumes DES Y3 Gold data
      self.read_Y3_2_2_npz()
    else: 
      print('Unrecognized file type: ' + self.filename) 
  

  # Read (filtered) DES Y3 2.2 gold data set
  def read_Y3_2_2_npz(self): 
    
    d = np.load(self.filename)
    data = d['data']
    # Testing
    #data = data[:10,:]

    feat_names = list(d['features'])

    # Features to use
    #self.features = ['lup_r', 'color_g_minus_r',
    #                 'color_i_minus_r', 'color_z_minus_r']
    self.features = ['color_g_minus_r', 'lup_r', 
                     'color_i_minus_r', 'color_z_minus_r']
    feat_inds = [feat_names.index(f) for f in self.features]
    self.data = data[:,feat_inds]
    # Trrrrranspose for DEMUD (feat x items)
    self.data = self.data.T

    # Scale some features as needed
    for f in self.features:
      if f == 'lup_r':
        # Subtract the mean value
        mean_lup_r = np.mean(self.data[self.features.index(f),:])
        self.data[self.features.index(f),:] -= mean_lup_r
        print 'Subtracting mean (%.2f) from %s.' % (mean_lup_r, f)
        newf = 'lup_r_minus_mean'
        self.features[self.features.index(f)] = newf
        f = newf
      '''
      if 'MAG' in f: # subtract the min
        minval = np.min(self.data[self.features.index(f),:])
        self.data[self.features.index(f),:] -= minval
        print 'Subtracting %f from %s.' % (minval, f)
        newf = f + '-sub%.2f' % minval
        self.features[self.features.index(f)] = newf
        f = newf
      '''
      print '%s range: ' % f,
      print self.data[self.features.index(f),:].min(),
      print self.data[self.features.index(f),:].max()

    # Also store errors for reporting in explanation plots
    # (when available)
    '''
    self.expl_features = ['']
    expl_feat_inds = [feat_names.index(f) for f in self.expl_features]
    self.expl_data = data[:,expl_feat_inds]
    '''

    # Labels
    self.labels = ['%s_%.6f_%.6f' % (id, ra, dec) for (id, ra, dec) in 
                   zip(data[:,0], data[:,1], data[:,2])]
    self.xvals    = np.arange(self.data.shape[0]).reshape(-1,1)
    self.features = np.array(self.features)
    

  # Read DES Y3 2.0 gold data set
  def read_Y3_2_0_npy(self): 
    
    data = np.load(self.filename)
    
    # We want R, G-R, I-R, Z-R
    self.data = data[3,:]
    self.features = ['MAG_R']

    # G-R
    self.data = np.vstack([self.data, data[2,:] - data[3,:]])
    self.features += ['G-R']

    # I-R
    self.data = np.vstack([self.data, data[4,:] - data[3,:]])
    self.features += ['I-R']

    # Z-R
    self.data = np.vstack([self.data, data[5,:] - data[3,:]])
    self.features += ['Z-R']
    
    # Filter out bogus MAG_R values
    # TODO: remove this with new version of data file (already filtered)
    keep = self.data[0,:] > -1
    self.data = self.data[:,keep]
    data = data[:,keep]
    print('Removed MAG_R values <= -1.')

    # Data is d x n
    print self.data.shape
    # Scale some features as needed
    for f in self.features:
      '''
      if 'MAG' in f: # subtract the min
        minval = np.min(self.data[self.features.index(f),:])
        self.data[self.features.index(f),:] -= minval
        print 'Subtracting %f from %s.' % (minval, f)
        newf = f + '-sub%.2f' % minval
        self.features[self.features.index(f)] = newf
        f = newf
      '''
      print '%s range: ' % f,
      print self.data[self.features.index(f),:].min(),
      print self.data[self.features.index(f),:].max()

    # TODO: add/readin id
    self.labels = ['None_%.6f_%.6f' % (ra, dec) for (ra, dec) in 
                   zip(data[0,:], data[1,:])]
    self.xvals    = np.arange(self.data.shape[0]).reshape(-1,1)
    self.features = np.array(self.features)


  # Read Science Verification data set
  def read_SV_fits(self):
    import fitsio

    subset_photoz_bin = False

    if subset_photoz_bin:
      # Subset to a single photo-z bin
      photoz_bin = 0

      Dataset.__init__(self, desfilename, "DESData_colordiff_bin" + 
                       str(photoz_bin), '')

    data = fitsio.read(self.filename)

    # Mask out the bad objects
    SVA1_FLAG_mask   = (data['SVA1_FLAG'] == 0)
    NGMIX_FLAG_mask  = (data['NGMIX_FLAG'] == 0)
    PHOTOZ_FLAG_mask = (data['PHOTOZ_BIN'] > -1)
    data = data[SVA1_FLAG_mask & 
                NGMIX_FLAG_mask &
                PHOTOZ_FLAG_mask]

    # Read in the desired columns.

    # from SVA_GOLD
    # WLINFO filtered by Umaa to omit objects with
    # SVA1_FLAG != 0
    # NGMIX_FLAG != 0
    # PHOTOZ_BIN != -1

    # We want R, G-R, R-I, I-Z
    self.data = data['MAG_AUTO_R']
    self.features = ['MAG_AUTO_R']

    # G-R
    self.data = np.vstack([self.data,
                           data['MAG_AUTO_G'] - data['MAG_AUTO_R']])
    self.features += ['G-R']

    # R-I
    self.data = np.vstack([self.data,
                           data['MAG_AUTO_R'] - data['MAG_AUTO_I']])
    self.features += ['R-I']

    # I-Z
    self.data = np.vstack([self.data,
                           data['MAG_AUTO_I'] - data['MAG_AUTO_Z']])
    self.features += ['I-Z']

    # MEAN_PHOTOZ
    self.data = np.vstack([self.data,
                           data['MEAN_PHOTOZ']])
    self.features += ['MEAN_PHOTOZ']

    # PHOTOZ_BIN
    self.data = np.vstack([self.data,
                           data['PHOTOZ_BIN']])
    self.features += ['PHOTOZ_BIN']

    # Data is d x n
    print self.data.shape
    # Scale some features as needed
    for f in self.features:
      if 'MAG_AUTO' in f: # subtract the min
        minval = np.min(self.data[self.features.index(f),:])
        self.data[self.features.index(f),:] -= minval
        print 'Subtracting %f from %s.' % (minval, f)
        newf = f + '-sub%.2f' % minval
        self.features[self.features.index(f)] = newf
        f = newf
      print '%s range: ' % f,
      print self.data[self.features.index(f),:].min(),
      print self.data[self.features.index(f),:].max()

    self.labels = ['%d_%.6f_%.6f' % (id,ra,dec) for (id,ra,dec) in \
                     zip(data['COADD_OBJECTS_ID'],
                         data['RA'],
                         data['DEC'])]

    self.xvals    = np.arange(self.data.shape[0]).reshape(-1,1)
    self.features = np.array(self.features)

    if subset_photoz_bin:
      # Subset to a single photo-z bin
      keep = (self.data[np.where(self.features == 'PHOTOZ_BIN')[0][0],:] ==  \
              photoz_bin)
      self.data   = self.data[:,keep]
      # Still annoys me that you can't index a list with a list
      self.labels = [self.labels[k] for k in np.where(keep)[0]]
      
    # Remove the MEAN_PHOTOZ and PHOTOZ_BIN features
    print('Removing PHOTOZ features.')
    features_keep = ((self.features != 'PHOTOZ_BIN') &
                     (self.features != 'MEAN_PHOTOZ'))
    self.data     = self.data[features_keep,:]
    self.features = self.features[features_keep]
    self.xvals    = np.arange(self.data.shape[0]).reshape(-1,1)


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

    fig = pylab.figure()
    ax = fig.add_subplot(1, 1, 1)

    x = np.array(x)
    xvals = [self.xvals[z][0] for z in range(self.xvals.shape[0])]
    x = [x[z] for z in range(x.shape[0])]

    # Make a line plot
    pylab.plot([xvals[i] for i in goodfeat],
               x, 'bo-', markersize=12, label='Observations')
    pylab.plot([xvals[i] for i in goodfeat], 
               r, 'rx-', markersize=12, label='Expected')

    # dashed line to show 0
    pylab.plot([0, len(self.features)], [0, 0], 'k--')

    pylab.xlabel(self.xlabel)
    pylab.ylabel(self.ylabel)
    pylab.title('DEMUD selection %d (%s),\n item %d, using K=%d' % \
                (m, label, ind, k))
    pylab.legend(fontsize=12)
    
    if len(self.features) == 0:
        pylab.xticks(pylab.arange(len(x)), range(len(x)), fontsize=12)
    else:
        pylab.xticks(pylab.arange(len(x)), self.features,
                     rotation=-30, ha='left', fontsize=12)
    pylab.tight_layout()
    
    if not os.path.exists('results'):
      os.mkdir('results')
    outdir = os.path.join('results', self.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    figfile = os.path.join(outdir, 'sel-%d-k-%d-(%s).png' % (m, k, label))
    pylab.savefig(figfile)
    print 'Wrote plot to %s' % figfile


  # Write a list of the selections in CSV format
  def write_selections_csv(self, i, k, orig_ind, label, ind, scores):
    outdir = os.path.join('results', self.name)
    selfile = os.path.join(outdir, 'selections-k%d.csv' % k)

    (objid, RA, DEC) = label.split('_')

    # If this is the first selection, open for write
    # to clear out previous run.
    if i == 0:
      fid = open(selfile, 'w')
      # Output a header.  For some data sets, the label is a class;
      # for others it is an object identifier.  To be generic,
      # here we call this 'Name'.
      fid.write('# Selection, Index, Name, RA, DEC, Score\n')

      # If scores is empty, the (first) selection was pre-specified,
      # so there are no scores.  Output 0 for this item.
      if scores == []:
        fid.write('%d,%d,%s,%s,%s,0.0\n' % (i, orig_ind, objid,
                                            RA, DEC))
      else:
        fid.write('%d,%d,%s,%s,%s,%g\n' % (i, orig_ind, objid,
                                           RA, DEC, scores[ind]))
    else:
      # Append to the CSV file
      fid = open(selfile, 'a')
      fid.write('%d,%d,%s,%s,%s,%g\n' % (i, orig_ind, objid,
                                         RA, DEC, scores[ind]))

    # Close the file
    fid.close()

    # Also, append selections to a growing .html file
    self.write_selections_html(10, i, k, ind, label, scores)


  # Write a list of n selections that are similar to selection i (index ind)
  # using scores (with respect to selection i).
  def write_selections_html(self, n, i, k, ind, label, scores):
    outdir = os.path.join('results', self.name)
    selfile = os.path.join(outdir, 'selections-k%d.html' % k)

    (objid, RA, DEC) = label.split('_')

    # If this is the first selection, open for write
    # to clear out previous run.
    if i == 0:
      # Start up the HTML file
      fid = open(selfile, 'w')
      fid.write('<html><head><title>DEMUD: %s, k=%d</title></head>\n' % (self.name, k))
      fid.write('<body>\n')
      fid.write('<h1>DEMUD experiments on %s with k=%d</h1>\n' % (self.name, k))
      fid.write('%d (%g) items analyzed.<br>\n' % 
                (self.data.shape[1], self.data.shape[1]))
      fid.write('<ul>\n')
      fid.write('<li>Selections are presented in decreasing order of novelty.</li>\n')
      fid.write('<li>Cutouts (left) are RGB images generated from the DES DR1 archive.</li>\n')
      fid.write('<li>The bar plot shows the <font color="blue">observed</font> values compared to the <font color="red">expected (modeled)</font> values.  Discrepancies explain why the chosen object is considered novel.  Click to enlarge.</li>\n')
      fid.write('<li>Scores close to 0 (for items other than the first one) indicate an arbitrary choice; novelty has been exhausted.</li>\n')
      fid.write('</ul>\n\n')

      # If scores is empty, the (first) selection was pre-specified,
      # so there are no scores.  Output -1 for this item.
      if scores == []:
        score = 'N/A'
      else:
        score = '%f' % scores[ind]
    else:
      # Append to the HTML file
      fid = open(selfile, 'a')
      score = scores[ind]

    fid.write('<h2>Selection %d: %s, RA %s, DEC %s, score %s</h2>\n' % 
              (i, objid, RA, DEC, score))
    fid.write('<a href="selection-%d-cutout.png"><img title="[%d] %s" src="selection-%d-cutout.png" height=270></a>\n' %
                  (i, i, objid, i))
    figfile = 'sel-%d-k-%d-(%s).png' % (i, k, label)
    fid.write('<a href="%s"><img height=270 src="%s"></a>\n\n' % 
              (figfile, figfile))

    # Close the file
    fid.close()


if __name__ == "__main__":
  # Run inline tests    
  import doctest                                                              

  (num_failed, num_tests) = doctest.testmod()
  filename                = os.path.basename(__file__)

  if num_failed == 0:
    print "%-20s All %3d tests passed!" % (filename, num_tests)
  else:
    sys.exit(1)
