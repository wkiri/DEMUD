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
#import pyfits, 
import fitsio
import pylab
from dataset import Dataset

class DESData(Dataset):

  def __init__(self, desfilename=None):
    """DESData(desfilename=None)

    Read in DES catalog data.
    """
    # Subset to a single photo-z bin
    photoz_bin = 0

    #Dataset.__init__(self, desfilename, "DESData", '')
    Dataset.__init__(self, desfilename, "DESData_colordiff_bin" + 
                     str(photoz_bin), '')

    self.readin()
    
    print self.data
    print self.features

  def read_Y3_npy(self): 
    
    #load data
    data = np.load(self.filename)
    #data = np.load('DES_Y3.npy')
    
    #Y3 Gold Photometry flags not yet implemented 
    
    #subset Y3 npy array to desired columns according to: 
    """
    0 COADD_OBJECT_ID
    1 SOF_FLAGS 
    2 SOF_CM_FLAGS
    3 SOF_CM_FRACDEV  
    4 SOF_CM_FRACDEV_ERR  
    5 FLUX_CM_SOF_G 
    6 FLUX_CM_SOF_R
    7 FLUX_CM_SOF_I
    8 FLUX_CM_SOF_Z
    9 FLUXERR_CM_SOF_G
    10 FLUXERR_CM_SOF_R
    11 FLUXERR_CM_SOF_I
    12 FLUXERR_CM_SOF_Z 
    13 FLUX_PSF_SOF_G 
    14 FLUX_PSF_SOF_R
    15 FLUX_PSF_SOF_I 
    16 FLUX_PSF_SOF_Z
    17 FLUXERR_PSF_SOF_G
    18 FLUXERR_PSF_SOF_R
    19 FLUXERR_PSF_SOF_I
    20 FLUXERR_PSF_SOF_Z
    """

# Uncomment for luptitudes (e.g. required for flux values in Y3 fits)

    #subset = data[:, 5:9].shape
#    datatrans = np.transpose(data)
#    datatranssubset = datatrans[:,1:]
    
    #convert to luptitudes 
    #lups = np.arcsinh(subset)
    #print('the transposed data shape subset is')
    #print(datatranssubset[0])
    
    #lups = np.arcsinh(datatranssubset)
    
    #compute luptitude colors
    
    #G-R 
    #GR = lups[:,0] - lups[:,1]

    #R - I 
    #RI = lups[:,1] - lups[:,2]

    #I - Z 
    #IZ = lups[:,2] - lups[:,3]
    
    #create data vector 
#    self.data = GR
#    self.data = np.vstack([self.data, RI])
#    self.data = np.vstack([self.data, IZ])
    
# This works for the luptitudes in the original fits file 
# that provides only fluxes
#    self.data = data[1:4]
#    self.features = ['LUPTITUDE_G_R']
#    self.features += ['LUPTITUDE_R-I']
#    self.features += ['LUPTITUDE_I-Z']
#    self.labels = ['%d_None_None' % (id) for (id) in data[0]]
#    self.xvals    = np.arange(self.data.shape[0]).reshape(-1,1)
#    self.features = np.array(self.features)
    
# This is for the the subsampled h5 catalog with more info. The subsampled catalog 
# provides magnitudes, so let's try those rather than luptitudes
    
# Filter extreme values    
    
    #print(np.max(data))
    #print(np.min(data))
    data = data[:,data[2,:]<30]
    data = data[:,data[3,:]<30]
    data = data[:,data[4,:]<30]
    data = data[:,data[5,:]<30]
    data = data[:,data[2,:]>-1]
    data = data[:,data[3,:]>-1]
    data = data[:,data[4,:]>-1]
    data = data[:,data[5,:]>-1]
    #print(np.max(data[3:6,:]))
    #print(np.min(data[3:6,:]))
    
    
# Define data and features
    # G-R
    self.data = data[2,:] - data[3,:]
    self.features += ['G-R']

    # R-I
    self.data = np.vstack([self.data,
                           data[3,:] - data[4,:]])
    self.features += ['R-I']

    # I-Z
    self.data = np.vstack([self.data,
                           data[4,:] - data[5,:]])
    self.features += ['I-Z']
    
    self.data = np.vstack([self.data,
                           data[3,:]])

    self.features += ['R']
    
    print np.array(self.features).shape
    print np.array(self.data).shape 
    
    
    for f in self.features:
      if 'sof_cm_mag' in f: # subtract the min
        minval = np.min(self.data[self.features.index(f),:])
        self.data[self.features.index(f),:] -= minval
        print 'Subtracting %f from %s.' % (minval, f)
        newf = f + '-sub%.2f' % minval
        self.features[self.features.index(f)] = newf
        f = newf
    
    self.labels = ['None_%.6f_%.6f' % (ra, dec) for (ra, dec) in zip(data[0], data[1])]
    self.xvals    = np.arange(self.data.shape[0]).reshape(-1,1)
    self.features = np.array(self.features)
    
    

  def read_SV_fits(self):
        #datafile = pyfits.open(self.filename)
    #data   = datafile[1].data[0:1000000] # start small
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


    self.features = ['MAG_AUTO_G',
                     'MAG_AUTO_R',
                     'MAG_AUTO_I',
                     'MAG_AUTO_Z',
                     'PHOTOZ_BIN',
                     'MEAN_PHOTOZ']
    #self.data = np.vstack([data.field(f) for f in self.features])
    self.data = np.vstack([data[f] for f in self.features])


    # Ok, now we want R, G-R, I-Z
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
    
    print self.features

    # Data is d x n
    print self.data.shape
    # Scale some features as needed
    for f in self.features:
      if 'CLASS_STAR' in f: # 0 to 1.0
        self.data[self.features.index(f),:] *= 100
      if 'SPREAD_MODEL' in f:  # -1.0 to 1.0
        self.data[self.features.index(f),:] += 1.0
        self.data[self.features.index(f),:] /= 2.0
        self.data[self.features.index(f),:] *= 100.0
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
    
    # Subset to a single photo-z bin
    #keep = np.where(self.data[np.where(self.features == 'PHOTOZ_BIN')[0][0],:] ==  \
    photoz_bin = 0 #why is this not defined globally?
    keep = (self.data[np.where(self.features == 'PHOTOZ_BIN')[0][0],:] ==  \
              photoz_bin)
    self.data   = self.data[:,keep]
    # Still annoys me that you can't index a list with a list
    self.labels = [self.labels[k] for k in np.where(keep)[0]]

    # Remove the PHOTOZ_BIN feature
    features_keep = (self.features != 'PHOTOZ_BIN')
    self.data     = self.data[features_keep,:]
    self.features = self.features[features_keep]
    self.xvals    = np.arange(self.data.shape[0]).reshape(-1,1)
    
    
  def  readin(self):
    #Read in DES catalog data from FITS file.
    #Modified to read in data from a npy, e.g. for DESY3 Gold 
    
    if self.filename.endswith('.fits'):
      self.read_SV_fits()
      print('this was a fits file')
      print self.features
    elif self.filename.endswith('.npy'): 
      self.read_Y3_npy()
      print('this was a .npy')
    else: 
      print('Unrecognized filetype.') 
    

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

    # dashed line to show 0
    pylab.plot([0, len(self.features)], [0, 0], '--')
  
    pylab.xlabel(self.xlabel)
    pylab.ylabel(self.ylabel)
    pylab.title('DEMUD selection %d (%s),\n item %d, using K=%d' % \
                (m, label, ind, k))
    pylab.legend(fontsize=10)
    
    padding = 1.19
    pylab.ylim([min(0, float(min(min(x), min(r))) * padding),
                max(0, float(max(max(x), max(r))) * padding)])
    
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



#  # Write a list of the selections in CSV format
#  def write_selections_csv(self, i, k, orig_ind, label, ind, scores):
#    outdir = os.path.join('results', self.name)
#    selfile = os.path.join(outdir, 'selections-k%d.csv' % k)
#
#    (objid, RA, DEC) = label.split('_')
#
#    # If this is the first selection, open for write
#    # to clear out previous run.
#    if i == 0:
#      fid = open(selfile, 'w')
#      # Output a header.  For some data sets, the label is a class;
#      # for others it is an object identifier.  To be generic,
#      # here we call this 'Name'.
#      fid.write('# Selection, Index, Name, RA, DEC, Score\n')
#
#      # If scores is empty, the (first) selection was pre-specified,
#      # so there are no scores.  Output 0 for this item.
#      if scores == []:
#        fid.write('%d,%d,%s,%s,%s,0.0\n' % (i, orig_ind, objid,
#                                            RA, DEC))
#      else:
#        fid.write('%d,%d,%s,%s,%s,%g\n' % (i, orig_ind, objid,
#                                           RA, DEC, scores[ind]))
#    else:
#      # Append to the CSV file
#      fid = open(selfile, 'a')
#      fid.write('%d,%d,%s,%s,%s,%g\n' % (i, orig_ind, objid,
#                                         RA, DEC, scores[ind]))
#
#    # Close the file
#    fid.close()
#
#    # Also, append selections to a growing .html file
#    self.write_selections_html(10, i, k, ind, label, scores)


  # Write a list of n selections that are similar to selection i (index ind)
  # using scores (with respect to selection i).
  def write_selections_html(self, n, i, k, ind, label, scores):
    outdir = os.path.join('results', self.name)
    selfile = os.path.join(outdir, 'selections-k%d.html' % k)

    #something like if 
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
