#!/usr/bin/env python
# File: dataset_decals.py
# Author: Kiri Wagstaff, 3/1/17
#
# Class for reading in and plotting DECaLS data in FITS format.
# DECaLS: Dark Energy Camera Legacy Survey
# Inspired by Arjun Dey
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
import numpy as np
import pyfits, pylab
from dataset import Dataset

class DECaLSData(Dataset):

  def __init__(self, decalsfilename=None):
    """DECaLSData(decalsfilename=None)

    Read in DECaLS catalog data from FITS file.
    """

    Dataset.__init__(self, decalsfilename, "DECaLSData", '')

    self.readin()


  @classmethod
  def  flux_to_magnitude(cls, flux, ivar):
    """flux_to_magnitude(cls, flux)

    Convert flux(es) into magnitude(s).

    Examples:

    >>> DECaLSData.flux_to_magnitude(20, 0.1)
    19.247425010840047

    >>> DECaLSData.flux_to_magnitude(-20, 0.1)
    21.25

    >>> DECaLSData.flux_to_magnitude([20, -20], [0.1, 0.1])
    array([ 19.24742501,  21.25      ])
    """

    # Version 1: convert abs(flux) to magnitude, 
    # and flip the sign of anything negative.
    #s = np.sign(flux)
    #magnitude = s * (-2.5 * np.log10(np.abs(flux)) + 22.5)

    # Version 2: convert abs(flux) to magnitude,
    # and replace any fluxes that are < 1 sigma with the 1-sigma upper limit.
    rms       = np.atleast_1d(1 / np.sqrt(ivar))
    # Convert flux to an np.array so this works whether input
    # is scalar or array
    flux              = np.atleast_1d(flux)
    flux2             = np.atleast_1d(rms) # initialize with rms
    flux2[flux > rms] = flux[flux > rms]   # override if flux is valid
    magnitude = (-2.5 * np.log10(flux2)) + 22.5

    # Return an array if more than one result; else return a scalar
    if len(magnitude) > 1:
      return magnitude
    else:
      return magnitude[0]
    

  def  readin(self):
    """readin()

    Read in DECaLS table data from FITS file.
    """

    datafile = pyfits.open(self.filename)

    # Read in the desired columns.
    '''
    # First the easy ones: we'll index them with [1,2,4] to get G,R,Z bands
    columns = ['DECAM_FLUX',
               'DECAM_ANYMASK',
               'DECAM_FRACFLUX',
               'DECAM_FRACMASKED',
               'DECAM_RCHI2']

    # Use the G,R,Z bands for several features
    self.data     = datafile[1].data[:].field(columns[0])[:,[1,2,4]]
    self.features = ['%s %s' % (columns[0], b) for b in ['G','R','Z']]

    for c in columns[1:]:
      self.data      = np.hstack([self.data,
                               datafile[1].data[:].field(c)[:,[1,2,4]]])
      self.features += ['%s %s' % (c, b) for b in ['G','R','Z']]
    '''

    # Compute the color ratios
    model_type = datafile[1].data[:].field('TYPE')
    #model_type_to_use = 'PSF'
    #model_type_to_use = 'SIMP'
    #model_type_to_use = 'EXP'
    model_type_to_use = 'DEV'
    use_data   = datafile[1].data[model_type == model_type_to_use]
    G = use_data.field('DECAM_FLUX')[:,1]
    R = use_data.field('DECAM_FLUX')[:,2]
    Z = use_data.field('DECAM_FLUX')[:,4]
    G_ivar = use_data.field('DECAM_FLUX_IVAR')[:,1]
    R_ivar = use_data.field('DECAM_FLUX_IVAR')[:,2]
    Z_ivar = use_data.field('DECAM_FLUX_IVAR')[:,4]
    # Z magnitude
    self.data = DECaLSData.flux_to_magnitude(Z, Z_ivar) 
    self.features = ['Z']
    # difference G-R
    self.data = np.vstack([self.data,
                           DECaLSData.flux_to_magnitude(G, G_ivar) - 
                           DECaLSData.flux_to_magnitude(R, R_ivar)])
    self.features += ['G-R']
    '''
    # difference G-Z
    self.data = np.vstack([self.data,
                           DECaLSData.flux_to_magnitude(G, G_ivar) - 
                           DECaLSData.flux_to_magnitude(Z, Z_ivar)])
    self.features += ['G-Z']
    '''
    # difference R-Z
    self.data = np.vstack([self.data,
                           DECaLSData.flux_to_magnitude(R, R_ivar) - 
                           DECaLSData.flux_to_magnitude(Z, Z_ivar)])
    self.features += ['R-Z']

    # WISE features
    W1 = use_data.field('WISE_FLUX')[:,0]
    W2 = use_data.field('WISE_FLUX')[:,1]
    W1_ivar = use_data.field('WISE_FLUX_IVAR')[:,0]
    W2_ivar = use_data.field('WISE_FLUX_IVAR')[:,1]
    # WISE difference Z - W1
    self.data = np.vstack([self.data,
                           DECaLSData.flux_to_magnitude( Z,  Z_ivar) - 
                           DECaLSData.flux_to_magnitude(W1, W1_ivar)])
    self.features += ['Z - W1']
    # WISE difference W1 - W2
    self.data = np.vstack([self.data,
                           DECaLSData.flux_to_magnitude(W1, W1_ivar) - 
                           DECaLSData.flux_to_magnitude(W2, W2_ivar)])
    self.features += ['W1 - W2']

    '''
    G = use_data.field('DECAM_APFLUX')[:,1,2]
    R = use_data.field('DECAM_APFLUX')[:,2,2]
    Z = use_data.field('DECAM_APFLUX')[:,4,2]
    G_ivar = use_data.field('DECAM_APFLUX_IVAR')[:,1,2]
    R_ivar = use_data.field('DECAM_APFLUX_IVAR')[:,2,2]
    Z_ivar = use_data.field('DECAM_APFLUX_IVAR')[:,4,2]
    # aperture difference G-R
    self.data = np.vstack([self.data,
                           DECaLSData.flux_to_magnitude(G, G_ivar) - 
                           DECaLSData.flux_to_magnitude(R, R_ivar)])
    self.features += ['AP G-R']
    # aperture difference G-Z
    self.data = np.vstack([self.data,
                           DECaLSData.flux_to_magnitude(G, G_ivar) - 
                           DECaLSData.flux_to_magnitude(Z, Z_ivar)])
    self.features += ['AP G-Z']
    # aperture difference R-Z
    self.data = np.vstack([self.data,
                           DECaLSData.flux_to_magnitude(R, R_ivar) - 
                           DECaLSData.flux_to_magnitude(Z, Z_ivar)])
    self.features += ['AP R-Z']
    '''

    #self.data   = self.data.T  # features x samples
    self.labels = ['%s_%d_%.6f_%.6f' % (b,id,ra,dec) for (b,id,ra,dec) in \
                     zip(use_data.field('BRICKNAME'),
                         use_data.field('OBJID'),
                         use_data.field('RA'),
                         use_data.field('DEC'))]

    datafile.close()

    self.xvals    = np.arange(self.data.shape[0]).reshape(-1,1)
    self.features = np.array(self.features)


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
    pylab.close()
  

  # Write a list of the selections in CSV format
  def write_selections_csv(self, i, k, orig_ind, label, ind, scores):
    outdir = os.path.join('results', self.name)
    selfile = os.path.join(outdir, 'selections-k%d.csv' % k)

    (brickname, objid, RA, DEC) = label.split('_')

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
        fid.write('%d,%d,%s_%s,%s,%s,0.0\n' % (i, orig_ind, brickname, objid,
                                                  RA, DEC))
      else:
        fid.write('%d,%d,%s_%s,%s,%s,%g\n' % (i, orig_ind, brickname, objid,
                                                 RA, DEC, scores[ind]))
    else:
      # Append to the CSV file
      fid = open(selfile, 'a')
      fid.write('%d,%d,%s_%s,%s,%s,%g\n' % (i, orig_ind, brickname, objid,
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

    (brickname, objid, RA, DEC) = label.split('_')

    # If this is the first selection, open for write
    # to clear out previous run.
    if i == 0:
      # Start up the HTML file
      fid = open(selfile, 'w')
      fid.write('<html><head><title>DEMUD: %s, k=%d</title></head>\n' % (self.name, k))
      fid.write('<body>\n')
      fid.write('<h1>DEMUD experiments on %s with k=%d</h1>\n' % (self.name, k))
      fid.write('<ul>\n')
      fid.write('<li>Selections are presented in decreasing order of novelty.</li>\n')
      fid.write('<li>The bar plot shows the <font color="blue">observed</font> values compared to the <font color="red">expected (modeled)</font> values.  Discrepancies explain why the chosen object is considered novel.  Click to enlarge.</li>\n')
      fid.write('<li>Clicking the object image will take you to the DECaLS sky survey.</li>\n')
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

    fid.write('<h2>Selection %d: RA %s, DEC %s, score %s</h2>\n' % (i, RA, DEC, score))
    fid.write('<a href="http://legacysurvey.org/viewer?ra=%s&dec=%s&zoom=13&layer=decals-dr3" id="[%d] %s %s">\n<img title="[%d] %s %s" src="http://legacysurvey.org/viewer/jpeg-cutout/?ra=%s&dec=%s&pixscale=0.27&size=256"></a>\n' %
                  (RA, DEC, 
                   i, brickname, objid,
                   i, brickname, objid, RA, DEC))
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
