#!/usr/bin/env python
# File: dataset_envi.py
# Author: Kiri Wagstaff, 5/7/13
#
# Readers and plotters for ENVI (hyperspectral) data sets
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

import os, sys
from PIL import Image
import numpy as np
import pylab, math, struct, pickle
import matplotlib
import h5py
from matplotlib.patches import Rectangle
from .dataset import Dataset
from .dataset_libs import LIBSData
#from nims_data import NimsQube
from ..log.log import printt

################### ENVI ##############
class ENVIData(Dataset):
  # Contains code needed to load, plot, and interpret 
  # ENVI (hyperspectral) data

  def  __init__(self, filename=None, shotnoisefilt=0, fwfile=''):
    """ENVIData(filename="")

    Read in ENVI (hyperspectral) data from a file (.img or .pkl).

    Optionally, specify the width of a median filter to apply.
    Optionally, specify a file containing per-feature weights.
    If one or both of these are specified, create an appropriate
    filename; read from it if it exists, or generate it if not.
    """

    #Dataset.__init__(self, filename,
    #                 'envi-' + \
    #                 filename[filename.rfind('/')+1:filename.find('.')],
    #                 '')
    Dataset.__init__(self, filename,
                     'envi-' + \
                       os.path.splitext(os.path.basename(filename))[0], '')

    # Specified filename must end in .img
    if not (filename.lower().endswith('.img') or
            filename.lower().endswith('.mat') or # hack to support MISE APL data
            filename.lower().endswith('.qub')): # hack to support NIMS data
      raise ValueError('Specify the ENVI filename, ending with .img (or .IMG).')

    # Construct the filename.
    # 1. If shotnoisefilt or fwfile are specified,
    #    we need to generate a new data file and save it into a .pkl.
    #    Construct the new output filename.
    fnsuffix = ''
    if shotnoisefilt != 0:
      fnsuffix += '-snf%d' % shotnoisefilt
    if fwfile != '':
      fnsuffix += '-fw-%s' % os.path.basename(fwfile)

    '''
    # If neither option was specified, use the .img file as-is (no pre-processing).
    # Otherwise, switch to a .pkl.
    if fnsuffix != '':
      # Pickled filename
      self.filename = self.filename[:-4] + fnsuffix + '.pkl'
    '''

    # If filename is a pickled file (ends in .pkl) and exists,
    # use it directly.
    # Otherwise, read in raw data and write that pickled file.
    if self.filename.endswith('.pkl'):
      if os.path.exists(self.filename):
        print('Reading ENVI data from pickled file %s (pre-processing already done).' % filename)
        self.readin()
      else: 
        print('Reading from ENVI file %s and writing to pickled file %s.' % (filename, self.filename))
        # Read in the ENVI file and apply any preprocessing needed
        self.read_from_scratch(filename, shotnoisefilt, fwfile)
        # Save results to a pickled file.
        outf = open(self.filename, 'w')
        pickle.dump((self.lines, self.samples, self.data, self.rgb_data,
                     self.xlabel, self.ylabel, self.xvals, self.labels), outf)
        outf.close()
    elif self.filename.endswith('.mat'):  # APL MISE data cube
      cube = h5py.File(self.filename, 'r').get('cube')
      cube = np.array(cube)
      # Cube is wavelengths x height x width
      raw_data = cube.T
      (self.lines, self.samples, wavelengths) = raw_data.shape
      self.data = np.zeros((wavelengths, self.lines * self.samples), 'uint16')
      for b in range(wavelengths):
        self.data[b,:] = raw_data[:,:,b].reshape(-1)

      self.xlabel = 'Wavelength (nm)'
      self.xvals  = np.arange(800,5000,10)
      self.ylabel = 'DN'

      # Let labels be x,y coordinates
      # (but store as x_y so CSV doesn't get confused)
      self.labels = []
      for l in range(self.lines):
        for s in range(self.samples):
          self.labels += ['%d_%d' % (l,s)]

      # Store the RGB data for later use
      self.rgb_data = self.get_RGB()

    elif self.filename.endswith('.qub'):  # NIMS data
      nims = NimsQube(self.filename)

      # nims.data is line x sample x band
      (self.lines, self.samples, wavelengths) = nims.data.shape

      self.data = np.zeros((wavelengths, self.lines * self.samples))
      for b in range(wavelengths):
        self.data[b,:] = nims.data[:,:,b].reshape(-1)

      self.xlabel = 'Wavelength ($\mu$m)'
      self.xvals  = nims.band_wavelengths
      self.ylabel = 'Radiance' if 'cr' in self.filename else 'Reflectance'

      # Ensure bands are in order
      ord = np.argsort(self.xvals)
      self.data  = self.data[ord]
      self.xvals = self.xvals[ord]

      # Let labels be x,y coordinates 
      # (but store as x_y so CSV doesn't get confused)
      self.labels = []
      for l in range(self.lines):
        for s in range(self.samples):
          self.labels += ['%d_%d' % (l,s)]

      # Store the RGB data for later use
      self.rgb_data = self.get_RGB()

      # Restrict wavelength range as appropriate
      if 'cr.qub' in self.filename:
        bandsuse = self.xvals >= 4.000 # at least 4 um
      else: # assume reflectance
        bandsuse = self.xvals <= 4.000 # below 4 um
      self.data  = self.data[bandsuse,:]
      self.xvals = self.xvals[bandsuse]

      # Replace assumed sentinel value of -1.70141143e+38 with NaN
      if 'ci.qub' in self.filename:
        bogus = np.where(self.data < -1e37)
        self.data[bogus] = np.nan
      
      # Filter shot noise
      if shotnoisefilt >= 3:
        self.data = LIBSData.medfilter(self.data, shotnoisefilt)

      print(self.xvals)

    else:
      print('Reading from ENVI file %s, no pre-processing.' % filename)
      self.read_from_scratch(filename)

    # Set up the priority map - 0 means not yet prioritized
    self.pr_map = np.zeros((self.lines, self.samples))


  def  readin(self):
    """readin()

    Read in ENVI (hyperspectral) data from (pickled) filename.
    """
    
    inf = open(self.filename, 'r')
    (self.lines, self.samples, self.data, self.rgb_data,
     self.xlabel, self.ylabel, self.xvals, self.labels) = \
        pickle.load(inf)
    print(' Dimensions: %d lines, %d samples.' % (self.lines, self.samples))

    inf.close()


  def  read_from_scratch(self, filename, shotnoisefilt=0, fwfile=''):
    """read_from_scratch()

    Read in ENVI (hyperspectral) data from filename.
    Assume header file is filename.hdr.

    Optionally, specify the width of a median filter to apply.
    Optionally, specify a file containing per-feature weights.

    Strongly inspired by enviread.m from Ian Howat, ihowat@gmail.com.

    See ROI_utils.py for full development and testing.
    """

    envi_file = filename

    # Read in the header file.  Try a few options to find the .hdr file. 
    hdrfilenames = [envi_file + '.hdr',
                    envi_file[0:envi_file.rfind('.IMG')] + '.hdr',
                    envi_file[0:envi_file.rfind('.img')] + '.hdr']
    for hdrfile in hdrfilenames:
      if os.path.exists(hdrfile):
        break
      
    info = ENVIData.read_envihdr(hdrfile)
    self.lines   = info['lines']
    self.samples = info['samples']
    print('%d lines, %d samples, %d bands.' % (self.lines, self.samples, info['bands']))

    # Set binary format parameters
    byte_order = info['byte order']
    if   (byte_order == 0):
      machine = 'ieee-le'
    elif (byte_order == 1):
      machine = 'ieee-be'
    else:
      machine = 'n'

    dtype = info['data type']
    if   (dtype == 1):
      format = 'uint8'
    elif (dtype == 2):
      format = 'int16'
    elif (dtype == 3):
      format = 'int32'
    elif (dtype == 4):
      format = 'float32'
    elif (dtype == 5):
      format = 'float64'  # Note: 'float' is the same as 'float64'
    elif (dtype == 6):
      print(':: Sorry, Complex (2x32 bits) data currently not supported.')
      print(':: Importing as double-precision instead.')
      format = 'float64'
    elif (dtype == 9):
      print(':: Sorry, double-precision complex (2x64 bits) data currently not supported.')
      return
    elif (dtype == 12):
      format = 'uint16'
    elif (dtype == 13):
      format = 'uint32'
    elif (dtype == 14):
      format = 'int64'
    elif (dtype == 15):
      format = 'uint64'
    else:
      print('Error: File type number: %d not supported' % dtype)
      return None
    print('Reading data format %s' % format)

    # Read in the data
    try:
      dfile = open(envi_file, 'r')
    except IOError:
      print(":: Error: data file '%s' not found." % envi_file)
      return None

    self.data = np.zeros((info['bands'], info['lines'] * info['samples']),
                            format)

    raw_data  = np.fromfile(dfile, format, -1)
    dfile.close()

    band_format = info['interleave'].lower()

    if (band_format == 'bsq'):
      print("Reading BSQ: Band, Row, Col; %s" % machine)
      raw_data = raw_data.reshape((info['bands'],info['lines'],info['samples']))
      for b in range(info['bands']):
        for i in range(info['lines'] * info['samples']):
          l = i // info['samples']
          s = i % info['samples']
          self.data[b,i] = raw_data[b,l,s]

    elif (band_format == 'bil'):
      print("Reading BIL: Row, Band, Col; %s" % machine)
      raw_data = raw_data.reshape((info['lines'],info['bands'],info['samples']))
      for b in range(info['bands']):
        for i in range(info['lines'] * info['samples']):
          l = i // info['samples']
          s = i % info['samples']
          self.data[b,i] = raw_data[l,b,s]
    
    elif (band_format == 'bip'):
      print("Reading BIP: Row, Col, Band; %s" % machine)
      raw_data = raw_data.reshape((info['lines'],info['samples'],info['bands']))
      for b in range(info['bands']):
        self.data[b,:] = raw_data[:,:,b].reshape(-1)

    # Determine whether we need to swap byte order
    little_endian = (struct.pack('=f', 2.3) == struct.pack('<f', 2.3))
    if (     little_endian and machine == 'ieee-be') or \
        (not little_endian and machine == 'ieee-le'):
      self.data.byteswap(True)

    self.xlabel = 'Wavelength (nm)'
    self.xvals  = info['wavelength']
    self.ylabel = 'Reflectance'

    # Let labels be x,y coordinates
    # (but store as x_y so CSV doesn't get confused)
    self.labels = []
    for l in range(info['lines']):
      for s in range(info['samples']):
        self.labels += ['%d_%d' % (l,s)]

    # Data pre-processing (UCIS specific)
    #if 'UCIS' in envi_file or 'ucis' in envi_file:
    if 'mars_yard' in envi_file:
      printt('Filtering out water absorption and known noisy bands,')
      printt(' from %d' % len(self.xvals))
      # Water: 1.38 and 1.87 nm
      # Also prune out the first 10 and last 3 bands
      waves_use = [w for w in self.xvals
                   if ((w > 480 and w < 1330) or 
                       (w > 1400 and w < 1800) or
                       (w > 1900 and w < 2471))]
      bands_use = [np.where(self.xvals == w)[0][0] for w in waves_use]
      self.data  = self.data[bands_use, :]
      self.xvals = self.xvals[bands_use]
      printt(' to %d bands.' % len(self.xvals))

    # Filter out shot noise (median filter)
    # warning: this is slow... (should be optimized?)
    from demud import read_feature_weights
    if shotnoisefilt >= 3:
      # Read in feature weights, if needed
      fw = read_feature_weights(fwfile, self.xvals)
      self.data = LIBSData.medfilter(self.data, shotnoisefilt, fw)
 
    # Store the RGB data for later use
    self.rgb_data = self.get_RGB()


  @classmethod
  def  read_envihdr(cls, hdrfile):
    """read_envihdr(hdrfile)

    Read ENVI image file header information.

    Examples:

    >>> read_envihdr('')
    :: Error: header file '' not found.

    >>> info = read_envihdr('%s/f970619t01p02_r02_sc04.a.rfl.hdr' % basedir)
    >>> assert info['lines']            == 512
    >>> assert info['samples']          == 614
    >>> assert info['bands']            == 224
    >>> assert info['wavelength units'] == 'Nanometers'
    >>> assert info['wavelength'].size  == info['bands']
    >>> assert info['fwhm'].size        == info['bands']
    
    """

    try:
      hfile = open(hdrfile, 'r')
    except IOError:
      print(":: Error: header file '%s' not found." % hdrfile)
      return None

    info = {}

    while 1:
      line = hfile.readline()
      if not line:
        break
      if '=' in line:
        toks  = line.strip().split('=')
        param = toks[0].strip()
        value = toks[1]
        if '{' in value and '}' not in value:
          while '}' not in value:
            value = ''.join([value, hfile.readline()])
        # Handle arrays and numeric values
        if '{' in value and '}' in value:
          open_brace  = value.find('{')
          close_brace = value.find('}')
          contents    = value[open_brace+1:close_brace]
          # If it has at least one comma, treat it as an array
          if ',' in value:
            if 'Band' in value:
              # Array of strings.  Strip off the band number, convert to float.
              # This is particular to the UCIS data.
              value = np.array([float(x.split(':')[0].split(' ')[-1]) \
                                   for x in contents.split(',')])
            else:
              # Array of floats
              value = np.array([float(x) for x in contents.split(',')])
          else: # it's a string
            value = contents.strip()
        else:
          # Try to convert it to an int or float.  If it fails, leave it as a string.
          value = value.strip()
          try:
            if value.find('.') == -1:
              value = int(value)
            else:
              value = float(value)
          except ValueError:
            pass
        info[param] = value

    hfile.close()

    return info


  def  get_RGB(self):
    """get_RGB(data)

    Get the bands corresponding to RGB data.
    """

    waves = self.xvals
    
    # Reshape if needed
    if len(self.data.shape) == 2:
      data = self.data.T.reshape((self.lines, self.samples, -1))
    else:
      data = self.data

    # Hua suggests 636.07 (red), 557.07 (green), 468.31 (blue)
    r_band = np.argmin([abs(w-636) for w in waves])
    g_band = np.argmin([abs(w-557) for w in waves])
    #b_band = np.argmin([abs(w-468) for w in waves])
    # We're pruning out the first 10 bands due to noise, so use the next one for the blue band
    b_band = np.argmin([abs(w-490) for w in waves])
      
    if '.mat' in self.filename:
      # diff bands for MISE
      r_band = 290 # 3700 nm
      g_band = 145 # 2250 nm
      b_band = 40  # 1200 nm
    elif '.qub' in self.filename: 
      # diff bands for NIMS (per Shirley et al., 2010)
      r_band = np.argmin([abs(w-1.500) for w in waves]) # brighter than water ice
      g_band = np.argmin([abs(w-1.300) for w in waves]) # water ice absorption
      b_band = np.argmin([abs(w-0.730) for w in waves])

    print('Using bands: %d red (%f), %d green (%f), %d blue (%f), zero-indexed.' % \
        (r_band, waves[r_band],
         g_band, waves[g_band],
         b_band, waves[b_band]))

    # Convert data to 8-bit format (was 16...)
    rgb_data = np.zeros((self.lines, self.samples, 3),
                           'uint8')
    # Normalize by max for each band
    r_maxval = np.nanmax(data[:,:,r_band])
    g_maxval = np.nanmax(data[:,:,g_band])
    b_maxval = np.nanmax(data[:,:,b_band])

    # Normalize per band
    rgb_data[:,:,0] = data[:,:,r_band] / float(r_maxval) * 255
    rgb_data[:,:,1] = data[:,:,g_band] / float(g_maxval) * 255
    rgb_data[:,:,2] = data[:,:,b_band] / float(b_maxval) * 255

    return rgb_data


  def  write_RGB(self, filename):
    """write_RGB(data, filename)

    Write out an RGB visualization of ENVI/AVIRIS data,
    using selected bands for red, green, and blue channels.
    Save the result as a PNG file of the specified name.
    """

    #rgb_data = self.get_RGB()
    png = Image.fromarray(self.rgb_data, 'RGB')
    png.save(filename)
    print("Wrote PNG visualization to %s." % filename)


  # Augment plot with spatial locator map
  def  plot_item(self, m, ind, x, r, k, label, U=[], scores=[], feature_weights=[]):
    """plot_item(self, m, ind, x, r, k, label, U, scores, feature_weights)

    Plot selection m (index ind, data in x) and its reconstruction r,
    with k and label to annotate of the plot.

    Also show a spatial plot indicating where the selected pixel is
    and an abundance plot of similarity across the data set.

    U and scores are optional; ignored in this method, used in some
    classes' submethods.

    If feature_weights are specified, omit any 0-weighted features from the plot.
    """
    
    if len(x) == 0 or len(r) == 0: 
      printt("Error: No data in x and/or r.")
      return

    (l,s) = [int(v) for v in label.split('_')]

    # Select the features to plot
    if len(feature_weights) > 0:
      goodfeat = [f for f in range(len(feature_weights)) \
                    if feature_weights[f] > 0]
    else:
      #goodfeat = list(range(len(self.xvals)))
      # Avoid NaNs
      goodfeat = np.where(~np.isnan(x) & ~np.isnan(r))[0]

    # Set up the subplots
    pylab.figure()
    #pylab.subplots_adjust(wspace=0.1, left=0)
    #pylab.subplots_adjust(wspace=0.05) # Argadnel - 14e006
    pylab.subplots_adjust(wspace=0.05, hspace=0.45) # Pwyll - 12e001

    # Plot #1: expected vs. observed feature vectors
    # xvals, x, and r need to be column vectors
    pylab.subplot(2,2,1)
    # plot lines where data is continuous in terms of xvals
    # NIMS wavelengths are not evenly spaced, so it is nontrivial
    # to figure out where real data gaps lie.
    # For gaps > 0.015 um, insert pseudo data points with nans 
    # to break the plot lines.
    xvals_full = [self.xvals[goodfeat[0]]]
    for v in self.xvals[goodfeat[1:]]:
      if (v-xvals_full[-1] > 0.015):
        xvals_full += [xvals_full[-1] + 0.015]
      xvals_full += [v]
    rused = dict(zip(self.xvals[goodfeat], r[goodfeat]))
    xused = dict(zip(self.xvals[goodfeat], x[goodfeat]))
    rcopy = np.ones_like(xvals_full) * np.nan
    for (i,v) in enumerate(xvals_full):
      if v in rused:
        rcopy[i] = rused[v]
    xcopy = np.ones_like(xvals_full) * np.nan
    for (i,v) in enumerate(xvals_full):
      if v in xused:
        xcopy[i] = xused[v]
    #import pdb; pdb.set_trace()
    pylab.plot(xvals_full, rcopy, 'r-', label='Expected',
               markersize=3)
    pylab.plot(xvals_full, xcopy, 'b-', label='Observations',
               markersize=3)
    pylab.ylim([0.0, max(1.0, np.nanmax(xcopy))])
    pylab.locator_params(axis='x', nbins=10)

    pylab.xlabel(self.xlabel)
    pylab.ylabel(self.ylabel)
    pylab.legend(fontsize=10)#, loc=2)

    # Plot #2: zoom of selected pixel, 20x20 context
    pylab.subplot(2,2,2)
    winwidth = 20
    minl = max(0, l - winwidth // 2)
    mins = max(0, s - winwidth // 2)
    maxl = min(self.lines,   l + winwidth // 2)
    maxs = min(self.samples, s + winwidth // 2)
    pylab.imshow(self.rgb_data[minl:maxl, mins:maxs],
                 interpolation='none') #, alpha=0.85)
    pylab.gca().add_patch(Rectangle((min(winwidth // 2, s) - 1,
                                     min(winwidth // 2, l) - 1),
                                     2, 2,
                                     fill=None, alpha=1))
    pylab.axis('off')
    pylab.title('Zoom')

    # Spatial selection plot
    # this is an inset axes over the main axes
    #a = pylab.axes([.15, .75, .3, .15])
    pylab.subplot(2,2,3)
    # Use alpha to lighten the RGB data
    plt = pylab.imshow(self.rgb_data, interpolation='none', 
                       alpha=0.85)
    pylab.plot(s, l, 'x', markeredgewidth=2, color='red', 
               scalex=False, scaley=False)
    #pylab.setp(a, xticks=[], yticks=[])
    pylab.axis('off')
    pylab.title('Selection')

    # Also update the priority map.
    self.pr_map[l,s] = m+1
    #print('setting %d, %d to %d' % (l, s, -m))
    n_tot = self.lines * self.samples
    n_pri = len(self.pr_map.nonzero()[0])
    n_unp = n_tot - n_pri
    printt(' %d prioritized; %d (%.2f%%) unprioritized remain' % \
             (n_pri, n_unp, n_unp * 100.0 / n_tot))

    # Abundance map
    # Compute distance from selected x to all other items
    abund = np.zeros((self.lines, self.samples))
    nbands = self.data.shape[0]
    for l_ind in range(self.lines):
      for s_ind in range(self.samples):
        if l_ind == l and s_ind == s:
          abund[l_ind,s_ind] = 0
          continue
        d = self.data[:, l_ind*self.samples + s_ind]
        # Use Euclidean distance.
        #abund[l,s] = math.sqrt(pow(np.sum(x - d), 2)) / float(nbands)
        # Use spectral angle distance
        goodfeat = np.where(~np.isnan(x) & ~np.isnan(d))[0]
        num   = np.dot(x[goodfeat].astype(float), d[goodfeat].astype(float))
        denom = np.linalg.norm(x[goodfeat].astype(float)) * \
            np.linalg.norm(d[goodfeat].astype(float))
        if num > denom: # ensure math.acos() doesn't freak out; clip to 1.0
          num = denom
        abund[l_ind,s_ind] = math.acos(num / denom)
        
        # Propagate current priority to similar items (not yet prioritized)
        # This threshold is subjectively chosen.
        # I used 0.10 for the Mars yard UCIS cube from Diana.
        # I used different values for the micro-UCIS cubes from Bethany
        # (see Evernote notes).
        # UCIS
        #if self.pr_map[l_ind,s_ind] == 0 and abund[l_ind,s_ind] <= 0.10:
        # micro-UCIS
        #if self.pr_map[l_ind,s_ind] == 0 and abund[l_ind,s_ind] <= 0.13:
        # MISE
        #if self.pr_map[l_ind,s_ind] == 0 and abund[l_ind,s_ind] <= 0.10:
        # NIMS
        #if self.pr_map[l_ind,s_ind] == 0 and abund[l_ind,s_ind] <= 0.05:
        if self.pr_map[l_ind,s_ind] == 0 and abund[l_ind,s_ind] <= 0.10:
          self.pr_map[l_ind,s_ind] = m+1


    printt('Abundance: ', np.nanmin(abund), np.nanmax(abund))
    pylab.subplot(2,2,4)
    # Use colormap jet_r so smallest value is red and largest is blue
    #pylab.imshow(abund, interpolation='none', cmap='jet_r', vmin=0, vmax=0.15)
    pylab.imshow(abund, interpolation='none', cmap='jet_r', vmin=0, vmax=0.2)
    pylab.axis('off')
    pylab.title('Abundance')

    pylab.suptitle('DEMUD selection %d (%s), item %d, using K=%d' % \
                   (m, label, ind, k))
          
    # Write the plot to a file.
    outdir = os.path.join('results', self.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)

    figfile = os.path.join(outdir, 'sel-%d-k-%d-(%s).pdf' % (m, k, label))
    pylab.savefig(figfile, bbox_inches='tight')
    print('Wrote plot to %s' % figfile)
    pylab.close()

    # Write the priority map to an image file
    pylab.figure()
    # Start with colormap jet_r so smallest value is red and largest is blue
    # Max_c must be at least 2 and no greater than 255.
    # Values greater than 255 will be mapped to the last color.
    # (Imposed because we're then saving this out as an ENVI classification map with bytes.
    #  May want to be more flexible in the future, but I can't imagine really wanting to see
    #  more than 255 distinct colors?)
    max_c    = 255 if m > 254   else m+2
    max_c    = 2   if max_c < 2 else max_c
    cmap     = matplotlib.cm.get_cmap('jet_r', max_c)
    # Tweak so 0 is white; red starts at 1
    jet_map_v    = cmap(np.arange(max_c))
    #jet_map_v[0] = [1,1,1,1]  # white
    cmap         = matplotlib.colors.LinearSegmentedColormap.from_list("jet_map_white", jet_map_v)
    pr_map_plot = np.copy(self.pr_map)
    # Set unprioritized items to one shade darker than most recent
    pr_map_plot[pr_map_plot == 0] = m+2
    #pylab.imshow(pr_map_plot, interpolation='none', cmap=cmap, vmin=1, vmax=m+1)
    pylab.imshow(pr_map_plot, interpolation='none', cmap=cmap)
    prmapfig = os.path.join(outdir, 'prmap-k-%d.png' % k)
    pylab.savefig(prmapfig)
    if (m % 10) == 0:
      prmapfig = os.path.join(outdir, 'prmap-k-%d-m-%d.png' % (k, m))
      pylab.savefig(prmapfig)
    print('Wrote priority map figure to %s (max_c %d)' % (prmapfig, max_c))
    pylab.close()

    # Write the priority map contents to a file as a 64-bit float map 
    # (retained for backward compatibility for Hua,
    # but superseded by ENVI data file next)
    prmapfile = os.path.join(outdir, 'prmap-k-%d-hua.dat' % k)
    fid = open(prmapfile, 'wb')
    self.pr_map.astype('float64').tofile(fid)
    fid.close()
    print("Wrote Hua's priority map (data) to %s" % prmapfile)

    # Write an ENVI data file and header file
    prmapfile = os.path.join(outdir, 'prmap-k-%d.dat' % k)
    fid = open(prmapfile, 'wb')
    self.pr_map.astype('uint8').tofile(fid)  # save out as bytes

    # This is a class map header file
    prmaphdr  = os.path.join(outdir, 'prmap-k-%d.dat.hdr' % k)
    fid = open(prmaphdr, 'w')
    fid.write('ENVI\n')
    fid.write('description = { DEMUD prioritization map }\n')
    fid.write('samples = %d\n' % self.samples)
    fid.write('lines = %d\n'   % self.lines)
    fid.write('bands = 1\n')
    fid.write('header offset = 0\n')          # 0 bytes
    fid.write('file type = Classification\n')
    fid.write('data type = 1\n')              # byte (max 255 priorities)
    fid.write('interleave = bip\n')           # Irrelevant for single 'band'
    fid.write('byte order = 0\n')             # Least-significant byte first
    fid.write('classes = %d\n' % k)           # Number of classes
    # Classes include None (0) and then integers up to number of classes.
    fid.write("class names = {'None', " + ', '.join(["'%d'" % a for a in range(1, max_c)]) + '}\n')
    fid.write('class lookup = {' + 
              ',\n                '.join([' %d, %d, %d' % (r*255,g*255,b*255) for (r,g,b,a) in jet_map_v]) + 
              ' }\n')
    fid.close()
    print('Wrote ENVI data/header to priority map figure to %s[.hdr]' % prmapfile)

    # Write the selections (spectra) in ASCII format
    selfile = os.path.join(outdir, 'selections-k%d.txt' % k)
    # If this is the first selection, open for write
    # to clear out previous run.
    if m == 0:
      fid = open(selfile, 'w')
      # Output a header
      fid.write('# Index, Score')
      for w in self.xvals.tolist():
        fid.write(', %.3f' % w)
      fid.write('\n')

      # If scores is empty, the (first) selection was pre-specified,
      # so there are no scores.  Output 0 for this item.
      if len(scores) == 0:
        fid.write('%d,0.0,' % (m))
    else:
      fid = open(selfile, 'a')
      fid.write('%d,%f,' % (m, scores[m]))

    # Now output the feature vector itself
    # Have to reshape x because it's a 1D column vector
    np.savetxt(fid, x.reshape(1, x.shape[0]), fmt='%.5f', delimiter=',')

    fid.close()
    
    ####################################################

class SegENVIData(ENVIData):
  # Handles ENVI data accompanied by a segmentation map
  # that specifies how to average the spectra into mean spectra
  # for each segment.

  def  __init__(self, filename=None, segmapfile=None):
    self.segmapfile = segmapfile

    ENVIData.__init__(self, filename)


  def  readin(self):
    """readin()
    
    Also read in segmentation map from segmapfile
    and average the data, re-storing it in reduced form
    in self.data.

    Set self.labels to record the segment ids.
    """

    super(SegENVIData, self).readin()
    # data is wavelengths x pixels

    # Segmentation maps from SLIC are "raster scan, 32-bit float" (per Hua)
    # Actually, nicer to read them as ints.
    self.segmap = np.fromfile(self.segmapfile, dtype='int32', count=-1)
    if self.lines * self.samples != self.segmap.shape[0]:
      printt('Error: mismatch in number of pixels between image and segmap.')
      return

    goodbands = list(range(len(self.xvals)))
    # For AVIRIS data:
    if 'f970619' in self.name:
      printt('Removing known bad bands, assuming AVIRIS data.')
      # Per Hua's email of July 3, 2013, use a subset of good bands:
      # Indexing from 1: [10:100 116:150 180:216]
      # Subtract 1 to index from 0, but not to the end values
      # because range() is not inclusive of end
      goodbands  = list(range(9,100)) + list(range(115,150)) + list(range(179,216))
    # For UCIS data:
    elif 'mars_yard' in self.name:
      printt('Removing known bad bands, assuming UCIS data.')
      # Per Hua's email of May 8, 2014, use a subset of good bands.
      # Exclude 1.4-1.9 um (per Diana).
      waterband_min = np.argmin([abs(x-1400) for x in self.xvals])
      waterband_max = np.argmin([abs(x-1900) for x in self.xvals])
      waterbands    = list(range(waterband_min, waterband_max+1))
      # Based on Hua's visual examination, exclude bands
      # 1-6, 99-105, and 145-155.
      # Good bands are therefore 7-98, 106-144, and 156-maxband.
      # Subtract 1 to index from 0, but not to the end values
      # because range() is not inclusive of end
      maxband    = len(self.xvals)
      goodbands  = list(range(6,98)) + list(range(105,144)) + list(range(155,maxband))
      # Remove the water bands
      printt('Removing water absorption bands.')
      printt('%d good bands -> ' % len(goodbands))
      goodbands  = list(set(goodbands) - set(waterbands))
      printt(' %d good bands' % len(goodbands))
      
    self.data  = self.data[goodbands,:]
    self.xvals = self.xvals[goodbands]
    
    #self.segmap = self.segmap.reshape(self.lines, self.samples)
    self.labels  = np.unique(self.segmap)
    printt('Found %d segments.' % len(self.labels))
    newdata = np.zeros((self.data.shape[0], len(self.labels)))
    for i, s in enumerate(self.labels):
      pixels  = np.where(self.segmap == s)[0]
      #print('%d: %s: %d pixels' % (i, str(s), len(pixels)))
      # Compute and store the mean
      newdata[:,i] = self.data[:,pixels].mean(1)

    printt('Finished averaging the spectra.')
    # Update data with the averaged version
    self.data = newdata

    self.name  = self.name + '-seg'


    
