#!/usr/bin/env python
# File: dataset_misr.py
# Author: Kiri Wagstaff, 5/7/13
#
# Readers and plotters for MISR data sets
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

import os, fnmatch, sys, pickle
import matplotlib.pyplot as plt
import numpy as np
import pylab
from numpy import nanmean
from pyhdf import HDF, SD
from dataset import Dataset
import matplotlib
#import matplotlib.plyplot as plt
import datetime

################### MISR ##############
###  Analysis per pixel (location). ###
#######################################
class MISRData(Dataset):
  # Contains code needed to load, plot, and interpret MISR data.

  def  __init__(self, rawdirname=None, AODdirname=None, filename=None, force_read=False):
    """MISRData(rawdirname="", AODdirname="", filename="")
    
    Read in raw and AOD MISR data (pickled) from filename.
    If it doesn't exist, read in the full data from
    the appropriate dirnames, save it out, and proceed.
    """

    Dataset.__init__(self, filename, "misr", '')

    if (not os.path.exists(filename)) or force_read:
      MISRData.read_misr_dir(rawdirname, AODdirname, filename)
    
    # Read in the data
    self.readin()
    

  def  readin(self):
    """readin()
    
    Read in MISR data (pickled) from self.filename.
    """

    print "Loading in MISR data from pickle..."

    inf = open(self.filename, 'r')
    (self.data, self.rgbimages, self.along_track, self.cross_track,
     self.latlons, self.datestr) = \
        pickle.load(inf)
    inf.close()
    
    print "...done."

    self.xlabel = 'Date'
    self.ylabel = 'AOD (tau)'
    self.xvals  = np.arange(self.data.shape[1]).reshape(-1,1)
    self.labels = self.latlons


  @classmethod
  def  orbit_to_date(cls, orbit):
    """orbit_to_date(orbit) -> date string

    Convert MISR orbit number to a date string and return it.
    Thanks to Mike Garay for this routine.

    Examples:

    >>> MISRData.orbit_to_date(1)
    '12/18/99 09:36'
    >>> MISRData.orbit_to_date(1506)
    '03/30/00 17:58'
    """

    # First MISR orbit: Dec. 18.4, 1999
    first_orbit = datetime.datetime(1999, 12, 18, 9, 36, 0)

    # Compute elapsed time (delta)
    num_orbits = orbit - 1
    num_secs   = num_orbits * 5933.14
    num_days   = num_secs / (60.*60.*24.)
    delta      = datetime.timedelta(num_days)

    now = first_orbit + delta

    return now.strftime('%m/%d/%y %H:%M')


  @classmethod
  def  read_misr_dir(cls, rawdirname, AODdirname, outfile):
    """read_misr_dir(rawdirname, AODdirname, outfile)

    Read in raw MISR data from .hdf files in rawdirname,
    and AOD data from all .hdf files in AODdirname.
    Pickle the result and save it to outfile.
    Note: does NOT update object fields.
    Follow this with a call to readin().
    """

      # Get the meta-information
      #meta = sd.attributes()
        #        for val in ['Origin_block.ulc.x',
        #             'Origin_block.ulc.y',
        #            'Local_mode_site_name']:
        #info[val] = meta[val]

      # Get orbit parameters?

    data = []
    rgbimages = []

    datestr = []
    datestr2 = []
    i = 0

    # Read in the AOD (from local mode) data; this is what we'll analyze
    files = sorted(os.listdir(AODdirname))
    for f in files:
      if fnmatch.fnmatch(f, '*.hdf'):
        print " %d / %d " % (i, len(files)),
        i += 1
      
        filename = AODdirname + f

        # Check that filename exists and is an HDF file
        if HDF.ishdf(filename) != 1:
          print "File %s cannot be found or is not an HDF-4 file." % filename
          continue

        orbit    = int(filename.split('_')[5].split('O')[1])
        thisdate = MISRData.orbit_to_date(orbit)
        print "orbit: %d -> %s " % (orbit, thisdate)
        datestr = datestr + [thisdate]
        
        sd = SD.SD(filename)

        # This is 3 (SOMBlock) x 32 (x) x 128 (y) x 4 (bands)
        dataset  = sd.select('RegBestEstimateSpectralOptDepth')
        dim      = dataset.dimensions()
        # Get all of the data for the green band (band = 1)
        along_track = dim['SOMBlockDim:RegParamsAer'] * dim['XDim:RegParamsAer'] 
        cross_track = dim['YDim:RegParamsAer']
        data_now = dataset.get((0,0,0,1),(dim['SOMBlockDim:RegParamsAer'],
                                          dim['XDim:RegParamsAer'],
                                          dim['YDim:RegParamsAer'],
                                          1)).squeeze()

        # Reshape to concatenate blocks
        nrows    = data_now.shape[0]*data_now.shape[1]
        ncols    = data_now.shape[2]
        data_now = data_now.reshape((nrows, ncols))

        # Set -9999 values to NaN
        naninds = np.equal(data_now, -9999)

        # Visualize this timeslice
        #pylab.imshow(data_now)
        #pylab.title(thisdate)
        #pylab.axis('off')
        #pylab.savefig(filename + '.png')

        # Set -9999 values to NaN
        data_now[naninds] = float('NaN')

        data_now = data_now.reshape((-1, 1))
        #print type(data_now)
        #print data_now.shape
        if data == []:
          data = [data_now]
        else:
          data.append(data_now)

        # Close the file
        sd.end()

        print '.',
        sys.stdout.flush()

    data = np.asarray(data).squeeze().T
    print data.shape
    
    print
    # Data is now n x d, where n = # pixels and d = # timepts
    print 'Read data set with %d pixels, %d time points.' % data.shape
    
    # TODO: Add lat/lon coords here
    latlons = ['Unknown'] * data.shape[0]

    # Read in the raw data (for later visualization)
    files = sorted(os.listdir(rawdirname + 'AN/'))
    print "+++++++++++++"
    print len(files)
    iii = 0
    for f in files:
      if fnmatch.fnmatch(f, '*.hdf'):
        filename = rawdirname + 'AN/' + f
        #print filename
        print " %d / %d " % (iii, len(files)),
        iii += 1

        # Check that filename exists and is an HDF file
        if HDF.ishdf(filename) != 1:
          print "File %s cannot be found or is not an HDF-4 file." % filename
          continue

        # We'll assume there's a one-to-one correspondence
        # with the AOD data.  But print it out anyway as a check.
        orbit    = int(filename.split('_')[6].split('O')[1])
        thisdate = MISRData.orbit_to_date(orbit)
        print "orbit: %d -> %s " % (orbit, thisdate)
        datestr2 = datestr2 + [thisdate]
        
        sd = SD.SD(filename)
        
        
        ##################################################################################################################################################################
        dataset  = sd.select('Green Radiance/RDQI')
        dim      = dataset.dimensions()
        data_g = dataset.get((60,0,0),
                             (4, dim['XDim:GreenBand'], dim['YDim:GreenBand']),
                             (1, 1, 1)
                             ).reshape([2048, 2048])
        
        mountains = np.equal(data_g, 65511)
        padding = np.equal(data_g, 65515)
        hlines = np.equal(data_g, 65523)
        
        data_g[data_g == 65515] = 0 # PADDING

        conv_factor_ds = sd.select('GreenConversionFactor')
        dim         = conv_factor_ds.dimensions()
        conv_factor = conv_factor_ds.get((60,0,0),
                                         (4, dim['XDim:BRF Conversion Factors'], dim['YDim:BRF Conversion Factors']),
                                         (1, 1, 1)
                                         ).reshape((32, 32))
        
        conv_factor[conv_factor < 0] = 0
        
        for x in range(0,data_g.shape[0],64):
          for y in range(0,data_g.shape[1],64):
            converted = np.multiply(data_g[x:x+64,y:y+64],
                                       conv_factor[x/64,y/64])
            data_g[x:x+64,y:y+64] = converted
        
        dataset  = sd.select('Red Radiance/RDQI')
        dim      = dataset.dimensions()
        data_r = dataset.get((60,0,0),
                             (4, dim['XDim:RedBand'], dim['YDim:RedBand']),
                             (1, 1, 1)
                             ).reshape([2048, 2048])
        
        data_r[data_r == 65515] = 0 # PADDING
        
        conv_factor_ds = sd.select('RedConversionFactor')
        dim         = conv_factor_ds.dimensions()
        conv_factor = conv_factor_ds.get((60,0,0),
                                         (4, dim['XDim:BRF Conversion Factors'], dim['YDim:BRF Conversion Factors']),
                                         (1, 1, 1)
                                         ).reshape((32, 32))
        conv_factor[conv_factor < 0] = 0
        
        for x in range(0,data_r.shape[0],64):
          for y in range(0,data_r.shape[1],64):
            converted = np.multiply(data_r[x:x+64,y:y+64],
                                       conv_factor[x/64,y/64])
            data_r[x:x+64,y:y+64] = converted
        
        dataset  = sd.select('Blue Radiance/RDQI')
        dim      = dataset.dimensions()
        data_b = dataset.get((60,0,0),
                             (4, dim['XDim:BlueBand'], dim['YDim:BlueBand']),
                             (1, 1, 1)
                             ).reshape([2048, 2048])
        
        data_b[data_b == 65515] = 0 # PADDING
        
        conv_factor_ds = sd.select('BlueConversionFactor')
        dim         = conv_factor_ds.dimensions()
        conv_factor = conv_factor_ds.get((60,0,0),
                                         (4, dim['XDim:BRF Conversion Factors'], dim['YDim:BRF Conversion Factors']),
                                         (1, 1, 1)
                                         ).reshape((32, 32))
        conv_factor[conv_factor < 0] = 0
        
        for x in range(0,data_b.shape[0],64):
          for y in range(0,data_b.shape[1],64):
            converted = np.multiply(data_b[x:x+64,y:y+64],
                                       conv_factor[x/64,y/64])
            data_b[x:x+64,y:y+64] = converted
        
        im = np.zeros([2048, 2048, 3])
        data_r = data_r / float(data_r.max()) * 256
        data_g = data_g / float(data_g.max()) * 256
        data_b = data_b / float(data_b.max()) * 256

        im[...,0] = data_r
        im[...,1] = data_g
        im[...,2] = data_b
        im = im.astype('uint8')
        
        im[np.equal(im, 0)] = 255
        
        
        im[0:512, 64:, :] = im[0:512, :-64, :]
        im[1024:, :-64, :] = im[1024:, 64:, :]
        im[1536:, :-64, :] = im[1536:, 64:, :]
        
        isnotwhite = np.not_equal(im, 255)
        isnotwhiterows = isnotwhite.sum(1)
        isnotwhitecols = isnotwhite.sum(0)
        goodrows = [i for i in range(im.shape[0]) if isnotwhiterows[i, :].sum() > 0]
        goodcols = [i for i in range(im.shape[1]) if isnotwhitecols[i, :].sum() > 0]
        im = im[goodrows[0]:goodrows[-1], goodcols[0]:goodcols[-1], :]
        
        rgbimages.append(im)

        # Close the file
        sd.end()

        print '.',
        sys.stdout.flush()
    
    outf = open(outfile, 'w')
    print len(datestr)
    
    # Assert that the raw and AOD sequences are corresponding
    for i in range(len(datestr)):
      if datestr[i] != datestr2[i]:
        print "ERROR!  Date sequences do not align."
        print "  detected at index %d: AOD %s, raw %s" % (i, datestr[i], datestr2[i])
    
    pickle.dump((data, rgbimages, along_track, cross_track,
                 latlons, datestr), outf)
    #pickle.dump((data, along_track, cross_track,
    #             latlons, datestr), outf)
    outf.close()


################### MISR ##############
###  Analysis per time point.       ###
#######################################
class MISRDataTime(MISRData):
  # Contains code needed to load, plot, and interpret MISR data.
  # Here, each item is one time point, so plot data spaitally.

  def  __init__(self, rawdirname=None, AODdirname=None, filename=None):
    MISRData.__init__(self, rawdirname, AODdirname, filename)


  def  readin(self, filename=None):
    """readin(filename="")
    
    Read in MISR data (pickled) from filename.
    """

    super(MISRDataTime, self).readin()
    # data is pixels x timepts

    self.name   = self.name + '-time'
    self.labels = self.datestr
    # print len(self.labels)

    
  def  plot_item(self, m, ind, x, r, k, label, U, scores):
    """
    plot_item(self, m, ind, x, r, k, label, U, scores):
    
    Plot selection m (index ind, data in x) and its reconstruction r,
    with k and label to annotate of the plot.

    Also show the residual and the RGB visualization of the selected image.

    U and scores are optional; ignored in this method, used in some
    classes' submethods.
    """
    
    if x == [] or r == []: 
      print "Error: No data in x and/or r."
      return

    vmin = min(np.nanmin(x), np.nanmin(r))
    vmax = max(np.nanmax(x), np.nanmax(r))

    # Create my own color map; middle is neutral/gray, high is red, low is blue.
    cdict = {
      'red':   ((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)),
      'green': ((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)),
      'blue':  ((0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0))
      }
    cmap = matplotlib.colors.LinearSegmentedColormap('MISR_res', cdict, 256)
   
    matplotlib.rc('axes', edgecolor = 'w')
   
    pylab.figure()
    pylab.subplots_adjust(wspace=0.1,left=0)

    # FIRST SUBPLOT: original AOD data
    pylab.subplot(2,2,1)
    im = pylab.imshow(x.reshape((self.along_track,
                                 self.cross_track)),
                                 vmin=vmin, vmax=vmax)
    pylab.tick_params(\
      axis='both',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom='off',      # ticks along the bottom edge are off
      left='off',      # ticks along the left edge are off
      right='off',      # ticks along the right edge are off
      top='off',         # ticks along the top edge are off
      labelbottom='off', # labels along the bottom edge are off
      labelleft='off')   # labels along the left edge are off?
    pylab.xlabel('Original Data')

    # Voodoo required to get colorbar to be the right height.
    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    #div = make_axes_locatable(pylab.gca())
    #cax = div.append_axes('right','5%',pad='3%')
    #pylab.colorbar(im, cax=cax)
    pylab.colorbar(im)

    # SECOND SUBPLOT: reconstructed data

    pylab.subplot(2,2,2)
    im = pylab.imshow(r.reshape((self.along_track,
                                 self.cross_track)),
                                 vmin=vmin, vmax=vmax)
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
  #  div = make_axes_locatable(pylab.gca())
  #  cax = div.append_axes('right','5%',pad='3%')
  #  pylab.colorbar(im, cax=cax)
    pylab.colorbar(im)
    
    # THIRD SUBPLOT: residual data
    
    pylab.subplot(2,2,3)
    resid = x - r
    #print "Residual Min: %5.3f, Avg: %5.3f, Max: %5.3f" % (np.nanmin(resid),
    #                                                       nanmean(resid),
    #                                                       np.nanmax(resid))
    
    im = pylab.imshow(resid.reshape((self.along_track,
                                     self.cross_track)),
                                     cmap=cmap, vmin=-1, vmax=1) 
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

    # Voodoo required to get colorbar to be the right height.
#    div = make_axes_locatable(pylab.gca())
#    cax = div.append_axes('right','5%',pad='3%')
#    cbar = pylab.colorbar(im, cax=cax)
    cbar = pylab.colorbar(im)
    tickvals = np.arange(-1,1.1,0.5)
    cbar.set_ticks(tickvals)
    cbar.set_ticklabels(tickvals)

    # FOURTH SUBPLOT: actual RGB image
    
    pylab.subplot(2,2,4)
    
    im = pylab.imshow(self.rgbimages[ind]) 
    pylab.tick_params(\
      axis='both',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom='off',      # ticks along the bottom edge are off
      left='off',      # ticks along the left edge are off
      right='off',      # ticks along the right edge are off
      top='off',         # ticks along the top edge are off
      labelbottom='off', # labels along the bottom edge are off
      labelleft='off')   # labels along the left edge are off?
    pylab.xlabel('RGB view')

    # Voodoo required to get colorbar to be the right height.
#    div = make_axes_locatable(pylab.gca())
#    cax = div.append_axes('right','5%',pad='3%')
#    cbar = pylab.colorbar(im, cax=cax)

    pylab.suptitle('DEMUD selection %d (%s), item %d, using K=%d' % \
                   (m, label, ind, k))
    
    outdir = os.path.join('results', self.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    figfile = os.path.join(outdir, '%s-sel-%d-k-%d.pdf' % (self.name, m, k))
    plt.savefig(figfile, bbox_inches='tight', pad_inches=0.1)
    #print 'Wrote plot to %s' % figfile
    
