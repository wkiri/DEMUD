#!/usr/bin/env python
# File: dataset_libs.py
# Author: Kiri Wagstaff, 5/7/13
#
# Readers and plotters for LIBS (ChemCam) data sets
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

import os, sys, pickle, fnmatch
import pylab, csv, math, copy
import numpy as np
from dataset import *
from log import printt

################### LIBS ##############
class LIBSData(Dataset):
  # Contains code needed to load, plot, and interpret 
  # LIBS (CSV) data file(s)

  def  __init__(self, inputname=None, initfilename=None,
                startsol=-1, endsol=-1, initpriorsols=False, 
                shotnoisefilt=0):
    """LIBSData(inputname="", sol=-1)

    Read in LIBS (ChemCam) data in CSV format from inputname.
    If inputname ends in .csv, treat it as a CSV file.
    If inputname ends in .pkl, treat it as a pickled file.
    Otherwise, treat it as a directory and look for a .pkl file inside;
    if not found, generate it with contents from all .csv files present.

    If present, also read in data from initfilename (must be .csv).
    This data will be used to initialize the DEMUD model.

    Optionally, specify a sol range (startsol-endsol) for data to analyze.
    Optionally, use data prior to startsol to initialize the model.
    Optionally, specify the width of a median filter to apply.
    """

    input_type = inputname[-3:]

    if input_type == 'csv':
      filename = inputname
      expname  = 'libs-' + \
          os.path.splitext(os.path.basename(filename))[0]
      #filename[filename.rfind('/')+1:filename.find('.')]
    elif input_type == 'pkl':
      if shotnoisefilt > 0:
        #filename = inputname[:-4] + ('-snf%d.pkl' % shotnoisefilt)
        filename = os.path.splitext(inputname)[0] + \
            ('-snf%d.pkl' % shotnoisefilt)
      else:
        filename = inputname
      expname  = 'libs-' + \
          os.path.splitext(os.path.basename(filename))[0]
      #filename[filename.rfind('/')+1:filename.find('.')]
    else:  # assume directory
      input_type = 'dir'
      #filename = inputname + '/libs-mean-norm.pkl'
      filename = os.path.join(inputname, 'libs-mean-norm.pkl')
      if shotnoisefilt > 0:
        #filename = filename[:-4] + ('-snf%d.pkl' % shotnoisefilt)
        filename = os.path.splitext(inputname)[0] + \
            ('-snf%d.pkl' % shotnoisefilt)
        #expname  = 'libs-' + inputname[inputname.rfind('/')+1:]
      expname  = 'libs-' + os.path.basename(inputname)

    Dataset.__init__(self, filename, expname, initfilename)

    printt('Reading %s data from %s.' % (input_type, self.filename))

    if input_type == 'dir' and not os.path.exists(filename):
      LIBSData.read_dir(inputname, filename, shotnoisefilt)
    
    self.readin(startsol, endsol, initpriorsols, shotnoisefilt)


  def  readin(self, startsol=-1, endsol=-1, initpriorsols=False, shotnoisefilt=0):
    """readin()
    
    Read in LIBS data from self.filename.
    Read in initialization data from self.initfilename.
    Normalize according to Nina's instructions.

    Optionally, specify a sol range (startsol-endsol) for data to analyze.
    Optionally, use data prior to startsol to initialize the model.
    Optionally, specify the width of a median filter to apply.
    """

    input_type = os.path.splitext(self.filename)[1][1:]

    self.data     = []
    self.initdata = []
    self.xlabel   = 'Wavelength (nm)'
    self.ylabel   = 'Intensity'

    if input_type == 'csv':
      (self.data, self.labels) = LIBSData.read_csv_data(self.filename)

      # Prune off first column (wavelengths)
      wavelengths = self.data[:,0]
      self.xvals  = wavelengths.reshape(-1,1)
      self.data   = self.data[:,1:]  # features x samples

      (self.data, self.xvals) = \
          LIBSData.prune_and_normalize(self.data, self.xvals, shotnoisefilt)
      
      (self.data, self.labels) = self.filter_data(self.data, self.labels)

    elif input_type == 'pkl':

      inf = open(self.filename, 'r')
      (self.data, self.labels, self.xvals) = pickle.load(inf)
      inf.close()

      # Temporary: until I re-run full extraction on shiva
      use = np.where(np.logical_and(self.xvals >= 270,
                                    self.xvals < 820))[0]
      self.xvals = self.xvals[use]
      self.data  = self.data[use,:]

      (self.data, self.labels) = self.filter_data(self.data, self.labels)

    else:  # Unknown format

      printt(' Error: Unknown input type for %s; no data read in' % \
             self.filename)

    # Read in the init data file, if present
    if self.initfilename != '':
      printt('Reading initialization data set from %s' % self.initfilename)
      (self.initdata, unused_labels) = LIBSData.read_csv_data(self.initfilename)

      # Prune off first column (wavelengths)
      wavelengths = self.initdata[:,0]
      self.initdata = self.initdata[:,1:] # features x samples
      (self.initdata, unused_xvals) = \
          LIBSData.prune_and_normalize(self.initdata, wavelengths, shotnoisefilt)
      print self.initdata.shape

      (self.initdata, unused_labels) = self.filter_data(self.initdata, unused_labels)
      print self.initdata.shape

    ########## Subselect by sol, if specified ##########
    if startsol > -1 and endsol >=-1:
      printt("Analyzing data from sols %d-%d only." % (startsol, endsol))
      current_sols  = [i for (i,s) in enumerate(self.labels) \
                       if (int(s.split('_')[0][3:]) >= startsol and \
                           int(s.split('_')[0][3:]) <= endsol)]
      if initpriorsols:
        previous_sols = [i for (i,s) in enumerate(self.labels) \
                         if int(s.split('_')[0][3:]) < startsol]
        printt("Putting previous sols' (before %d) data in initialization model." % startsol)
        # Concatenate initdata with data from all previous sols
        if self.initdata != []:
          print self.initdata.shape
          print self.data[:,previous_sols].shape
          self.initdata = np.hstack((self.initdata, self.data[:,previous_sols]))
        else:
          self.initdata = self.data[:,previous_sols]

      # Prune analysis data set to only include data from the sol of interest
      self.data   = self.data[:,current_sols]
      self.labels = self.labels[current_sols]


  def  filter_data(self, data, labels):
    """filter_data(data, labels)

    Filter out bad quality data, using criteria provided by Nina Lanza:
    1) Large, broad features (don't correspond to narrow peaks)
    2) Low SNR

    For each item thus filtered, write out a plot of the data
    with an explanation:
    1) Annotate in red the large, broad feature, or
    2) Annotate in text the SNR.

    Returns updated (filtered) data and label arrays.
    """

    n = data.shape[1]

    newdata = data
    remove_ind = []

    printt("Filtering out data with large, broad features.")
    #for i in [78]: # test data gap
    #for i in [1461]: # test broad feature
    #for i in [3400]: # test broad feature
    for i in range(n):
      waves     = range(data.shape[0])
      this_data = data[waves,i]
      peak_ind  = this_data.argmax()
      peak_wave = self.xvals[waves[peak_ind]]

      # Set min peak to examine as 30% of max
      min_peak = 0.15 * this_data[peak_ind]

      # Track red_waves: indices of bands that contribute to deciding
      # to filter out this item (if indeed it is).
      # These same wavelengths will be removed from further consideration
      # regardless of filtering decision
      red_waves = []
        
      # Iterate over peaks sufficiently big to be of interest
      while this_data[peak_ind] >= min_peak:
        #print "%d) Max peak: %f nm (index %d, %f)" % (i,
        #                                              self.xvals[waves[peak_ind]],
        #                                              peak_ind,
        #                                              this_data[peak_ind])
        red_waves = [waves[peak_ind]]
        
        # Set the low value to look for (indicates nice narrow peak)
        low_value = 0.1 * this_data[peak_ind]
        
        filter_item = True # guilty until proven innocent
        # Note: band resolution/spacing is not the same for diff ranges?
        # Sweep left and right up to 400 bands (10 nm), looking for low_value
        min_wave_ind = peak_ind
        max_wave_ind = peak_ind
        for j in range(1,401):
          min_wave_ind = max(min_wave_ind-1, 0)
          max_wave_ind = min(max_wave_ind+1, len(waves)-1)
          red_waves += [waves[min_wave_ind]]
          red_waves += [waves[max_wave_ind]]

          # If there's a data gap, ignore it
          if ((self.xvals[waves[min_wave_ind]+1] -
               self.xvals[waves[min_wave_ind]]) > 1):
            min_wave_ind += 1
          if ((self.xvals[waves[max_wave_ind]] -
               self.xvals[waves[max_wave_ind]-1]) > 1):
            max_wave_ind -= 1

          # Stop if we've gone more than 10 nm
          if (((self.xvals[waves[peak_ind]] - 
                self.xvals[waves[min_wave_ind]]) > 10) or
              ((self.xvals[waves[max_wave_ind]] -
                self.xvals[waves[peak_ind]]) > 10)):
            filter_item = True
            #print '%.2f: %.2f to %.2f' % (self.xvals[waves[peak_ind]],
            #                              self.xvals[waves[min_wave_ind]],
            #                              self.xvals[waves[max_wave_ind]])
            break
          
          #print 'checking %f, %f' % (self.xvals[waves[min_wave_ind]],
          #                           self.xvals[waves[max_wave_ind]])
          if this_data[min_wave_ind] <= low_value or \
             this_data[max_wave_ind] <= low_value:
            # success! data is good
            #print '  %f: %f' % (self.xvals[waves[min_wave_ind]],
            #                    this_data[min_wave_ind])
            #print '  %f: %f' % (self.xvals[waves[max_wave_ind]],
            #                    this_data[max_wave_ind])
            filter_item = False
            break
          
        # Remove the wavelengths we've considered
        [waves.remove(w) for w in red_waves if w in waves]

        # Filter the item out
        if filter_item:
          print "Filter item %d (%s) due to [%.2f, %.2f] nm " % (i,
                                                                 labels[i],
                                                        self.xvals[min(red_waves)],
                                                        self.xvals[max(red_waves)])
          # record it for later removal
          remove_ind += [i]

          '''
          # generate a plot, highlighting the problematic feature in red_waves
          pylab.clf()
          pylab.plot(self.xvals, data[:,i], 'k-', linewidth=1)
          pylab.plot(self.xvals[min(red_waves):max(red_waves)+1],
                     data[min(red_waves):max(red_waves)+1,i], 'r-', linewidth=1)
          pylab.xlabel(self.xlabel, fontsize=16)
          pylab.ylabel(self.ylabel, fontsize=16)
          pylab.xticks(fontsize=16)
          pylab.yticks(fontsize=16)
          pylab.title('Filtered item %d, %s' % (i, labels[i]))
          
          if not os.path.exists('filtered'):
            os.mkdir('filtered')
          pylab.savefig(os.path.join('filtered', 
                                     '%s-filtered-%d.pdf' % i))
          '''
          
          break
        
        else: # keep going
          # Update this_data to ignore previously considered wavelengths
          this_data = data[waves,i]
          peak_ind  = this_data.argmax()

    # Remove all filtered items
    newdata   = np.array([data[:,i] for i in range(data.shape[1]) \
                          if i not in remove_ind]).T
    newlabels = np.array([labels[i] for i in range(len(labels)) \
                          if i not in remove_ind])
    printt(" ... from %d to %d items (%d removed)." % (n, newdata.shape[1],
                                                      n-newdata.shape[1]))
    n = newdata.shape[1]

    printt("Filtering out low-SNR data.")

    # Filter out any item left that has a max peak value < 0.01.
    # (these are normalized probabilities now)
    remove_ind = []
    for i in range(n):
      if max(newdata[:,i]) < 0.01:
        remove_ind +=[i]

    # Remove all filtered items
    newdata   = np.array([newdata[:,i] for i in range(newdata.shape[1]) \
                          if i not in remove_ind]).T
    newlabels = np.array([newlabels[i] for i in range(len(newlabels)) \
                          if i not in remove_ind])

    print " ... from %d to %d items (%d removed)." % (n, newdata.shape[1],
                                                      n-newdata.shape[1])

    #sys.exit(0)
    
    return (newdata, newlabels)
  

  @classmethod
  def  read_dir(cls, dirname, outfile, shotnoisefilt=0):
    """read_dir(dirname, outfile)

    Read in raw LIBS data from .csv files in dirname.
    Pickle the result and save it to outfile.
    Note: does NOT update object fields.
    Follow this with a call to readin().
    """

    # First read in the target names and sol numbers.
    targets = {}
    sols    = {}
    # Location of this file is hard-coded!
    # My latest version goes through sol 707.
    metafile = 'msl_ccam_obs.csv'
    with open(os.path.join(dirname, metafile)) as f:
      datareader = csv.reader(f)
      # Index targets, sols by spacecraft clock value
      for row in datareader:
        [sol, edr_type, sclk, target] = [row[i] for i in [0,1,2,5]]
        if edr_type != 'CL5':
          continue
        prior_targets = [t for t in targets.values() if target in t]
        n_prior = len(prior_targets)
        # Add 1 so shots are indexed from 1, not 0
        targets[sclk] = target + ':%d' % (n_prior + 1)
        sols[sclk]    = sol
    print 'Read %d target names from %s.' % (len(targets), metafile)

    print 'Now reading LIBS data from %s.' % dirname

    data        = []
    labels      = []
    wavelengths = []

    files = os.listdir(dirname)
    f_ind = 0
#    for f in files[:len(files)]:
      # Select only CSV files
#      if fnmatch.fnmatch(f, 'CL5_*.csv'):
    for f in fnmatch.filter(files, 'CL5_*.csv') +  \
          fnmatch.filter(files, 'cl5_*.csv'):
      # Extract site_drive_seqid from the filename
      filename = f[f.rfind('/')+1:]
      printt(' Processing %s.' % filename)
      sclk  = filename[4:13]
      site  = filename[18:21]
      drive = filename[21:25]
      seqid = filename[29:34]
      target = targets[sclk]
      sol    = sols[sclk]
      
      # If it's a cal target, skip it
      if 'Cal Target' in target:
        print 'Skipping %s' % target
        continue
        
      #site_drive_seqid_target = '%s_%s_%s_%s' % (site, drive, seqid, target)
      #drive_sclk_target = '%s_%s_%s' % (drive, sclk, target)
      sol_sclk_target = 'Sol%s_%s_%s' % (sol, sclk, target)
      print(' Spacecraft clock %s, identifier %s.' % \
              (sclk, sol_sclk_target))

      with open(os.path.join(dirname, f), 'r') as csvfile:
        datareader = csv.reader(csvfile)

        row = datareader.next()
        while row[0][0] == '#':
          # Save the last row (comment line)
          lastrow = row
          row    = datareader.next()
        # The last comment line contains the header strings
        # starting with 'wave' or 'nm'
        mylabels = [l.strip() for l in lastrow]

        mydata = [[float(x) for x in row]]
        for row in datareader:
          # Skip over empty lines
          if row[0] == '':
            continue
          mydata += [[float(x) for x in row]]

        mydata = np.array(mydata)

        # Store the wavelengths
        waveind = [ind for ind,name in enumerate(mylabels) \
                     if 'wave' in name]
        if len(waveind) != 1:
          printt('Expected 1 match on "wave"; got %d.' % len(waveind))
          sys.exit(1)
        mywaves = mydata[:,waveind[0]]

        # Keep only the shots
        #shots = [ind for ind,name in enumerate(mylabels) \
        #         if 'shot' in name]
        # Keep only the mean
        shots = [ind for ind,name in enumerate(mylabels) \
                   if 'mean' in name]
                   #myshotnames = ['%s_%d_%s' % (site_drive_sclk_target,
                   #                      f_ind, mylabels[shot])
        myshotnames = ['%s_%s' % (sol_sclk_target, mylabels[shot])
                       for shot in shots]

        mydata = mydata[:,[l for l in shots]]
        
        printt(' Read %d new items, %d features.' % mydata.shape[::-1])

        if wavelengths != [] and np.any(wavelengths != mywaves):
          printt('Error: wavelengths in file %d do not match previous.' % f_ind)
        if f_ind == 0:
          data        = mydata
          wavelengths = mywaves
        else:
          data   = np.concatenate((data, mydata),1)
        labels  += myshotnames

        f_ind = f_ind + 1
        printt('Total so far: %d items, %d files.' % (data.shape[1], f_ind))

    print
    if data == []:
      printt('No data files found, exiting.')
      sys.exit()

    printt('Read a total of %d items, %d features.' % data.shape[::-1])

    labels  = np.array(labels)

    # Prune and normalize
    (data, wavelengths) = LIBSData.prune_and_normalize(data, wavelengths, shotnoisefilt)

    printt('Saving to %s.' % outfile)
    outf = open(outfile, 'w')
    pickle.dump((data, labels, wavelengths), outf)
    outf.close()
    print 'Done!'
    

  @classmethod
  def  prune_and_normalize(cls, data, wavelengths, shotnoisefilt):
    """prune_and_normalize(cls, data, wavelengths, shotnoisefilt)

    Subset LIBS data to only use wavelengths below 850 nm,
    set negative values to zero,
    then normalize responses for each of the three spectrometers.

    If shotnoisefilt >= 3, run a median filter on the data with width as specified.

    Return the pruned and normalized data.
    """

    print 'Pruning and normalizing the data.'

    # Only use data between 270 and 820 nm (ends are noisy)
    use = np.where(np.logical_and(wavelengths >= 270,
                                  wavelengths < 820))[0]
    # Instead of stripping these bands out now, first do shot noise
    # filtering (if desired).  Then strip the bands later.

    # If desired, apply a median filter to strip out impulsive noise
    # Note: this is slow. :)  Probably can be optimized in some awesome fashion.
    if shotnoisefilt >= 3:
      printt('Filtering out shot noise with width %d.' % shotnoisefilt)
      # Create a vector of 0's (to ignore) and 1's (to use).
      fw = np.zeros((data.shape[0],1))
      fw[use] = 1
      data = LIBSData.medfilter(data, shotnoisefilt, fw)

    # Now remove bands we do not use
    wavelengths = wavelengths[use]
    data        = data[use,:]
    
    # Replace negative values with 0
    negvals = (data < 0)
    printt('--- %d negative values.' % len(np.where(negvals)[0]))
    data[np.where(data < 0)] = 0
    printt('--- %d negative values.' % len(np.where(data<0)[0]))

    # Normalize the emission values for each of the
    # three spectrometers independently
    # Nina: VIS begins at 382.13812; VNIR starts at 473.1842
    vis_spec  = 382.13812
    vnir_spec = 473.1842
    spec1 = np.where(np.logical_and(wavelengths >= 0,
                                    wavelengths < vis_spec))[0]
    spec2 = np.where(np.logical_and(wavelengths >= vis_spec,
                                    wavelengths < vnir_spec))[0]
    spec3 = np.where(wavelengths >= vnir_spec)[0]
    for waves in [spec1, spec2, spec3]:
      data[waves,:] = data[waves,:] / np.sum(data[waves,:], axis=0)

    return (data, wavelengths)


  @classmethod
  def  medfilter(cls, data, L, fw=[]):
    """medfilter(cls, data, L)

    Filter each column of data using a window of width L.
    Replace each value with its median from the surrounding window.
    Inspired by http://staff.washington.edu/bdjwww/medfilt.py .

    Optionally, specify feature weights so they can factor in
    to the median calculation.  Any zero-valued weights make the
    median calculation ignore those items.  Values greater than
    zero are NOT weighted; they all participate normally.
    """

    if data == []: 
      print 'Error: empty data; cannot filter.'
      return data
    
    if L < 3:
      print 'Error: L (%d) is too small; minimum 3.' % L
      return data

    printt('Filtering shot noise with a width of %d (this may take some time).' % L)

    Lwing = (L-1)/2

    (d,n) = data.shape  # assume items are column vectors
    data2 = np.zeros_like(data)

    for j in range(n):
      for i in range(d):

        # Specify the range over which to compute the median
        if (i < Lwing):
          ind = range(0, i+Lwing+1)
        elif (i >= d - Lwing):
          ind = range(i-Lwing, d)
        else:
          ind = range(i-Lwing, i+Lwing+1)

        # If featureweights are specified,
        # adjust ind to only include the nonzero ones.
        if fw != []:
          # If there aren't any features with nonzero weights,
          # this won't use anything (set data value to 0)
          ind = [i for i in ind if fw[i]>0]

        # Perform the median filter.
        # If there are no valid features to use, set this point to 0
        # (it won't be used later anyway)
        if ind == []:
          data2[i, j] = 0
        else:
          data2[i, j] = np.median(data[ind, j])

    return data2


  @classmethod
  def read_csv_data(cls, filename):
    data = []
    
    with open(filename, 'r') as csvfile:
      datareader = csv.reader(csvfile)

      row = datareader.next()
      while row[0][0] == '#':
        # Save the last row (comment line)
        lastrow = row
        row    = datareader.next()
      # The last comment line contains the header strings
      # starting with 'wave' or 'nm'
      #print lastrow
      #print row
      labels = lastrow[1:]

      data = [[float(x) for x in row]]
      for row in datareader:
        # Skip over empty lines
        if row[0] == '':
          continue
        data += [[float(x) for x in row]]

    data = np.array(data)

    return (data, labels)


  def  plot_item(self, m, ind, x, r, k, label, U, rerr, feature_weights):
    """plot_item(self, m, ind, x, r, k, label, U, rerr, feature_weights)

    Plot selection m (index ind, data in x) and its reconstruction r,
    with k and label to annotate of the plot.
    Use fancy ChemCam elemental annotations.

    If feature_weights are specified, omit any 0-weighted features from the plot.
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

    pylab.clf()
    # xvals, x, and r need to be column vectors
    pylab.plot(self.xvals[goodfeat], r[goodfeat], 'r-', linewidth=0.5)
    pylab.plot(self.xvals[goodfeat], x[goodfeat], 'k-', linewidth=1)
    # Boost font sizes for axis and tick labels
    pylab.xlabel(self.xlabel, fontsize=16)
    pylab.ylabel(self.ylabel, fontsize=16)
    pylab.xticks(fontsize=16)
    pylab.yticks(fontsize=16)
    pylab.title('DEMUD selection %d (%s), item %d, using K=%d' % \
                (m, label, ind, k))

    #print 'Reading in emission bands.'
    # Read in the emission bands
    emissions = {}
    with open('LIBS-elts-RCW-NL.txt') as f:
      for line in f:
        vals = line.strip().split()
        if len(vals) < 2:
          break
        (wave, elt) = vals
        emissions[wave] = elt
      f.close()
      
    # Get unique elements
    elts = list(set(emissions.values()))
    # Generate per-element colors
    #colors = ['c','m','b','r','g','b']
    colors = ['#ff0000', '#00ff00', '#0000ff',
              '#00ffff', '#ff00ff', '#ffff00', 
              '#aa0000', '#00aa00', '#0000aa',
              '#00aaaa', '#aa00aa', '#aaaa00', 
              '#550000', '#005500', '#000055',
              '#005555', '#550055', '#555500']
    elt_color = {}
    for (i,e) in enumerate(elts):
      elt_color[e] = colors[i % len(colors)]

    # record the selection number
    outdir  = os.path.join('results', self.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    selfile = os.path.join(outdir, 'sels-%s.txt' % self.name)
    if m == 0:
      with open(selfile, 'w') as f:
        f.write('%d\n' % ind)
        f.close()
    else:
      with open(selfile, 'a') as f:
        f.write('%d\n' % ind)
        f.close()

    res = x - r
    abs_res = np.absolute(res)
    mx = abs_res.max()
    mn = abs_res.min()
    #printt('Absolute residuals: min %2.g, max %.2g.\n' % (mn, mx))
    if mn == mx and mx == 0:
      return

    sorted_abs_res = np.sort(abs_res,0)
    #frac_annotate = 0.002
    frac_annotate = 0.004
    width = 8
    min_match_nm  = 2
    num_annotate = int(math.floor(frac_annotate * len(abs_res)))
    thresh = sorted_abs_res[-num_annotate]
    #printt('Annotating top %.3f%% of residuals (%d above %.2g).' % \
    #    (frac_annotate * 100, num_annotate, thresh))

    band_ind = (np.where(abs_res >= thresh)[0]).tolist()
    for band in band_ind:
      w = float(self.xvals[band])
      [b, elt] = LIBSData.find_closest(w, emissions, min_match_nm)
      reproj = r[band]
      #printt('%.2f nm (%s): Expected %g, got %g' % (w, elt, reproj, x[band]))
      if b == -1:
        b = 1
        printt('No match for %.2f nm (%f)' % (w, r[band]))
        # Draw the triangle using gray, but don't add to legend
        pylab.fill([w-width, w+width, w],
                   [reproj,  reproj,  x[band]],
                   '0.6', zorder=1,
                   label='_nolegend_')
      else:
        if x[band] > reproj:
          sn = '+'
        else:
          sn = '-'
        pylab.fill([w-width, w+width, w],
                   [reproj,  reproj,  x[band]],
                   elt_color[elt], zorder=2,
                   label='%s%s %.2f' % (sn, elt, w))

      pylab.legend(fontsize=8)
    figfile = '%s/%s-sel-%d.pdf' % (outdir, self.name, m)
    pylab.savefig(figfile)
    print 'Wrote plot to %s' % figfile
    pylab.close()

    # I don't think this works -- hasn't been tested?
    '''
    # Plot projection into first two PCs' space
    top_two = U.T[:2,]
        
    projected = np.dot(top_two, X)
    pc1 = projected[0].tolist()
    pc2 = projected[1].tolist()
        
    projx = np.dot(top_two, x)
    x1 = [float(projx[0])]
    x2 = [float(projx[1])]
        
    maxerr = max(rerr)
    maxind = rerr.argmax()
    grays = [rerr[i] / maxerr for i in range(len(rerr))]
    zorders = [1000 * float(i) for i in grays]
    pickedcolor = grays[maxind]
        
    sc = plt.scatter(pc1, pc2, c=grays, marker='o', s=4, label='Sources', edgecolors='none', zorder=zorders, cmap=cm.jet, alpha=0.9)
    plt.scatter(x1, x2, edgecolor=cm.jet(256), c='w', marker='o', s=16, label='Selected source', zorder=[773038], linewidth=1)
    plt.xlabel('PC 1 (%5.2f%% data variance)' % float(100 * pcts[0]))
    plt.ylabel('PC 2 (%5.2f%% data variance)' % float(100 * pcts[1]))
    plt.title("Unseen data after sel. %d projected into first two PCs' space\nKID %s" % (selection, label))
        
    figfile = '%s/sel-%d-(%s)-the-cloud.pdf' % (outdir, selection, label)
    cb = plt.colorbar(sc)
    cb.set_label('Reconstruction Error')
    plt.legend()
    plt.savefig(figfile)      
    '''

  @classmethod
  def  find_closest(cls, w, emissions, min_match_nm):
    if min_match_nm == 0:
      b   = -1
      elt = 'None'
      return (b,elt)

    # Binary search for best match
    waves = emissions.keys()
    waves.sort()
    lo_b = 0
    hi_b = len(waves)
    b    = int(math.floor((hi_b + lo_b) / 2))
    while lo_b < hi_b:
      diff = float(waves[b]) - w
      ldiff = float('Inf')
      if b-1 >= 0:
        ldiff = float(waves[b-1]) - w
      rdiff = float('Inf')
      if b+1 <= len(waves)-1:
        rdiff = float(waves[b+1]) - w

      if abs(diff) < abs(ldiff) and abs(diff) < abs(rdiff):
        break

      if diff > 0: # too high
        hi_b = b
      else:  # too low
        lo_b = b
      b = int(math.floor((hi_b + lo_b) / 2))

    # Quality control: must be within match nm
    if abs(float(waves[b]) - w) > min_match_nm:
      b   = -1
      elt = 'None'
    else:
      elt = emissions[waves[b]]
      
    return (b,elt)
       

