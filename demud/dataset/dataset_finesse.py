#!/usr/bin/env python
# File: dataset_finesse.py
# Author: Kiri Wagstaff, 8/12/13
#
# Readers and plotters for FINESSE simulated data sets
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
import pylab, math
from .dataset import Dataset

################### FINESSE ##############
class FINESSEData(Dataset):
  # Contains code needed to load, plot, and interpret 
  # FINESSE simulator (CSV) data file(s)

  def  __init__(self, rawdirname=None):
    """FINESSEData(rawdirname="")

    Read in raw FINESSE simulator data from (pickled) filename.
    If it doesn't exist, read in individual files
    in rawdirname and save them out to the pickle file.
    """

    # Pick one
    #self.data_type   = 'emission'
    self.data_type   = 'transmission'

    filename = rawdirname + self.data_type + '-norm.pkl'

    Dataset.__init__(self, filename, 'finesse-%s' % self.data_type, '')

    print('Reading %s data from %s.' % (self.data_type, self.filename))

    if not os.path.exists(filename):
      FINESSEData.read_dir(rawdirname, filename, self.data_type)
    
    self.readin()


  def  readin(self):
    """readin()

    Read in FINESSE data (pickled) from self.filename.
    """

    inf = open(self.filename, 'r')
    (self.data, self.labels, wavelengths) = pickle.load(inf)
    inf.close()

    self.xlabel = 'Wavelength'
    if self.data_type == 'emission':
      self.ylabel = 'Fp/Fstar (normalized)'
    else:
      self.ylabel = '(Rp/Rstar)^2 (normalized)'
    self.xvals  = wavelengths


  @classmethod
  def  read_dir(cls, rawdirname, outfile, data_type):
    """read_dir(rawdirname, outfile, data_type)

    Read in raw FINESSE simulator data from .dat files in rawdirname.
    Only use the files of specified data_type
      (emission or transmission).
    Normalize according to Pieter's instructions.
    Pickle the result and save it to outfile.
    Note: does NOT update object fields.
    Follow this with a call to readin().
    """

    print('Reading FINESSE data from %s.' % rawdirname)

    # Read in the normalization coefficients
    norm_fn     = os.path.join(rawdirname, 'scalings_forKiri.csv')
    norm_data   = np.genfromtxt(norm_fn, delimiter=',',
                                   names = True,
                                   converters = {"file_identifier":
                                                 lambda s: int(s.strip('"'))})
    norm_data = np.array(norm_data)
    
    data        = []
    labels      = []
    wavelengths = []

    files = os.listdir(rawdirname)
    for f in files:
      # Select only files of the specified type (emission or transmission)
      if fnmatch.fnmatch(f, '*%s.dat' % data_type):
        filename = rawdirname + f

        # Skip the first (header) line
        planet_data = np.genfromtxt(filename, skip_header=1)
        # planet_data is now 4649 wavelengths by 2 (wavelengths, Fp/Fstar)

        # Save the wavelengths, if needed
        # (note: we ASSUME this is the same for all planets!)
        if len(wavelengths) == 0:
          wavelengths = planet_data[:,0]

        pd = planet_data[:,1].reshape((-1,1))

        # Find the normalization entry that matches this id
        file_id = int(f[0:10])
        ind = np.where(norm_data['file_identifier'] == file_id)
        if len(ind) == 0:
          print('Error: could not find norm constants for file id %d' % file_id)
          break
        else:
          ind = ind[0]

        ############## Normalization
        # For transmission spectra:
        # Subtract the 'trans B' component from the transmission spectra,
        # then normalize with (divide by) the 'trans A' component.
        # The peak to peak of the spectral features will be in the
        # 5 - 7 range.
        if data_type == 'transmission':
          pd = (pd - norm_data['trans_B'][ind]) / norm_data['trans_A'][ind]

        # For emission spectra:
        # Compute the blackbody ratio: BB(Tplanet)/BB(TSTAR) and
        # multiply this ratio with the 'emission A' scale factor.
        else:
          pd = pd / (norm_data['Tplanet_K'][ind]/norm_data['Tstar_K'][ind] * \
                     norm_data['emission_A'][ind])
           
        source_label = f[0:f.rfind('_')]
        if float(source_label) in (503,504,505):
          truelabel = 'interesting' #interesting
        else:
          truelabel = 'not' #not interesting
          
        if len(data) == 0:
          data   = pd
          labels = [truelabel]
        else:
          data   = np.concatenate((data, pd), 1)
          labels = labels + [truelabel]

        sys.stdout.write('.')
        sys.stdout.flush()

    print
    # Data is now d x n, where d = # wavelengths and n = # planets 
    print('Read data set with %d wavelengths, %d planets.' % data.shape)
    labels = np.array(labels)

    print('Saving to %s.' % outfile)
    outf = open(outfile, 'w')
    pickle.dump((data, labels, wavelengths), outf)
    outf.close()
    print('Done!')


  def  plot_item(self, m, ind, x, r, k, label):
    """plot_item(self, m, ind, x, r, k, label)

    Plot selection m (index ind, data in x) and its reconstruction r,
    with k and label to annotate of the plot.
    """
    
    if len(x) == 0 or len(r) == 0:
      print("Error: No data in x and/or r.")
      return
  
    pylab.clf()
    # xvals, x, and r need to be column vectors
    pylab.plot(self.xvals, r, color='0.6', label='Expected')
    # Color code:
    # positive residuals = red
    # negative residuals = blue
    pylab.plot(self.xvals, x, color='0.0', label='Observed')
    posres = np.where((x-r) > 0)[0]
    negres = np.where((x-r) < 0)[0]
    pylab.plot(self.xvals[posres], x[posres], 'r.', markersize=3, label='Higher')
    pylab.plot(self.xvals[negres], x[negres], 'b.', markersize=3, label='Lower')

    pylab.xlabel(self.xlabel)
    pylab.ylabel(self.ylabel)
    pylab.title('DEMUD selection %d (%s), item %d, using K=%d' % \
                (m, label, ind, k))
    pylab.legend() #fontsize=10)
  
    outdir = os.path.join('results', self.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    figfile = os.path.join(outdir, 'sel-%d-k-%d-(%s).png' % (m, k, label))
    pylab.savefig(figfile)
    print('Wrote plot to %s' % figfile)
    pylab.close()
