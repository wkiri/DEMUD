#!/usr/bin/env python
# File: dataset_kepler.py
# Author: James Bedell, 2013-06-24
#
# Kepler data set reader and plotter for light curves
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

import os, sys, copy
import pickle
import time as Time
import csv, pylab
import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.cm as cm
# import pylab
# import scipy.stats as ss
from dataset import Dataset
import glob, optparse

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class KeplerData(Dataset):
  
#_______________________________________________________________________________
#_______________________________________init____________________________________
#

  def __init__(self, filepath = None, extension = None):
    """KeplerData(filepath="", extension="") -> KeplerData
    
    Creates a new KeplerData object based on the data in the folder given by filepath
    and with the extension given.
    Name is used in output filename prefixes.

    """
    
    Dataset.__init__(self, None, "kepler", '')
    
    if filepath == 'keplerslc':
      filepath = '/proj/imbue/data/kepler/sources/slc/'
    if filepath == 'Q0':
      filepath = '/proj/imbue/data/kepler/Q0_public/'
      
    self.filepath  = filepath
    self.xlabel    = 'Time (hours)'
    self.ylabel    = 'Flux'
    
    self.is_fft = False
    
    #dirname = filepath if filepath[-1] != '/' else filepath[:-1]
    #self.name += "-" + dirname.split('/')[-1]
    self.name += "-" + os.path.basename(filepath)
    print(self.name)
    
    self.data      = []
    self.cadence   = []
    self.extension = extension
    
    self.archive = os.path.join(filepath, "kepler." + extension + ".pkl")
    
    # Determine if we need to preprocess the data
    if not os.path.exists(self.archive):
      KeplerData.read_kepler_dir(self, filepath, extension)
    else:
      print("Found pickle at " + self.archive)
    
    self.readin()

#_______________________________________________________________________________
#______________________________________readin___________________________________
#  
  
  def readin(self):
    """
    readin()
    """
    
    inf = open(self.archive, 'r')
    (self.data, self.time_data, self.cadence, self.labels) = pickle.load(inf)
    inf.close()
    
    # Use hours from 0 as x axis scale
    self.xvals = self.time_data - self.time_data[0]
    
    #print(self.data)
      
#_______________________________________________________________________________
#_____________________________________plot_item_________________________________
#
      
  def plot_item(self, selection, index, x, r, k, label, U, mu, S, X, pcts, rerr, drawsvd=False, drawcloud=False): 
  
    ############################################################################
    # If not an FFT:
    
    if not self.is_fft:
      if not os.path.exists('results'):
        os.mkdir('results')
      outdir = ospath.join('results', self.name)
      if not os.path.exists(outdir):
        os.mkdir(outdir)
        
        
      # LIGHT CURVE AND RECONSTRUCTION
      pylab.clf()
      pylab.plot(self.time_data, x, 'b-', label='Observations')
      pylab.plot(self.time_data, r, 'r-', label='Expected')
      pylab.xlabel('Time (days)')
      pylab.ylabel('Flux')
      pylab.title('DEMUD selection %d, item %d, using K=%d\n%s' % \
                  (selection, index, k, label))
      pylab.legend()
      
      figfile = os.path.join(outdir, 'sel-%d-(%s).pdf' % (selection, label))
      pylab.savefig(figfile)
      pylab.close()
      
      
      # SECOND PLOT: First nine PCs being plotted
      plt.close('all')

      if drawsvd:
        #assert (k == U.shape[1])
      
        m = len(pcts) # Number of nonzero vectors in U
      
        colors = ['b','g','r','c','m','y','k','#666666', 'Orange']
        c = len(colors)

        umu = np.zeros(U.T.shape)
        for i in range(min(m, c)):
          umu[i] = U.T[i] # + mu[i]
          
        if m == 0:
          umu = [U.T[i] + mu[i] for i in range(c)]
        
        zzz = c if m == 0 else min(m, c)
        for i in range(zzz):
          vector = umu[i]
          if i == 0 and m == 1:
            vector[0] -= 1
          svlabel = 'PC %d, SV %5.2e (%5.2f%%)' % (i, S[i], float(100 * pcts[i]))
          plt.plot(self.xvals, vector, color=colors[i], label=svlabel, zorder = zzz - i)
          
        pylab.xlabel('Time (days)')
        pylab.ylabel('Flux')
        pylab.title('DEMUD selection %d, item %d, using K=%d\n%s' % \
                  (selection, index, k, label))

        l = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=3, prop={'size':8})
        l.set_zorder(11)
        
        figfile = os.path.join(outdir, 'sel-%d-(%s)-PCs.pdf' % (selection, label))
        plt.savefig(figfile)
        
    
      ## THIRD AND A HALFTH SUBPLOT: residuals
      plt.close('all')
      
      plt.plot(self.time_data, x - r, 'b-', label='Residuals')
      
      pylab.xlabel('Time (days)')
      pylab.ylabel('Flux')
      pylab.title('DEMUD selection %d, item %d, using K=%d\n%s' % \
                  (selection, index, k, label))
      plt.legend()

      figfile = os.path.join(outdir, 'sel-%d-(%s)-resids.pdf' % (selection, label))
      plt.savefig(figfile)
        
        
      ## FOURTH SUBPLOT: First two PCs projection
      plt.close('all')
      
      if drawcloud:
       
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
        
        figfile = os.path.join(outdir, 'sel-%d-(%s)-the-cloud.pdf' % (selection, label))
        cb = plt.colorbar(sc)
        cb.set_label('Reconstruction Error')
        plt.legend()
        plt.savefig(figfile)      
    
        return
    
    ############################################################################
    #__________________________________________________________________________#
    #_____________________don't look at me, i'm just a box_____________________#
    #                                                                          #
    ############################################################################
    
    # If FFT:
    
    elif self.is_fft:
      
      keplerid = int(label.split(':')[0].split('-')[0])
      
      if not os.path.exists('results'):
        os.mkdir('results')
      outdir = os.path.join('results', self.name)
      if not os.path.exists(outdir):
        os.mkdir(outdir)
        
      plt.close('all')
      
      # FIRST SUBPLOT: FFT and reconstruction
      plt.close('all')
    
      plt.plot(self.time_data, x, 'b-', label='Observations')
      plt.plot(self.time_data, r, 'r-', label='Expected')
      plt.xlabel('Frequency (oscillations / day)')
      plt.ylabel('Intensity')
      plt.xscale('log')
      ax = plt.gca()
      ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
      pylab.xlim(0.005, self.time_data[-1])
      plt.title('FFT plot and model for sel. %d\nKID %s' % (selection, label))
      plt.legend()

      figfile = os.path.join(outdir, 'sel-%d-(%s)-FFT.pdf' % (selection, label))
      plt.savefig(figfile)

      # SECOND SUBPLOT: original light curve
      plt.close('all')     

      plt.plot((self.cadence_time - self.cadence_time[0]) * 24, self.flux_data.T[index], 'b-')
      plt.xlabel('Time (hours)')
      plt.ylabel('Flux')
      plt.title('Detrended light curve for sel. %d\nKID %s' % (selection, label))
      
      ymin, ymax = plt.ylim()
      d = ymax - ymin
      plt.ylim(ymin - 0.07*d, ymax + 0.07 * d)
      
      figfile = os.path.join(outdir, 'sel-%d-(%s)-light-curve.pdf' % (selection, label))
      plt.savefig(figfile)
      
      # THIRD SUBPLOT: First nine PCs being plotted
      plt.close('all')

      if drawsvd:
        #assert (k == U.shape[1])
      
        m = len(pcts) # Number of nonzero vectors in U
      
        colors = ['b','g','r','c','m','y','k','#666666', 'Orange']
        c = len(colors)

        umu = np.zeros(U.T.shape)
        for i in range(min(m, c)):
          umu[i] = U.T[i] # + mu[i]
          
        if m == 0:
          umu = [U.T[i] + mu[i] for i in range(c)]
        
        zzz = c if m == 0 else min(m, c)
        for i in range(zzz):
          vector = umu[i]
          if i == 0 and m == 1:
            vector[0] -= 1
          svlabel = 'PC %d, SV %5.2e (%5.2f%%)' % (i, S[i], float(100 * pcts[i]))
          plt.plot(self.xvals, vector, color=colors[i], label=svlabel, zorder = zzz - i)
          
        plt.xlabel('Time (hours)')
        plt.ylabel('Flux')
        plt.xscale('log')
        pylab.xlim(0.005, self.xvals[-1])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.title('Top %d PCs after sel. %d\n KID %s' % (min(m, c) if m > 0 else c, selection, label))
        l = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=3, prop={'size':8})
        l.set_zorder(11)
        figfile = os.path.join(outdir, 'sel-%d-(%s)-PCs.pdf' % (selection, label))
        plt.savefig(figfile)
        
        
      ## THIRD AND A HALFTH SUBPLOT: residuals
      plt.close('all')
      
      plt.plot(self.time_data, x - r, 'b-', label='Residuals')
      
      plt.xlabel('Frequency (oscillations / day)')
      plt.ylabel('Intensity')
      plt.xscale('log')
      ax = plt.gca()
      ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
      pylab.xlim(0.005, self.time_data[-1])
      plt.title('FFT plot and model for sel. %d\nKID %s' % (selection, label))
      plt.legend()

      figfile = os.path.join(outdir, 'sel-%d-(%s)-resids.pdf' % (selection, label))
      plt.savefig(figfile)
        
      ## FOURTH SUBPLOT: First two PCs projection
      plt.close('all')
      
      if drawcloud:
       
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
        
        figfile = os.path.join(outdir, 'sel-%d-(%s)-the-cloud.pdf' % (selection, label))
        cb = plt.colorbar(sc)
        cb.set_label('Reconstruction Error')
        plt.legend()
        plt.savefig(figfile)      
    
        return
        
    else:
      print("Fatal error: I'm too hungry for this right now.")
      
#_______________________________________________________________________________
#____________________________________fftransform________________________________
#
    
  def  fftransform(self, fancy=True, cutoff_freq=8.83):
    """fftransform(KeplerData kd)
    
      Perform the FFT of the data stored in self and return the same KeplerData
        object with the transformed data.
    """
    #fancy = False
    
    print("________________________________________")
    if fancy:
      print("          performing fancy FFT          ")
    else:
      print("             performing FFT             ")
    
    if not self.is_fft:
      self.flux_data = copy.deepcopy(self.data)
      self.cadence_time = copy.deepcopy(self.time_data)
    
    dataT = self.data.T
    
    newdata = []
    
    for j in range(len(dataT)):  
    
      pdc_data = dataT[j]
   #   print("pdc_data: ", pdc_data)
      time = self.time_data 
    
      # Ensure that no NaNs exist
      for i in range(len(pdc_data)):
          if pylab.isnan(pdc_data[i]):
            pdc_data[i] = pdc_data[i-1]
      
      # Ensure that time starts from zero
      # Time is now in days
      time = time - time[0]
      
      pdc_data = pylab.detrend_mean(pdc_data)
  
      pdc_data = pdc_data / np.std(pdc_data)
      
      # Do FFT
      n = len(pdc_data)
      if fancy:
        fft_pdc = abs(10 * np.fft.fft(pdc_data, 10 * n))   # More data points
      else:
        fft_pdc = abs(np.fft.fft(pdc_data, n))               # Normal FFT
      n = len(fft_pdc)
      timediffs = [self.cadence_time[i + 1] - self.cadence_time[i] for i in \
                                              range(len(self.cadence_time) - 1)]
      spacing = np.mean(np.array(timediffs)[np.where(~np.isnan(timediffs))])
      freq = np.fft.fftfreq(n, spacing)
      half_n = np.floor(n / 2.0)
      fft_pdc_half = (2.0 / n) * fft_pdc[:half_n]
      freq_half = freq[:half_n]
      
      # Low pass filter
      # Use frequency of 8.83: exoplanet with highest frequency is 2.207 / day
      #    Capture two harmonics
      if fancy:
        wherefq = np.where(freq_half > cutoff_freq)
        cutoff = wherefq[0][0]
        freq_half = freq_half[:cutoff]
        fft_pdc_half = fft_pdc_half[:cutoff]
      
      if newdata == []:
        newdata = np.zeros((dataT.shape[0], len(fft_pdc_half)))
      
      newdata[j] = fft_pdc_half
    #  print("newdata[%d]: " % j, newdata[j])
    #  print("newdata[0]: ", newdata[0])
    #  print("j: ", j)
    #  Time.sleep(5)
      
      
    self.time_data = freq_half
    self.xvals = freq_half
    self.data = newdata.T
    self.is_fft = True
    
#_______________________________________________________________________________
#_________________________________read_kepler_dir_______________________________
#
  
  @classmethod
  def read_kepler_dir(cls, kd, filepath, extension):
    """read_kepler_dir(filepath, extension)
    
    Read in all files with given extension in the folder given by filepath and save to a pickle.
    The pickle is called filepath/kepler.extension.pkl
    """
    
    import pyfits
    
    # GET ALL FILES WITH GIVEN EXTENSION IN FILEPATH
    files = glob.glob(str(filepath) + "*" + str(extension))
    
    print("found %d files with extension %s in %s:" % (len(files), extension, filepath))
    assert len(files) > 0
    
    numfiles = len(files)
    seen = 0
    percent = 0.0
    printed = [False for foo in range(1000)]
        
    cadence = []
    flux = []
    data = []
    labels = []
    
    for filename in files:
    
      hdulist = pyfits.open(filename)
      d = np.array([(t,cad,s,pdc,q) for  \
                       (t,corr,cad,s,s_err,bg,bg_err,pdc,pdc_err,
                        q,psf1,psf1_err,psf2,psf2_err,
                        mom1,mom1_err,mom2,mom2_err,pc1,pc2) in 
                        hdulist[1].data])
      
 #     print("Read in %s" % filename)
     
      use = ((d[:,2] > 0).nonzero())[0]
      #data = d[use,2]  # ignore all nans
      time_data = d[:,0]
      cadence_data = d[:,1]
      flux_data = d[:,2]
      pdc_data = d[:,3]
      
      keplerid = int(hdulist[0].header['KEPLERID'])
      obsmode = hdulist[0].header['OBSMODE'].split()[0]
    
      # Replace nan with predecessor (fill)
      # Need to do this to use "detrend" below
      # Temporary and breaks value of "missingmethod" in demud
      for i in range(len(flux_data)):
        if pylab.isnan(flux_data[i]):
          flux_data[i] = flux_data[i-1]
          
      for i in range(len(pdc_data)):
        if pylab.isnan(pdc_data[i]):
          pdc_data[i] = pdc_data[i-1]

      # Make sure all sources have same cadence (they should)
      if cadence == []:
        cadence = cadence_data
      assert (cadence == cadence_data).all()

      # Subtract mean from data
      pdc_data = pylab.detrend_mean(pdc_data)
      flux_data = pylab.detrend_mean(flux_data)

      flux += [[float(x) for x in pdc_data]]
      labels.append(str(keplerid) + '-' + obsmode)
      
      seen += 1
      
      # output read-in progress
      if percent < 100:
        if (round((seen / float(numfiles)) * 100, 1) >= percent) and (printed[int(percent * 10)] == False):
          print("...%3.1f%%..." % percent)
          printed[int(percent * 10)] == True
          percent = round(((seen / float(numfiles)) * 100), 1) + 0.1
    print("...100%...")
    data = np.array(flux).T
    
    # Output the pickle
    outf = open(kd.archive, 'w')
    pickle.dump((data, time_data, cadence, labels), outf)
    outf.close()
    print("Wrote pickle to " + kd.archive)
    
#_______________________________________________________________________________
#_______________________________________main____________________________________
#
    
if __name__ == '__main__':

  default_filepath = '/proj/imbue/data/kepler/sources/x10666x/'

  parser = optparse.OptionParser()
  parser.add_option("-f", "--filepath", default=default_filepath, dest="filepath",
                      help="Kepler inputs filepath", action="store")
  parser.add_option("-x", "--extension", default=".fits", dest="extension",
                      help="Kepler inputs extension", action="store")
  (options, args) = parser.parse_args()
  
  
 
  kd = KeplerData(options.filepath, options.extension)
    
