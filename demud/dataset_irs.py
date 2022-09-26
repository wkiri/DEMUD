#!/usr/bin/env python
# File: dataset_irs.py
# Author: Kiri Wagstaff, 6/24/13
#
# Data reader and plotter for IRS (Spitzer) data
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
import pyfits
import scipy, pylab, math
from scipy import signal
from dataset import *

################### ENVI ##############
class IRSData(Dataset):
  # Contains code needed to load, plot, and interpret 
  # IRS (Spitzer) data

  def  __init__(self, filename=None, datefile=None,
                modelfile=None, wavecalfile = None):
    """IRSData(filename="")

    Read in IRS (Spitzer) data from (FITS) filename.
    """

    Dataset.__init__(self, filename,
                     'irs-' + \
                       os.path.splitext(os.path.basename(filename))[0], '')
    self.datefile    = datefile
    self.modelfile   = modelfile
    self.wavecalfile = wavecalfile
    
    self.readin()


  def  readin(self):
    """readin()

    Read in IRS (Spitzer) data from (FITS) filename.
    """

    # Read in the data
    datafile = pyfits.open(self.filename)
    self.data = datafile[0].data  # time x wavelength
    datafile.close()

    # Read in the timestamps and use them as feature labels
    datefile   = pyfits.open(self.datefile)
    self.xvals = datefile[0].data
    datefile.close()

    # Read in the wavelength calibration data; use wavelengths as labels
    wavecalfile = pyfits.open(self.wavecalfile)
    self.labels = wavecalfile[0].data 
    wavecalfile.close()

    # Read in the model file showing expected signal across time
    modelfile      = pyfits.open(self.modelfile)
    self.modeldata = modelfile[0].data
    modelfile.close()

    # Detrend the data, removing (linear) spectral trends
    self.data  = scipy.signal.detrend(self.data, axis=0)
    # Detrend the data, removing (linear) temporal trends
    #self.data  = scipy.signal.detrend(self.data, axis=1)

    self.xlabel = 'Time (hjd)'

    '''
    # Change to wavelength x time, and swap the axis and data labels
    self.data = self.data.T  # wavelength x time
    # Normalize each time slice by the 12-um value?
    self.xlabel = 'Wavelength (um)'
    tmp = self.labels
    self.labels = self.xvals
    self.xvals  = tmp
    '''

    self.ylabel = 'Intensity'




