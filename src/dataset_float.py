#!/usr/bin/env python
# File: dataset_float.py
# Author: Kiri Wagstaff, 9/24/14
#
# Superclass for unlabeled data sets with floating point values:
# readers and plotters
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
from dataset import Dataset

class FloatDataset(Dataset):
  # Supersubclass for data sets

  def __init__(self, filename=None, name='Floatdata'):
    """Dataset(filename="", name="") -> Dataset
    
    Creates a new Dataset based on the data in filename
    and with the name given.
    Name is used in output filename prefixes.
    Assumes CSV, floating point values, and no class labels.
    Commented lines (with #) are ignored.

    This top-level class can't do anything itself,
    but needs to be instantiated with one of its subclasses.
    """

    Dataset.__init__(self, filename, name, '')


  def  readin(self, nskip):
    """readin()
    """

    with open(self.filename, 'r') as csvfile:
      lines = csvfile.readlines()

      for line in lines:
        # Skip over empty lines
        if line.strip() == '' or line[0] == '#':
          continue
        attributes = re.split(',* *', line.strip())

        self.data += [[float(x) for x in attributes[nskip:]]]
        if nskip > 0:
          self.labels.append(attributes[0])
        else:  # fake labels
          self.labels.append('None')


    self.data = numpy.array(self.data)

    self.data   = self.data.T  # features x samples
    
    self.xvals  = numpy.arange(self.data.shape[0]).reshape(-1,1)
  

