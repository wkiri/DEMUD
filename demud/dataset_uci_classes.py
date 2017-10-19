#!/usr/bin/env python
# File: dataset_uci_classes.py
# Author: James Bedell, 8/20/13
#
# Specififc details for UCI CSV data sets
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
import csv, numpy, pylab
from dataset_uci import UCIDataset

################################################################################
#
#                                    GLASS
#
################################################################################
class GlassData(UCIDataset):
  # Contains code needed to load, plot, and interpret glass (CSV) data.

  def  __init__(self, filename=None):
    """GlassData(filename="")

    Read in glass (UCI) data in CSV format from filename.
    """

    # Subset to a single class
    cl = 'headlamp'

    #UCIDataset.__init__(self, filename, "glass")
    UCIDataset.__init__(self, filename, "glass_" + cl)
    
    self.readin(1)
    self.update_labels()

    # Subset to a single class
    keep = [i for (i,l) in enumerate(self.labels) if l == cl]
    self.data   = self.data[:,keep]
    # Still annoys me that you can't index a list with a list
    self.labels = [self.labels[k] for k in keep]   


  def  update_labels(self):
    
    label_names = {'1': 'building_float',
                   '2': 'building_nonfloat',
                   '3': 'vehicle_float',
                   '4': 'vehicle_nonfloat',
                   '5': 'container',
                   '6': 'tableware',
                   '7': 'headlamp'}

    self.labels = [label_names[i] for i in self.labels]
    
    self.features = numpy.array(['R.I.', #refractive index
                                 'Na',
                                 'Mg',
                                 'Al',
                                 'Si',
                                 'K',
                                 'Ca',
                                 'Ba',
                                 'Fe'])
    
    self.xlabel = 'Element attributes of glass samples'
    self.ylabel = 'Attribute values: Refraction index or oxide prevalence'


################################################################################
#
#                                    IRIS
#
################################################################################
class IrisData(UCIDataset):
  # Contains code needed to load, plot, and interpret iris (CSV) data.

  def  __init__(self, filename=None):
    """IrisData(filename="")

    Read in iris (UCI) data in CSV format from filename.
    """

    UCIDataset.__init__(self, filename, "iris")
    
    self.readin(0)
    self.update_labels()

  def  update_labels(self):
    
    self.features = numpy.array(['sepal length',
                                 'sepal width',
                                 'petal length',
                                 'petal width'])
    
    self.xlabel = 'Element attributes of iris samples'
    self.ylabel = 'Attribute values: cm'


################################################################################
#
#                                    ECOLI
#
################################################################################
class EcoliData(UCIDataset):
  # Contains code needed to load, plot, and interpret e.coli (CSV) data.

  def  __init__(self, filename=None):
    """EcoliData(filename="")

    Read in e. coli (UCI) data in CSV format from filename.
    """

    UCIDataset.__init__(self, filename, "ecoli")
    
    self.readin(1)
    self.update_labels()
    
  def  update_labels(self):
    
    label_expand = {'cp': 'cytoplasm',
                    'im': 'inner_membrane_no_signal',
                    'pp': 'perisplasm',
                    'imU': 'inner_membrane_uncleavable',
                    'om': 'outer_membrane',
                    'omL': 'outer_membrane_lipoprotein',
                    'imL': 'inner_membrane_lipoprotein',
                    'imS': 'inner_membrane_cleavable'}
                    
    self.labels = [label_expand[i] for i in self.labels]
                    
    self.features = numpy.array(['mcg',
                                 'gvh',
                                 'lip',
                                 'chg',
                                 'aac',
                                 'ALOM-1',
                                 'ALOM-2'])
    
    self.xlabel = 'Element attributes of e. coli samples'
    self.ylabel = 'Attribute values'
    
################################################################################
#
#                                   ABALONE
#
################################################################################
class AbaloneData(UCIDataset):
  # Contains code needed to load, plot, and interpret abalone (CSV) data.

  def  __init__(self, filename=None):
    """AbaloneData(filename="")

    Read in abalone (UCI) data in CSV format from filename.
    """

    UCIDataset.__init__(self, filename, "abalone")
    
    self.readin(1)
    self.update_labels()

  def  update_labels(self):
    
    self.labels = ['%s_rings' % str(i).zfill(2) for i in self.labels]
    
    self.features = numpy.array(['leng',
                                 'diam',
                                 'hght',
                                 'weight',
                                 'meat',
                                 'gut',
                                 'shell',])
    
    self.xlabel = 'Element attributes of abalone samples'
    self.ylabel = 'Attribute values: Length or weight'


################################################################################
#
#                                   ISOLET
#
################################################################################
class IsoletData(UCIDataset):
  # Contains code needed to load, plot, and interpret isolet (CSV) data.

  def  __init__(self, filename=None):
    """AbaloneData(filename="")

    Read in ISOLET (UCI) data in CSV format from filename.
    """

    UCIDataset.__init__(self, filename, "isolet")
    
    self.readin(0)
    self.update_labels()

  def  update_labels(self):
    
    abcs = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    self.labels = [abcs[int(i[:-1])] for i in self.labels]
    
    self.features = numpy.array(['?'] * 617)
    
    self.xlabel = 'Element attributes of ISOLET samples'
    self.ylabel = 'Attribute values'


################################################################################
#
#                                    MAIN
#
################################################################################
if __name__ == "__main__":
  import doctest

  (num_failed, num_tests) = doctest.testmod()
  filename                = os.path.basename(__file__)

  if num_failed == 0:
    print "%-20s All %3d tests passed!" % (filename, num_tests)

