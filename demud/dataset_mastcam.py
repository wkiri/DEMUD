#!/usr/bin/env python
# File: dataset_kepler.py
# Author: James Bedell, 2014-06-27
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
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.cm as cm
import cv2
# import pylab
# import scipy.stats as ss
from dataset import Dataset
import glob, optparse
import pylab
import matplotlib

import PIL
from pprint import pprint
from PIL import Image
from pds.imageextractor import ImageExtractor
from pds.core.common import open_pds
from pds.core.parser import Parser

from skimage import color, feature
from skimage import segmentation as seg
from skimage import filter as filt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class MastcamData(Dataset):
  
#_______________________________________________________________________________
#_______________________________________init____________________________________
#

  def __init__(self, filepath='multidrcl', suffix='DRCL', extension='.IMG', lblext='.LBL', force_read=True, unit='s', feature='sh', eye='L', do_print=True, initdatadir=None, initdata=None, readintuple=None):
    """MastcamData(filepath="", extension="") -> MastcamData
    
    Creates a new MastcamData object based on the data in the folder given by filepath
    and with the extension given.
    Name is used in output filename prefixes.

    """

    Dataset.__init__(self, None, "mastcam")

    if readintuple != None:
      (self.data, self.fullimages, self.segmentation, self.labels, self.xlabel, self.ylabel, self.xvals, self.rgbdict, self.lblext) = readintuple[0:9]
      if initdata != None:
        self.initdata = initdata
        if self.initfilename != None:
          self.initfilename = initarchive
        else:
          self.initfilename = 'param'
      return
    
    if do_print: print filepath
    
    if filepath == '388':
      filepath = '/proj/imbue/data/msl-mastcam/sol388/'
      
    if filepath == 'multidrcl':
      filepath = '/proj/imbue/data/msl-mastcam/multispectral_drcl/'
      
    self.filepath  = filepath
    self.xlabel    = 'TBD'
    self.ylabel    = 'TBD'
    
    #dirname = filepath[:-1]
    #subsetname = dirname.split('/')[-1]
    subsetname = os.path.basename(filepath)
    self.name += "-" + subsetname
    if len(suffix) > 0:
      self.name += "-" + eye + '-' + suffix + '-' + unit + '-' + feature
    if do_print: print "Dataset name: " + self.name
    
    self.data      = []
    self.cadence   = []
    
    self.unit      = unit
    self.feature   = feature
    self.eye       = eye

    self.rgbdict   = {}
    self.extension = extension
    self.lblext    = lblext
    self.suffix    = suffix
    
    self.archive   = os.path.join(filepath,
                                  subsetname + eye + "_" + suffix + '_' + unit + '_' + feature + ".pkl")

    if initdata != None:
      self.initdata = initdata
      if self.initfilename != None:
        self.initfilename = initarchive
      else:
        self.initfilename = 'param'
    elif initdatadir != None:
      print "Reading in initialization data..."
      #initsubsetname = initdatadir[:-1].split('/')[-1]
      initsubsetname = os.path.basename(initdatadir)
      initarchive = os.path.join(initdatadir,
                                 initsubsetname + eye + "_" + suffix + '_' + unit + '_' + feature + ".pkl")
      if os.path.exists(initarchive):
        with open(initarchive, 'r') as f:
          self.initdata = pickle.load(f)[0]
          self.initfilename = initarchive
        print "...done!"
        print "initdata.shape:", self.initdata.shape
      else:
        print "...initialization data does not exist!"
        print "Desired pickle was: %s" % initarchive
    
    # Determine if we need to preprocess the data
    if (not os.path.exists(self.archive)) or force_read:
      self.read_mastcam_dir(filepath, suffix, unit, feature, extension, lblext, eye)
    else:
      if do_print: print "Found pickle at " + self.archive
    
    self.readin()

#_______________________________________________________________________________
#______________________________________readin___________________________________
#  
  
  def readin(self):
    """
    readin()
    """
    
    inf = open(self.archive, 'r')
    # (self.data, self.fullimages, self.segmentation, self.labels, self.xlabel, self.ylabel, self.xvals, self.rgbdict, self.lblext, self.initdata, self.initfilename) = pickle.load(inf)
    intuple = pickle.load(inf)
    inf.close()

    (self.data, self.fullimages, self.segmentation, self.labels, self.xlabel, self.ylabel, self.xvals, self.rgbdict, self.lblext) = intuple[0:9]
    if len(intuple) == 11:
      (self.initdata, self.initfilename) = intuple[9:]
   

#_______________________________________________________________________________
#______________________________________get_RGB__________________________________
#
  def get_RGB(self, label):
    print label
    labelroot = label.split('_')[0]
    
    # we have a superpixel segment, not a rectangular image window
    if '~' in label:
      segment = int(label.split('~')[1].split('.')[0])
    
    # we have a sequence and need to get the one rgb
    if '_DR' not in label and '_XX' not in label: 
      labelz = self.rgbdict[label.split('[')[0].split('_')[0]]
      if '[' in label: labelz += '[' + label.split('[')[1]
      
      # segment
      if '~' in label:
        
        img = self.fullimages[labelroot]
        segm = np.copy(self.segmentation[labelroot])
        part = np.equal(segm, segment)

        print "Chose segment %d" % segment

        # pylab.clf()
        # pylab.imshow(segm == segment)
        # pylab.show()

        segm[segm != segment] = 0
        segm[segm == segment] = 2
        segm[segm == 0] = 1
        segimg = seg.mark_boundaries(img[::4, ::4], segm[::4, ::4], color=(0, 1, 1), outline_color=None)
        # segimg = img
        # imgx = np.copy(segimg)
        imgx = np.copy(img)
        # imgx[~part] = 255 - ((255 - imgx[~part]) * 0.5)
        imgx[~part] = 255
        W = np.argwhere(part)
        
        # print W
        (ystart, xstart), (ystop, xstop) = W.min(0), W.max(0) + 1
        boundingbox = imgx[ystart:ystop, xstart:xstop]
        return (boundingbox, segimg)

      # full image
      elif len(labelz.split('[')) == 1:
        name = '_'.join(labelz.split('_')[:2])
        img = self.fullimages[labelroot]
        return (img, None)

      # window in image
      elif len(label.split('[')) == 2:
        name = '_'.join(labelz.split('[')[0].split('_')[:2])
        offsetstr = labelz.split('[')[1][:-1]
        offsetstr = offsetstr.split(', ')
        a = int(offsetstr[0].split('-')[0])
        b = int(offsetstr[0].split('-')[1])
        c = int(offsetstr[1].split('-')[0])
        d = int(offsetstr[1].split('-')[1])
        print "from label, found [%d-%d, %d-%d]" % (a, b, c, d)
        img = self.fullimages[labelroot]
        subframe = img[a:b+1, c:d+1]
        imgwithsubframe = np.ones((img.shape[0], img.shape[1]))
        imgwithsubframe[a:b+1, c:d+1] = 2
        imgwithsubframe = seg.mark_boundaries(img[::4, ::4], imgwithsubframe[::4, ::4], color=(0, 1, 1), outline_color=None)
        return (subframe, imgwithsubframe)
    
    # this is one RGB or grayscale image
    # full image
    if len(label.split('[')) == 1:
      name = '_'.join(label.split('_')[:2])
      img = self.fullimages[labelroot]
      return (img, None)
    # window in image
    elif len(label.split('[')) == 2:
      name = '_'.join(label.split('[')[0].split('_')[:2])
      offsetstr = label.split('[')[1][:-1]
      offsetstr = offsetstr.split(', ')
      a = int(offsetstr[0].split('-')[0])
      b = int(offsetstr[0].split('-')[1])
      c = int(offsetstr[1].split('-')[0])
      d = int(offsetstr[1].split('-')[1])
      print "from label, found [%d-%d, %d-%d]" % (a, b, c, d)
      (img, lbl) = self.load_image(name, self.filepath, lblext=self.lblext)
      subframe = img[a:b+1, c:d+1]
      imgwithsubframe = np.ones(img.shape)
      imgwithsubframe[a:b+1, c:d+1] = 2
      imgwithsubframe = seg.mark_boundaries(img[::4, ::4], imgwithsubframe[::4, ::4], color=(0, 1, 1), outline_color=None)
      return (subframe, imgwithsubframe)

    return None
   
#_______________________________________________________________________________
#_____________________________________plot_item_________________________________
#
      
  def plot_item(self, m, ind, x, r, k, label, U, scores, show=False):
    """plot_item(self, m, ind, x, r, k, label, U, scores)

    Plot selection m (index ind, data in x) and its reconstruction r,
    with k and label to annotate of the plot.

    U and scores are optional; ignored in this method, used in some
    classes' submethods.
    """
    
    if x == [] or r == []: 
      print "Error: No data in x and/or r."
      return
  
    (rgb_window, where_in_image) = self.get_RGB(label)
    feature = label.split('[')[0].split('~')[0].split('_')[2]
    unit = label.split('_')[1]

    if unit in ['s', 'w'] and feature in ['sh']:
      return self.plot_window_item(m, ind, x, r, k, label, U, scores, rgb_window, where_in_image)

    pylab.clf()
    # xvals, x, and r need to be column vectors
    # if feature == 'sh':
      # l = len(r) / 2
      # print "Length of r is:", len(r)
      # print r.shape
      # print "Length of self.xvals:", len(self.xvals)
      # print self.xvals.shape
      # xx = [i for i in np.asarray(self.xvals).T]
      # ry = [i for i in np.asarray(r[:l]).T]
      # xy = [i for i in np.asarray(x[:l]).T]
      # rl = [i for i in np.asarray(r[l:]).T]
      # xl = [i for i in np.asarray(x[l:]).T]

      # pylab.clf()
      # print xx
      # print ry
      # pylab.plot(xx, ry, 'r-', label='Expected')
      # pylab.errorbar(xx, ry, yerr=rl, fmt=None, ecolor='r')
      # pylab.plot(xx, xy, 'b.-', label='Observations')
      # pylab.errorbar(xx, xy, yerr=xl, fmt=None, ecolor='b')

    #   pylab.plot(self.xvals, r, 'r-', label='Expected')
    #   pylab.plot(self.xvals, x, 'b.-', label='Observations')
    # else:
    
    # print len(self.xvals)
    # print len(r)

    pylab.plot(self.xvals, r, 'r.-', label='Expected')
    pylab.plot(self.xvals, x, 'b.-', label='Observations')
    #pylab.ylim([0.0, max(1.0, x.max())])
    pylab.xlim([0.87*self.xvals.min(), 1.13*self.xvals.max()])
    
    if self.feature in ['sh']:
      if np.greater(self.data, 1.01).any():
        #print self.data.max()
        pylab.ylim(0, 366)
      else:
        pylab.ylim(0, 1)

    pylab.xlabel(self.xlabel)
    pylab.ylabel(self.ylabel)
    pylab.title('DEMUD selection %d, item %d, using K=%d\nItem name: %s' % \
                (m, ind, k, label))
    pylab.legend() #fontsize=10)

    # Plot grayscale histogram overlay for rectangular image
    if self.feature in ['gh']:
      axesloc = [.55, .46, .35, .3]
    elif self.feature in ['sh']:
      axesloc = [.15, .55, .45, .3]
    # this is an inset axes over the main axes
    a = pylab.axes(axesloc)
    # Get RGB
    

    plt = pylab.imshow(rgb_window, interpolation='nearest')
    # Make these ticks invisible
    pylab.setp(a, xticks=[], yticks=[])
      
    outdir = os.path.join('results', self.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    figfile = os.path.join(outdir, 'sel-%d-(%s).pdf' % (m, label))
    pylab.savefig(figfile)
    print 'Wrote plot to %s' % figfile
    pylab.clf()

    pylab.close("all")

#_______________________________________________________________________________
#_____________________________________plot_item_________________________________
#
      
  def plot_window_item(self, m, ind, x, r, k, label, U, scores, rgb_window, where_in_image):
    
    feature = label.split('[')[0].split('~')[0].split('_')[2]
    unit = label.split('_')[1]
    camera = label[5]

    matplotlib.rc('axes', edgecolor = 'w')
   
    fff = pylab.figure()
    pylab.subplots_adjust(wspace=0.05, hspace=0.26)

    # FIRST SUBPLOT: original AOD data
    pylab.subplot(2,2,1)
    pylab.plot(self.xvals + 1, r, 'r.-', label='Expected')
    pylab.plot(self.xvals + 1, x, 'b.-', label='Observations')
    pylab.xlim([0.87*self.xvals.min(), 1.13*self.xvals.max()])
    pylab.ylim(0, 255)
    pylab.tick_params(\
      axis='both',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
    #   bottom='off',      # ticks along the bottom edge are off
    #   left='off',      # ticks along the left edge are off
      right='off',      # ticks along the right edge are off
      top='off')         # ticks along the top edge are off
    #   labelbottom='off', # labels along the bottom edge are off
    #   labelleft='off')   # labels along the left edge are off?
    pylab.xlabel('Color Band (Wavelength)')
    pylab.ylabel('Average Intensity of Band')
    pylab.legend(prop={'size':10})

    # Voodoo required to get colorbar to be the right height.
    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    #div = make_axes_locatable(pylab.gca())
    #cax = div.append_axes('right','5%',pad='3%')
    #pylab.colorbar(im, cax=cax)

    # SECOND SUBPLOT: reconstructed data

    matplotlib.rc('axes', edgecolor = 'w')

    pylab.subplot(2,2,2)
    # im = pylab.imshow(255 * np.ones(where_in_image.shape))
    if camera == 'L':
      lbltxt = str(' Band Labels:\n\n' +
                   '1. 440 nm\n' +
                   '2. Bayer Filter Blue [460 nm]\n' +
                   '3. 525 nm\n' +
                   '4. Bayer Filter Green [540 nm]\n' +
                   '5. Bayer Filter Red [620 nm]\n' +
                   '6. 675 nm\n' +
                   '7. 750 nm\n' +
                   '8. 865 nm\n' +
                   '9. 1035 nm\n')
    elif camera == 'R':
      lbltxt = str(' Band Labels:\n\n' +
                   '1. 440 nm\n' +
                   '2. Bayer Filter Blue [460 nm]\n' +
                   '3. 525 nm\n' +
                   '4. Bayer Filter Green [540 nm]\n' +
                   '5. Bayer Filter Red [620 nm]\n' +
                   '6. 800 nm\n' +
                   '7. 905 nm\n' +
                   '8. 935 nm\n' +
                   '9. 1035 nm\n')
    else:
      print "Bad camera: %s" % camera
      lbltxt = "Bad camera."

    pylab.annotate(lbltxt, xy=(.2, .95),  xycoords='axes fraction',
                horizontalalignment='left', verticalalignment='top', size=10)

    pylab.tick_params(\
      axis='both',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom='off',      # ticks along the bottom edge are off
      left='off',      # ticks along the left edge are off
      right='off',      # ticks along the right edge are off
      top='off',         # ticks along the top edge are off
      labelbottom='off', # labels along the bottom edge are off
      labelleft='off')   # labels along the left edge are off?
    pylab.xlabel('')
  #  div = make_axes_locatable(pylab.gca())
  #  cax = div.append_axes('right','5%',pad='3%')
  #  pylab.colorbar(im, cax=cax)
    # pylab.colorbar(im)
    
    # THIRD SUBPLOT: residual data
    
    pylab.subplot(2,2,3)
    
    pylab.imshow(rgb_window, interpolation='nearest')

    pylab.tick_params(\
      axis='both',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom='off',      # ticks along the bottom edge are off
      left='off',      # ticks along the left edge are off
      right='off',      # ticks along the right edge are off
      top='off',         # ticks along the top edge are off
      labelbottom='off', # labels along the bottom edge are off
      labelleft='off')   # labels along the left edge are off?
    pylab.xlabel('Selected window')

    # Voodoo required to get colorbar to be the right height.
#    div = make_axes_locatable(pylab.gca())
#    cax = div.append_axes('right','5%',pad='3%')
#    cbar = pylab.colorbar(im, cax=cax)
    # cbar = pylab.colorbar(im)
    # tickvals = numpy.arange(-1,1.1,0.5)
    # cbar.set_ticks(tickvals)
    # cbar.set_ticklabels(tickvals)

    # FOURTH SUBPLOT: actual RGB image
    
    pylab.subplot(2,2,4)
    
    pylab.imshow(where_in_image, interpolation='nearest')

    pylab.tick_params(\
      axis='both',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom='off',      # ticks along the bottom edge are off
      left='off',      # ticks along the left edge are off
      right='off',      # ticks along the right edge are off
      top='off',         # ticks along the top edge are off
      labelbottom='off', # labels along the bottom edge are off
      labelleft='off')   # labels along the left edge are off?

    pylab.xlabel('Location within original image')

    # Voodoo required to get colorbar to be the right height.
#    div = make_axes_locatable(pylab.gca())
#    cax = div.append_axes('right','5%',pad='3%')
#    cbar = pylab.colorbar(im, cax=cax)

    pylab.suptitle('DEMUD selection %d (%s), item %d, using K=%d' % \
                   (m, label, ind, k))
    
    outdir = os.path.join('results', self.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    figfile = os.path.join(outdir, 'sel-%d-(%s).pdf' % (m, label))
    plt.savefig(figfile, bbox_inches='tight', pad_inches=0.1)
    print 'Wrote plot to %s' % figfile

    pylab.close(fff)
    pylab.close("all")


#_______________________________________________________________________________
#_________________________________read_mastcam_dir______________________________
#
  
  def read_mastcam_dir(self, filepath, suffix, unit, feature, extension = '.IMG', lblext='.LBL_label', eye='LR', margin=6):
    """read_mastcam_dir(filepath, extension)
    
    Read in all files with given extension in the folder given by filepath and save to a pickle.
    The pickle is called filepath/kepler.extension.pkl
    """
    
    if eye == 'L':
      eyez = 'ML'
    elif eye == 'R':
      eyez = 'MR'
    elif eye == 'LR':
      eyez = ''
      pass
    else:
      raise ValueError('Eye name %s is not valid!  Use L, R, or LR.' % eye)
    
    # GET ALL FILES WITH GIVEN EXTENSION IN FILEPATH
    files = sorted(glob.glob(str(filepath) + "*" + eyez + "*" + str(suffix) + "*" + str(extension)))
    fileprefixes = sorted(list(set([f.split('/')[-1][0:12] for f in files])))
    print fileprefixes
    
    print "found %d files among %d sequences with eye %s and extension %s in %s:" % (len(files), len(fileprefixes), eye, extension, filepath)
    assert len(files) > 0
    
    numfiles = len(fileprefixes)
    seen = 0
    percent = 0.0
    printed = [False for foo in range(1000)]
        
    fullimages = {}
    segmentation = {}

    data = []
    self.labels = []
    
    for fileprefix in fileprefixes:
      print "  " + fileprefix    
    
      thissequence = sorted(glob.glob(str(filepath) + fileprefix + "*" + str(suffix) + "*" + str(extension)))
      asdfghjkl = 0
      
      parser = Parser()
      seqfiltstr = ""
      dimlist = []
      for w in thissequence:
        labels = parser.parse(open_pds(w.replace(extension, lblext))) 
        filt = labels['INSTRUMENT_STATE_PARMS']['FILTER_NAME'][9]
        seqfiltstr += filt
        h = int(labels['IMAGE']['LINES'])
        w = int(labels['IMAGE']['LINE_SAMPLES'])
        dimlist.append([h, w])
        #print "    %s %s %s" % (filt, h, w)

        print "Filter name:", labels['INSTRUMENT_STATE_PARMS']['FILTER_NAME']
        
      #print seqfiltstr
      # print dimlist
      seqstocombine = []
      
      # Handle cases which appear to be several series of observations
      if len(seqfiltstr) % 7 == 0:
        for i in range(len(seqfiltstr) / 7):
          subseq = thissequence[7*i:7*i+7]
          subseqfilt = seqfiltstr[7*i:7*i+7]
          if subseqfilt == '0123456':
            cont = False
            for j in range(7*i, 7*i+7):
              if dimlist[7*i] != dimlist[j]:
                print "SIZE ERROR"
                cont = True
            if cont:
                continue
            
            seqstocombine.append(subseq)
            
          else:
            if seqfiltstr == '00112233445566':
              seq1 = [thissequence[2*i] for i in range(len(thissequence)/2)]
              seq2 = [thissequence[2*i+1] for i in range(len(thissequence)/2)]
              
              seqstocombine.append(seq1)
              seqstocombine.append(seq2)
              
              break
            else:
              print "Length multiple of 7 but bad sequence"

      # Non-7 number of observations
      else:
        for i in range(len(seqfiltstr)):
          subseq = thissequence[i:i+7]
          subseqfilt = seqfiltstr[i:i+7]
          if subseqfilt == '0123456':
            cont = False
            for j in range(i, i+7):
              if dimlist[i] != dimlist[j]:
                print "SIZE ERROR"
                cont = True
            if cont: continue
            
            seqstocombine.append(subseq)
      
      # No actual multispectral images exist, so use all RGB (sol 388)
      if len(seqstocombine) == 0 and 'sol388' in self.archive:
        seqstocombine = [[f] for f in thissequence]
              
      # Now, download each sequence with this prefix
      for subseq in seqstocombine:
        qwertyuiop = 0
        bigimage = None
        
        err = False
        # Get each image within sequence
        for filename in subseq:
          namestem = filename.split('.')[0].split('/')[-1]

          try:
            (image, lbls) = self.load_image(namestem, filepath, ext=extension, lblext=lblext)
          except ValueError as e:
            #print "An error happened while processing %s" % filename
            err = True
            break

          (h, w, b) = image.shape
          
          if b == 3:
            self.rgbdict[fileprefix + str(asdfghjkl)] = namestem
            fullimages[fileprefix + str(asdfghjkl)] = image
            #print "Stored %s to rgbdict" % (fileprefix + str(asdfghjkl))
          
          if bigimage == None and 'sol388' not in filepath:
            bigimage = np.zeros([h, w, 9], dtype='uint8')
          elif bigimage == None:
            bigimage = np.zeros([h, w, b], dtype='uint8')
            
          bigimage[:,:,qwertyuiop:qwertyuiop+b] = image

          qwertyuiop += b
          

        # Reorder images based on camera so filters are ordered
        if eye in ['L', 'R']:
          bi = np.zeros([h, w, 9], dtype='uint8')
          if eye == 'L':
            bi[:, :, 0] = bigimage[:, :, 0]
            bi[:, :, 1] = bigimage[:, :, 1]
            bi[:, :, 2] = bigimage[:, :, 2]
            bi[:, :, 3] = bigimage[:, :, 4]
            bi[:, :, 4] = bigimage[:, :, 3]
            bi[:, :, 5] = bigimage[:, :, 6]
            bi[:, :, 6] = bigimage[:, :, 5]
            bi[:, :, 7] = bigimage[:, :, 7]
            bi[:, :, 8] = bigimage[:, :, 8]
          elif eye == 'R':
            bi[:, :, 0] = bigimage[:, :, 2]
            bi[:, :, 1] = bigimage[:, :, 1]
            bi[:, :, 2] = bigimage[:, :, 0]
            bi[:, :, 3] = bigimage[:, :, 4]
            bi[:, :, 4] = bigimage[:, :, 3]
            bi[:, :, 5] = bigimage[:, :, 5]
            bi[:, :, 6] = bigimage[:, :, 6]
            bi[:, :, 7] = bigimage[:, :, 7]
            bi[:, :, 8] = bigimage[:, :, 8]
          bigimage = bi

        if err:
          print "   ...didn't load sequence.  There was an error."
          continue
          
        print "   ...loaded one sequence:", (fileprefix + str(asdfghjkl))
        
        if 'sol388' not in self.archive:
          name = fileprefix + str(asdfghjkl) + '_' + unit + '_' + feature
        else:
          name =  namestem + '_' + unit + '_' + feature

        
        (segments, segmentlabels) = self.segment_image(bigimage, unit=unit)
        segmentation[fileprefix + str(asdfghjkl)] = segments[0][1]

        for i in range(len(segments)):
          data += [[float(x) for x in self.process_image(segments[i], name + segmentlabels[i], feature=feature)]]
        
        asdfghjkl += 1
        
        ###########################################
      
      seen += 1
      
      # output read-in progress
      if percent < 100:
        if (round((seen / float(numfiles)) * 100, 1) >= percent) and (printed[int(percent * 10)] == False):
          #print "...%3.1f%%..." % percent
          printed[int(percent * 10)] == True
          percent = round(((seen / float(numfiles)) * 100), 1) + 1
    print "...100%..."
    print "Transposing data..."
    data = np.array(data).T
    self.xvals.sort()
    
    # Output the pickle
    print "Writing pickle to " + self.archive + " ..."
    outf = open(self.archive, 'w')
    pickle.dump((data, fullimages, segmentation, self.labels, self.xlabel, self.ylabel, self.xvals, self.rgbdict, self.lblext, self.initdata, self.initfilename), outf)
    outf.close()
    print "Wrote pickle to " + self.archive
    
#_______________________________________________________________________________
#____________________________________load_image_________________________________
#

  def load_image(self, name, rootdir, show=False, rgb=False, ext='.IMG', lblext='.LBL_label'):
      imgpath = rootdir + name + ext
      lblpath = rootdir + name + lblext

      parser = Parser()

      labels = parser.parse(open_pds(lblpath))

      h = int(labels['IMAGE']['LINES'])
      w = int(labels['IMAGE']['LINE_SAMPLES'])

      #print "h: %d, w: %d, type: %s" % (h, w, labels['IMAGE']['SAMPLE_TYPE'])
      if labels['IMAGE']['SAMPLE_TYPE'] == 'UNSIGNED_INTEGER':
          dt = "uint" + labels['IMAGE']['SAMPLE_BITS']
          #pprint(labels['IMAGE'])
          imgarr = np.fromfile(imgpath, dtype=dt, sep="")
          imgarr = imgarr.astype('uint8')
          
      elif labels['IMAGE']['SAMPLE_TYPE'] == 'MSB_UNSIGNED_INTEGER':
          #dt = "uint" + labels['IMAGE']['SAMPLE_BITS']
          dt = ">u" + str(int(labels['IMAGE']['SAMPLE_BITS'])/8)
          #pprint(labels['IMAGE'])
          imgarr = np.fromfile(imgpath, dtype=dt, sep="")
          #print imgarr.dtype
          imgarr = imgarr.byteswap().newbyteorder()
          #print imgarr.dtype
          #imgarr = np.bitwise_and(4095, imgarr)
          imgarr = imgarr.astype('float16')
          imgarr = imgarr / 4095 * 255
          raw_input("WARNING.  THE CODE HAS JUST CHANGED THE DATA VALUES FROM float16 TO uint8.\n"
                    " THIS WAS INITIALLY DONE FOR VISUALIZATION BUT HAS CAUSED PROBLEMS.\n"
                    "  PLEASE FIX, IF POSSIBLE!  NEEDS RE-TYPING OF ALL ARRAYS.\n"
                    "\n"
                    "ALSO, THE RADIANCE OFFSET AND SCALING FACTOR IN THE LOG FILES NEED TO BE APPLIED.\n"
                    "\n"
                    "    ...press enter if you really want to continue. ")
          imgarr = imgarr.astype('uint8')
          
      else:
          print "Error: unknown sample type: %s" % labels['IMAGE']['SAMPLE_TYPE']
          exit()

      L = len(imgarr)
      B = int(labels['IMAGE']['BANDS'])
      
      #print "%7d  %s  %s" % (B, name, labels['INSTRUMENT_STATE_PARMS']['FILTER_NAME'])
      
      if B == 1:
        #print "One band: %s" % name
        #print imgarr.shape
        pass
      X = L / B
      #print "len: %d, h*w: %d, bands: %d, div: %d" % (L, h*w, B, X)
      assert (L % B == 0)
      #print "max: %d, min: %d, mean: %d" % (max(imgarr), min(imgarr), np.mean(imgarr))
      
      if B == 3:
        img = np.zeros((h, w, B), dtype='uint8')
        for b in range(B):
          img[...,b] = imgarr[b*X:(b+1)*X].reshape([h, w])
      elif B == 1:
        img = imgarr.reshape([h,w])
        img2 = np.zeros((h, w, 1), dtype='uint8')
        img2[:,:,0] = img
        img = img2
      else:
        # should never get here
        print "Error: Your dimensionality is ridiculous."
        print "   Why does the image have %d bands?" % B
        exit(-1)
      
      if show:
        img_show = PIL.Image.fromarray(img)
        img_show.save(os.path.join('mastcam_images', '%s_temp.png' % name))
      
      return (img, labels)


#_______________________________________________________________________________
#__________________________________segment_image________________________________
#
  
  def segment_image(self, image, unit='f', windowsize=120, windowoffset=60, n_segments=None, compactness=0.75, margin=6, segment_size_pixels=16.0):
    #print image.shape
    (h, w, b) = image.shape

    if n_segments == None:
      n_segments = int((h / segment_size_pixels) * (w / segment_size_pixels))
      print "Using %d segments" % n_segments

    if unit == 'f':
      return ([image], [''])
    elif unit == 'w':
      subunits = []
      sublabels = []
      for i in np.arange(0, h-windowsize, windowoffset):
        for j in np.arange(0, w-windowsize, windowoffset):
          subframe = image[i:i+windowsize, j:j+windowsize, :]
          subunits.append(subframe)
          sublabel = '[%d-%d, %d-%d]' % (i, i+windowsize-1, j, j+windowsize-1)
          sublabels.append(sublabel)
          #pylab.clf()
          #pylab.imshow(subframe)
          #pylab.show()
          #raw_input()
      return (subunits, sublabels)
    elif unit == 's':

      segmentation = seg.slic(image, n_segments=n_segments, compactness=compactness, enforce_connectivity=True, convert2lab=False)
      segflat = segmentation.flatten()

      # Remove segments bordering the image or close to borders
      for i in sorted(list(set(segflat))):
        segm = np.equal(segmentation, i)
        xs = segm.sum(1)
        ys = segm.sum(0)
        if any(xs[:margin] > 0) or any(xs[-margin:] > 0) or any(ys[:margin] > 0) or any(ys[-margin:] > 0):
          segmentation[segm] = 0

      (segmentation, _, _) = seg.relabel_sequential(segmentation)

      segflat = segmentation.flatten()
      print "Number of segments should be: ", n_segments
      n_segments_now = len(list(set(segflat)))
      print "Number of segments actually is: ", n_segments_now
      subunits = []
      sublabels = []
      
      for i in sorted(list(set(segflat))):
        if i == 0: continue
        subunits.append((image, segmentation, i))
        sublabels.append('~' + str(i))
        
      # newim = np.zeros([image.shape[0], image.shape[1], 3])
      # newim[...,0] = image[...,0]
      # newim[...,1] = image[...,0]
      # newim[...,2] = image[...,0]
      # pylab.imshow(seg.mark_boundaries(newim, segmentation))
      # pylab.savefig('segmentation.png')
      # exit()
      # pylab.show()
      # raw_input()
        
      return (subunits, sublabels)
    else:
      print "Error: unit %s not supported" % unit
      print "Accepted units are:"
      print "  f: full image"
      print "  w: window regions within image"
      print "  s: image segments"
      exit()

#_______________________________________________________________________________
#__________________________________process_image________________________________
#

  def process_image(self, image, name, feature='gh', fftlen=12):

    if type(image) == tuple:
      return self.process_segment(image, name, feature)

    #print "...processing %s..." % name

    # Start letters: c, f, g, s
    # End letters:   d, e, h, m, t

    self.xlabel = 'Features'
    self.ylabel = 'Response values'
    
    featurevector = []
    
    ############################################################################
    #
    # BASIC EDGE AND GRADIENT THINGS
    #
    ############################################################################
    
    # Do edge detection
    if 'ce' in feature:
      thisimage = imagegauss
      imageedge = filt.canny(thisimage)
      #pylab.clf()
      #pylab.imshow(image, cmap='gray', interpolation='nearest')
      #pylab.savefig('./mastcam-edges/%s.png' % name)
    
    # Gradient directions and magnitudes
    if 'gd' in feature:
      # Do Gaussian blur a little bit to obscure minor textural variation
      SIGMA = 2
      imagegauss = filt.gaussian_filter(image, SIGMA)
      
      thisimage = imagegauss
      imagexgrad = cv2.Sobel(thisimage, cv2.CV_64F, 1, 0, ksize=5)
      imageygrad = cv2.Sobel(thisimage, cv2.CV_64F, 0, 1, ksize=5)
      imagegradmag = np.sqrt(np.add(np.square(imagexgrad), np.square(imageygrad)))
      imagegraddir = np.arctan2(-1.0 * imageygrad, imagexgrad)
    
      #pylab.clf()
      #pylab.imshow(imagegradmag, cmap='gray', interpolation='nearest')
      #pylab.colorbar()
      #pylab.savefig('./mastcam-grad/%s.png' % name)
      #pylab.clf()
      #pylab.imshow(imagegraddir, cmap='jet', interpolation='nearest')
      #pylab.colorbar()
      #pylab.savefig('./mastcam-grad/%s-dir.png' % name)

      (graddirhist, foo) = np.histogram(imagegraddir, np.arange(-math.pi, 1.0001*math.pi, 0.01562501*math.pi), weights=imagegradmag)
      graddirhist = graddirhist / float(sum(graddirhist))
      
      featurevector.extend(graddirhist) # 128-item vector, pdf
    
    
    ############################################################################
    #
    # FILTER BANKS AND SIFT
    #
    ############################################################################
    
    
    # Gabor filter bank?
    #filt.rank.gabor_filter(image, lambda)
    
    
    # Get SIFT features
    if 'sh' in feature:
      thisimage = image
    
    ############################################################################
    #
    # FFT STUFF
    #
    ############################################################################
    
    # Do FFT, truncate image to lower frequencies, and flatten into a feature vector
    if 'ft' in feature:
      thisimage = image
      imagexgrad = cv2.Sobel(thisimage, cv2.CV_64F, 1, 0, ksize=5)
      imageygrad = cv2.Sobel(thisimage, cv2.CV_64F, 0, 1, ksize=5)
      imagegradmag = np.sqrt(np.add(np.square(imagexgrad), np.square(imageygrad)))
      #fftimage = np.abs(np.fft.fftshift(np.fft.fft2(thisimage)))
      #fftimagefull = fftimage
      #fftimage = fftimage[fftimage.shape[0]/2.0-fftlen/2.0:fftimage.shape[0]/2.0+fftlen/2.0, fftimage.shape[1]/2.0-fftlen/2.0:fftimage.shape[1]/2.0+fftlen/2.0]
      fftvector = fftimage.flatten()
      fftvector = fftvector/ float(sum(fftvector))
      
      pylab.clf()
      #fftimagefull[fftimagefull.shape[0]/2-1:fftimagefull.shape[0]/2, fftimagefull.shape[1]/2-1:fftimagefull.shape[1]/2] = 0
      pylab.imshow(fftimage, cmap='gray', interpolation='nearest')
      pylab.colorbar()
      pylab.savefig(os.path.join('mastcam-fft', '/%s.png' % name))
      pylab.close()
      
      featurevector.extend(fftvector) # 144-item vector, pdf
    
    ############################################################################
    #
    # COLOR INTENSITY HISTOGRAMS
    #   these rely on color-corrected inputs which is why they are last
    #
    ############################################################################
    
    
    # Color correct dat image!
    if self.unit in ['w', 's']:
      m = np.mean(image.flatten())
      image = np.minimum(np.maximum(np.subtract(image, m - 128), 0), 255)
      #print "Now: [%d - %d - %d]" % (np.min(image), np.mean(image), np.max(image))
    
    # Get color histograms
    if 'ch' in feature:
      (redhist, foo) = np.histogram(image[...,0].flatten(), np.arange(0,257,2))
      redhist = redhist / float(sum(redhist))
      (greenhist, foo) = np.histogram(image[...,1].flatten(), np.arange(0,257,2))
      greenhist = greenhist / float(sum(greenhist))
      (bluehist, foo) = np.histogram(image[...,2].flatten(), np.arange(0,257,2))
      bluehist = bluehist / float(sum(bluehist))
      
      featurevector.extend(redhist) # 128-item vector, pdf
      featurevector.extend(greenhist) # 128-item vector, pdf
      featurevector.extend(bluehist) # 128-item vector, pdf
    
    # Now make it grayscale
    #image = self.collapse_RGB(image, name)

    # Get grayscale histogram
    if 'gh' in feature:
      thisimage = self.collapse_RGB(image, name)
      (grayhist, foo) = np.histogram(thisimage.flatten(), np.arange(0,257,1))
      grayhist = grayhist / float(sum(grayhist))
      
      featurevector.extend(grayhist) # 128-item vector, pdf
    
    # Get multispectral histogram
    if 'sh' in feature:
      B = image.shape[2]
      k = 1
      hist = np.zeros(k*B)
      stds = np.zeros(B)
      for b in np.arange(0, B*k, k):
        if k == 1:
          hist[b] = np.mean(image[...,b].flatten())
        else:
          thishist = np.histogram(image[...,b/k].flatten(), np.arange(0,257,256.0/k))[0]
          thishist = thishist / float(sum(thishist))
          hist[b:b+k] = thishist
        stds[b] = np.std(image[...,b].flatten())
      
      # if k == 1:
      #   hist = np.minimum(np.maximum(np.subtract(hist, np.mean(hist) - 128), 0), 255)
      
      #print len(hist)
      #print hist
      
      featurevector.extend(hist)
      # featurevector.extend(stds)

    if 'sh' in feature:
      if name[5] == 'L':
        self.xvals = np.array([460, 540, 620, 440, 525, 675, 750, 865, 1035])
      elif name[5] == 'R':
        self.xvals = np.array([460, 540, 620, 440, 525, 800, 905, 935, 1035])
      else:
        print "name[5] is actually", name[5]
        self.xvals = np.array(range(0, len(featurevector)))

    else:
      self.xvals = np.array(range(0, len(featurevector)))
    
    self.labels.append(name)
    
    args = np.argsort(self.xvals)

    return [featurevector[a] for a in args]

#_______________________________________________________________________________
#__________________________________process_segment________________________________
#

  def process_segment(self, image, name, feature='gh'):
    try:
      segno = int(name.split('~')[-1])
      if segno % 25 == 0:
        print name
    except:
      pass
    img = image[0]
    segmentation = image[1]
    whichseg = image[2]
    
    seg = np.equal(segmentation, whichseg)
    (h, w, b) = image[0].shape
    
    flat = []
    for i in range(h):
      for j in range(w):
        if seg[i,j]:
          flat.append(img[i,j,:])

    flat = np.asarray(flat)
    # print flat.shape
    flatgray = flat.sum(1) / float(b)

    featurevector = []
  
    # Grayscale histogram
    if feature == 'gh':
      (grayhist, foo) = np.histogram(flatgray, np.arange(0,257,1))
      grayhist = grayhist / float(sum(grayhist))
      
      featurevector.extend(grayhist) # 128-item vector, pdf

    # Spectral band intensity
    if feature == 'sh':
      avgs = flat.sum(0) / float(flat.shape[0])
      stds = flat.std(0)

      featurevector.extend(avgs)
      # featurevector.extend(stds)

    # Return values and cleanup      
    # self.xvals  = np.array(range(0, len(featurevector) / 2))
    if 'sh' in feature:
      if name[5] == 'L':
        self.xvals = np.array([460, 540, 620, 440, 525, 675, 750, 865, 1035])
      elif name[5] == 'R':
        self.xvals = np.array([460, 540, 620, 440, 525, 800, 905, 935, 1035])
      else:
        self.xvals = np.array(range(0, len(featurevector)))

    else:
      self.xvals = np.array(range(0, len(featurevector)))
    
    self.labels.append(name)
    
    args = np.argsort(self.xvals)

    return [featurevector[a] for a in args]

#_______________________________________________________________________________
#__________________________________collapse_RGB_________________________________
#
  
  def collapse_RGB(self, image, name):
    if len(image.shape) == 3:
      if image.shape[2] == 1:
        #print "  this image (%s)\n   is only one channel as of reaching process_image" % name
        pass
      elif image.shape[2] == 3:
        #image = image.astype('uint8')
        #img_show = PIL.Image.fromarray(image[...,0])
        #img_show.save('./mastcam_images/%s_bandR.png' % name)
        #img_show = PIL.Image.fromarray(image[...,1])
        #img_show.save('./mastcam_images/%s_bandG.png' % name)
        #img_show = PIL.Image.fromarray(image[...,2])
        #img_show.save('./mastcam_images/%s_bandB.png' % name)
        image = image.astype('uint16')
        image = image[...,0] + image[...,1] + image[...,2]
        image = image / 3
        image = image.astype('uint8')
        #img_show = PIL.Image.fromarray(image)
        #img_show.save('./mastcam_images/%s_rgb.png' % name)
      elif image.shape[2] == 7:
        image = image.astype('uint16')
        image = image[...,0] + image[...,1] + image[...,2] + image[...,3] + image[...,4] + image[...,5] + image[...,6]
        image = image / 7
        image = image.astype('uint8')
      else:
        print "This image has %d color bands and that's weird." % image.shape[2]
        exit()
    elif len(image.shape) == 2:
      print "This image has only one channel!"
      pass
    else:
      print "This image has %d dimensions and that's weird." % len(image.shape)
      exit()
      
    return image

#_______________________________________________________________________________
#_______________________________________main____________________________________
#
    
if __name__ == '__main__':

  #
  #    \\\|||///
  # ,-. ___ ___ .-,
  #  \  (0) (0)  /
  #   '    ^    '
  #      \___/
  #

  default_filepath = '/proj/imbue/data/kepler/sources/x10666x/'

  parser = optparse.OptionParser()
  parser.add_option("-f", "--filepath", default=default_filepath, dest="filepath",
                      help="Kepler inputs filepath", action="store")
  parser.add_option("-x", "--extension", default=".fits", dest="extension",
                      help="Kepler inputs extension", action="store")
  (options, args) = parser.parse_args()
  
  
 
  kd = KeplerData(options.filepath, options.extension)
    
