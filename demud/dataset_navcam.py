# Author: Anantha Ravi Kiran
# Data  : 06/04/14
# Readers and plotters for image sequence
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

import os, sys, glob, pdb, scipy, scipy.misc
import numpy as N
import cv2 as cv2
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pylab
import pickle as pickle
from dataset import *

# For color_mask_img function
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from skimage import data, color, io, img_as_float

#from mlabwrap import mlab

################## Image Sequence Data ################
class NavcamData(Dataset):
  # Containes the load, init and plot functions 
  # for sequence of image dataset - uses sol number

  _VL_SIFT_ = 0

  def __init__(self, input_folder=None, sol_number=None, init_sols=None, scaleInvariant=None):
    
    self.input_folder = None
    self.sol_number = None
    self.init_sols = None
    self.dataset_folder = os.path.join(input_folder, 'sol%.4d' % sol_number)
    self.datafiles = []
    self.init_data_folder = []
    self.data_files_count = 0
    self.img_label_split = [0]
    self.data_split = [0]
    self.selections = []
    self.init_sols = []
    self.priority = 0
    self.score = []
    self.shadow_score = []
    self.met_score = []
    self.select_rect = []
    self.rand_score = []
    self.shadow_rand_score = []
    self.met_rand_score = []
    self.rec_features = {}
    self.orig_features = {}
    self.loc = {}
    self.zoom_window = {}

    # Variables from TCData
    self.feature_string = ('dsift')
    # Used for extracting sub images to extract features
    self.winsize = 100
    self.nbins   = 101
    self.scaleInvariant = scaleInvariant
  
    if ~(input_folder is None):
        self.input_folder = input_folder
    if ~(sol_number is None):
        self.sol_number = sol_number
    if ~(init_sols is None):
        self.init_sols = init_sols
    if ~(scaleInvariant is None):
        self.scaleInvariant = scaleInvariant

    # Data folder for analysis
    print('Input Data')
    for i,data_file in enumerate(glob.glob('%s/*eff*.img.jpg'%(self.dataset_folder))):
      print(data_file)
      self.datafiles.append(data_file)
      if not scaleInvariant:
        pkl_file = data_file.split('.')[0] + '.pkl'
      else:
        pkl_file = data_file.split('.')[0] + '.si'

      if not i:
      	# Initialized for the first run and extended thereafter
        Dataset.__init__(self, pkl_file, 
                         'tc-sol%d-prior%s' % (self.sol_number, 
                                               len(self.init_sols)))
#                     pkl_file[pkl_file.rfind('/')+1:pkl_file.rfind('_')+1])
        if not scaleInvariant:
            (self.data, self.labels, feature_string, self.width, self.height, \
                self.winsize, self.nbins) = self.read_ppm(data_file, pkl_file)
        else:
            (self.data, self.labels, feature_string, self.width, self.height, \
                self.winsize, self.nbins) = self.si_read_ppm(data_file, pkl_file)
                     
        self.npixels = self.width * self.height
        self.xlabel = 'Grayscale intensity'
        self.ylabel = 'Probability'
        self.xvals  = scipy.arange(self.data.shape[0]).reshape(-1,1)   
        self.img_label_split.extend([len(self.labels)])
        self.data_split.extend([self.data.shape[1]])
        self.selections.append(N.zeros((self.height, self.width)))
        self.select_rect.append({})
        self.width      = N.array([self.width])
        self.height     = N.array([self.height])
        self.xvals      = N.array([self.xvals])
        continue
      
      if not scaleInvariant:
        extracted_features = self.read_ppm(data_file, pkl_file)
      else:
        extracted_features = self.si_read_ppm(data_file, pkl_file)
        
      self.extend(extracted_features)
      self.data_files_count = self.data_files_count + 1
      self.selections.append(N.zeros((self.height[i], self.width[i])))
      self.select_rect.append({})
      
    # Data folder for initialization
    print('Init Data Folders')
    for init_sol in init_sols:
      init_dataset_folder = os.path.join(input_folder, 'sol%.4d' % init_sol)
      print(init_dataset_folder)
      if os.path.isdir(init_dataset_folder):
        for init_data_file in glob.glob('%s/*eff*.img.jpg'%(init_dataset_folder)):
          self.initfilename = init_data_file
          if not scaleInvariant:
            init_pkl_file = init_data_file.split('.')[0] + '.pkl'
          else:
            init_pkl_file = init_data_file.split('.')[0] + '.si'

          if not scaleInvariant:
                (initdata, labels, features_string, width, height, \
                    winsize, nbins) = self.read_ppm(init_data_file, init_pkl_file)
          else:
                (initdata, labels, features_string, width, height, \
                    winsize, nbins) = self.si_read_ppm(init_data_file, init_pkl_file)

          if not len(self.initdata):
            self.initdata = initdata
          else:
            self.initdata = N.concatenate((self.initdata, initdata),axis=1)

  @classmethod
  def  extract_sift(cls, rawfilename, winsize, nbins):
    """read_ppm(rawfilename, filename)

    Read in raw pixel data from rawfilename (.ppm).
    Create a histogram around each pixel to become
    the feature vector for that obsevation (pixel).
    Pickle the result and save it to filename.
    Note: does NOT update object fields.
    Follow this with a call to readin().
    """
    if cls._VL_SIFT_:
      # VLSIFT matlab 

      im  = Image.open(rawfilename)
      (width, height) = im.size

      mlab.bb_sift(N.array(im), 'temp.mat')
      sift_features = scipy.io.loadmat('temp.mat')
      kp = sift_features['f_']
      sift_features = sift_features['d_']
      sift_features  = scipy.concatenate((sift_features.transpose(), kp[2:4].transpose()), 1).transpose()

      labels = [];
      for ikp in kp.transpose():
        (x,y) = ikp[0:2]
        labels    += ['(%d,%d)' % (y,x)]
    else:
      #Opencv SIFT 
      img = cv2.imread(rawfilename)
      gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      height, width = gray.shape

      # Computing SIFT
      sift = cv2.SIFT(edgeThreshold = 3)
      kp, des = sift.detectAndCompute(gray,None)

      labels  = []
      sift_features = N.transpose(des)
      scale_angle = []

      for ikp in kp:
        (x,y) = ikp.pt
        scale_angle.append([ikp.size/12, ikp.angle])
        labels    += ['(%d,%d)' % (y,x)]
    
      scale_angle = N.array(scale_angle)
      sift_features  = scipy.concatenate((sift_features.transpose(), scale_angle), 1).transpose()

    return (sift_features, labels, width, height)

  @classmethod
  def  extract_dsift(cls, rawfilename, winsize, nbins):
    """read_ppm(rawfilename, filename)

    Read in raw pixel data from rawfilename (.ppm).
    Create a histogram around each pixel to become
    the feature vector for that obsevation (pixel).
    Pickle the result and save it to filename.
    Note: does NOT update object fields.
    Follow this with a call to readin().
    """
    im  = Image.open(rawfilename)
    (width, height) = im.size

    # To be removed in the future
    # Pick up all windows, stepping by half of the window size
    labels  = []
    halfwin = int(winsize/2)
    for y in range(halfwin, height-halfwin, int(halfwin/2)):
      for x in range(halfwin, width-halfwin, int(halfwin/2)):
        labels    += ['(%d,%d)' % (y,x)]

    mlab.bb_dsift(N.array(im), winsize, 'temp.mat')
    sift_features = scipy.io.loadmat('temp.mat')
    sift_features = sift_features['d_']

    return (sift_features, labels, width, height)

  @classmethod
  def  extract_hist(cls, rawfilename, winsize, nbins):
    # This function extracts the histogram features from the image

    im  = Image.open(rawfilename)
    
    (width, height) = im.size
    npixels = width * height
    pix = scipy.array(im)

    # Generate one feature vector (histogram) per pixel
    #winsize = 20  # for test.pgm
    #winsize = 0  # for RGB
    halfwin = int(winsize/2)

    bins    = scipy.linspace(0, 255, nbins)

    # Only use windows that are fully populated
    mywidth  = width-winsize
    myheight = height-winsize
    #data     = scipy.zeros((nbins-1, mywidth * myheight))
    #data     = scipy.zeros((3*winsize*winsize, mywidth * myheight))
    data    = []
    labels  = []

    # Pick up all windows, stepping by half of the window size
    for y in range(halfwin, height-halfwin, int(halfwin/2)):
      for x in range(halfwin, width-halfwin, int(halfwin/2)):
        # Read in data in row-major order
        ind = (y-halfwin)*mywidth + (x-halfwin)
        #data[:,ind] = \
        #    scipy.histogram(pix[y-halfwin:y+halfwin,
        #                        x-halfwin:x+halfwin],
        #                        bins)[0]
        # Just RGB
        #data[:,ind] = pix[y,x]
        # RGB window
        #data[:,ind] = pix[y-halfwin:y+halfwin,x-halfwin:x+halfwin].flat
        hist_features = TCData.extract_hist_subimg(pix[y-halfwin:y+halfwin,x-halfwin:x+halfwin])
        if data == []:
          data = hist_features.reshape(-1,1)
        else:
          data = scipy.concatenate((data, hist_features.reshape(-1,1)),1)
        labels    += ['(%d,%d)' % (y,x)]

    return (data, labels, width, height)

  @staticmethod
  def extract_hist_subimg(sub_image):
    hist_bins = range(0,260,1)
    hist_features = N.histogram(sub_image.ravel(), hist_bins)[0]
    return hist_features
    
  def si_read_ppm(self, rawfilename, filename):
    # This function reads the ppm/jpg file and extracts the features if the 
    # features pkl file doesn't exist. It is also compatible for extension 
    # of the feauture vector and doesn't compute the already computed features

    new_feature_string = []
    updated_feature = 0
    data = N.array([], dtype=int)
    if os.path.exists(filename):
      pkl_f = open(filename, 'r')
      (data, labels, feature_string, width, height, winsize, nbins)= pickle.load(pkl_f)
      self.winsize = winsize
      self.nbins = nbins
      new_feature_string = list(feature_string)
      pkl_f.close()      

    if not new_feature_string.count('sift'):
      updated_feature = 1
      (sift_features, labels, width, height) = self.extract_sift(rawfilename, self.winsize, self.nbins)

      if data.size:
        data = scipy.concatenate((data.transpose(), sift_features.transpose()), 1).transpose()
      else:
        data = sift_features
      new_feature_string.append('sift')

    if updated_feature:
      outf = open(filename, 'w')
      pickle.dump((data, labels, new_feature_string, width, height, self.winsize, self.nbins),outf)
      outf.close()
      print('Saved data to %s.' % filename)
    
    return (data, labels, new_feature_string, width, height, self.winsize, self.nbins)

  def read_ppm(self, rawfilename, filename):
    # This function reads the ppm/jpg file and extracts the features if the 
    # features pkl file doesn't exist. It is also compatible for extension 
    # of the feauture vector and doesn't compute the already computed features

    new_feature_string = []
    updated_feature = 0
    data = N.array([], dtype=int)
    if os.path.exists(filename):
      pkl_f = open(filename, 'r')
      (data, labels, feature_string, width, height, winsize, nbins)= pickle.load(pkl_f)
      self.winsize = winsize
      self.nbins = nbins
      new_feature_string = list(feature_string)
      pkl_f.close()      

    if not new_feature_string.count('dsift'):
      updated_feature = 1
      (sift_features, labels, width, height) = self.extract_dsift(rawfilename, self.winsize, self.nbins)
      if data.size:
        data = scipy.concatenate((data.transpose(), sift_features.transpose()), 1).transpose()
      else:
        data = sift_features
      new_feature_string.append('dsift')

    if not new_feature_string.count('histogram'):
      updated_feature = 1 
      (hist_features, labels, width, height) = self.extract_hist(rawfilename, self.winsize, self.nbins)
      hist_features = hist_features/(self.winsize)
      if data.size:
        data = scipy.concatenate((data.transpose(), hist_features.transpose()), 1).transpose()
      else:
        data = hist_features
      new_feature_string.append('histogram')

    '''
    if not new_feature_string.count('position'):
      updated_feature = 1 
      
      position_features = []
      for label in labels:
        (y,x) = map(int, label.strip('()').split(','))
        position_features.append([x,y]) 
      position_features = N.array(position_features)
    
      if data.size:
        data = scipy.concatenate((data.transpose(), position_features), 1).transpose()
      else:
        data = position_features
      new_feature_string.append('position')
    '''
    if updated_feature:
      outf = open(filename, 'w')
      pickle.dump((data, labels, new_feature_string, width, height, self.winsize, self.nbins),outf)
      outf.close()
      print('Saved data to %s.' % filename)
    
    return (data, labels, new_feature_string, width, height, self.winsize, self.nbins)

  def extend(self, extracted_features):
    # This method reads the pkl files in a folder and adds them to the 
    # existing data for processing in the TCData class.


    (data, labels, feature_string, width, height, winsize, nbins) = extracted_features
    npixels = width * height

    xlabel = 'Grayscale intensity'
    ylabel = 'Probability'
    xvals  = scipy.arange(self.data.shape[0]).reshape(-1,1)
    self.data       = N.concatenate((self.data, data),axis=1) 
    self.width      = N.append(self.width, width)
    self.height     = N.append(self.height, height)
    self.xvals      = N.append(self.xvals, xvals)
    self.labels.extend(labels)
    
    self.img_label_split.extend([len(self.labels)])
    self.data_split.extend([self.data.shape[1]])

  def compute_score(self, img_idx, y, x, mask):
    " Compute the score for deck or met with idx "
    qtrwin = self.winsize/2
    if mask==0:
      mask_file = self.datafiles[img_idx].split('.')[0] + '.jpg'
    elif mask==1:
      mask_file = self.datafiles[img_idx].split('.')[0] + '.msk.jpg'
    else:
      mask_file = self.datafiles[img_idx].split('.')[0] + '.shadow.jpg'
      
    selections_pad = N.zeros((self.height[img_idx] + self.winsize, 
                              self.width[img_idx] + self.winsize))
    mask_img  = cv2.imread(mask_file, 0)
    selections_pad[qtrwin:self.height[img_idx]+qtrwin,
                   qtrwin:self.width[img_idx]+qtrwin] = mask_img
    csel_mask = selections_pad[y:y+self.winsize, x:x+self.winsize]
    
    # Matches are pixels with intensity 255, so divide by this
    # to get number of matching pixels.
    return (csel_mask.sum()/255) 

  def save_rec(self, reconst_features, ind, orig_features, k):

    img_idx = N.where(self.img_label_split > ind)[0][0] - 1
    (y,x) = map(int, self.labels[ind].strip('()').split(','))

    outdir  = os.path.join('results', self.name)
    figfile = os.path.join(outdir, 
                           '%s/%s-priority-k-%d-%d.png' % (self.name, k, img_idx))

    if figfile in self.rec_features:
      self.rec_features[figfile].append(reconst_features) 
      self.orig_features[figfile].append(orig_features) 
      self.loc[figfile].append([x,y])
    else:
      self.rec_features[figfile]= [reconst_features] 
      self.orig_features[figfile]= [orig_features] 
      self.loc[figfile] = [[x,y]]

  def  plot_item(self, m, ind, x, r, k, label):
    """plot_item(self, m, ind, x, r, k, label)

    Plot selection m (index ind, data in x) and its reconstruction r,
    with k and label to annotate the plot.
    """
    
    img_idx = N.where(self.img_label_split > ind)[0][0] - 1
    img_data_file = self.datafiles[img_idx]

    rand_ind = random.randint(0, self.img_label_split[-1])
    rand_idx = N.where(self.img_label_split > rand_ind)[0][0] - 1
    
    if x == [] or r == []: 
      print("Error: No data in x and/or r.")
      return
  
#    im = Image.fromarray(x.reshape(self.winsize, self.winsize, 3))
    outdir  = os.path.join('results', self.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
#    figfile = '%s/%s-sel-%d-k-%d.pdf' % (outdir, self.name, m, k)
#    im.save(figfile)
#    print('Wrote plot to %s' % figfile)

    # record the selections in order, at their x,y coords
    # subtract selection number from n so first sels have high values
    mywidth  = self.width[img_idx] - self.winsize
    myheight = self.height[img_idx] - self.winsize
    # set all unselected items to a value 1 less than the latest

    (y,x) = map(int, label.strip('()').split(','))
    qtrwin = self.winsize/2
    if y < qtrwin:
        y = qtrwin

    if x < qtrwin:
        x = qtrwin
      
    if y + qtrwin > mywidth:
      y = mywidth - qtrwin
    if x + qtrwin > mywidth:
      x = mywidth - qtrwin

    im  = cv2.imread(img_data_file,0)
    im1 = cv2.equalizeHist(im)
    im1 = cv2.medianBlur(im1,5)

    # Selection matrix manipulation
    #self.selections[ind/mywidth, ind%myheight] = priority
    self.priority = self.priority + 1
    self.selections[img_idx][y-qtrwin:y+qtrwin, x-qtrwin:x+qtrwin] = self.priority
    self.select_rect[img_idx][self.priority] = ((x-qtrwin, y-qtrwin), (x+qtrwin, y+qtrwin))
    figfile = os.path.join(outdir, 
                           '%s-priority-k-%d-%d.pdf' % (self.name, k, img_idx))
    figfile_jpg = os.path.join(outdir, 
                               '%s-priority-k-%d-%d.png' % (outdir, self.name, k, img_idx))
    (img_masked, cmap, num_classes)= self.color_mask_img(im1, im, self.selections[img_idx], self.select_rect[img_idx], self.priority, figfile, 0, 0)

    # Saving the masked image separately
    img_disp = plt.imshow(img_masked) 
    plt.axis('off')
    plt.savefig(figfile_jpg, bbox_inches='tight')

    self.zoom_window[len(self.score)] = im[y-qtrwin:y+qtrwin, x-qtrwin:x+qtrwin]

    # Deck mask
    score = self.compute_score(img_idx, y, x, 0) * 100.0 / self.winsize / self.winsize
    print('Deck score: %.2f%%' % score)
    self.score.append(score) 
    # Meteorite mask
    met_score = self.compute_score(img_idx, y, x, 1) * 100.0 / self.winsize / self.winsize
    print('Meteorite score: %.2f%%' % met_score)
    self.met_score.append(met_score)
    # Shadow mask
    score = self.compute_score(img_idx, y, x, 2)
    self.shadow_score.append(score) 

    # zoom pictures
    (left_top, bottom_right) = ((x-qtrwin, y-qtrwin), (x+qtrwin, y+qtrwin))
    zoom_file = os.path.join(outdir, '%d.png' % (self.priority-1))
    f, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(img_masked)
    ax1.set_title('Original Image with selected block')
    ax1.axis('off')
    ax2.imshow(im[y-qtrwin:y+qtrwin,x-qtrwin:x+qtrwin], cmap = cm.Greys_r)
    ax2.set_title('Selected Block (Filtered)')
    ax2.axis('off')
    plt.savefig(zoom_file, bbox_inches='tight')

    print('writing selection to %s/sel-%d.png' % (outdir, self.priority-1))
    scipy.misc.imsave(os.path.join(outdir, 'sel-%d.png' % (self.priority-1)),
                      im[y-qtrwin:y+qtrwin,x-qtrwin:x+qtrwin])
  

    # rand choices
    (y,x) = map(int, self.labels[rand_ind].strip('()').split(','))
    score = self.compute_score(rand_idx, y, x, 0)
    self.rand_score.append(score) 
    met_score = self.compute_score(rand_idx, y, x, 1)
    self.met_rand_score.append(met_score) 
    score = self.compute_score(rand_idx, y, x, 2)
    self.shadow_rand_score.append(score) 

  def  plot_score(self, outdir):
    # Summary scoring
    print('Avg deck score: %.2f%%' % N.mean(self.score))
    print('Avg meteorite score: %.2f%%' % N.mean(self.met_score))

    # Deck scoring technique
    pylab.clf()
    pylab.scatter(range(0,len(self.score)),self.score)
    pylab.xlabel('Iterations')
    pylab.ylabel('Score')
    pylab.title('Deck score')
    figfile = os.path.join(outdir, 'deck_score.png')
    pylab.savefig(figfile, bbox_inches='tight')
    pylab.clf()

    # Deck scoring technique
    pylab.scatter(range(0,len(self.score)),self.met_score)
    pylab.xlabel('Iterations')
    pylab.ylabel('Score')
    pylab.title('Meteorite Score')
    figfile = os.path.join(outdir, 'met_score.png')
    pylab.savefig(figfile, bbox_inches='tight')
    pylab.clf()

     # Deck scoring technique
    pylab.scatter(range(0,len(self.score)),self.rand_score)
    pylab.xlabel('Iterations')
    pylab.ylabel('Score')
    pylab.title('Random Deck Score')
    figfile = os.path.join(outdir, 'deck_rand_score.png')
    pylab.savefig(figfile, bbox_inches='tight')
    pylab.clf()

    # Deck scoring technique
    pylab.clf()
    pylab.scatter(range(0,len(self.score)),self.met_rand_score)
    pylab.xlabel('Iterations')
    pylab.ylabel('Score')
    pylab.title('Random Meteorite Score')
    figfile = os.path.join(outdir, 'met_rand_score.png')
    pylab.savefig(figfile, bbox_inches='tight')

     # Deck scoring technique
    pylab.clf()
    pylab.scatter(range(0,len(self.score)),self.shadow_score)
    pylab.xlabel('Iterations')
    pylab.ylabel('Score')
    pylab.title('Shadow overlap Score')
    figfile = os.path.join(outdir, 'shadow_score.png')
    pylab.savefig(figfile, bbox_inches='tight')

    # Deck scoring technique
    pylab.clf()
    pylab.scatter(range(0,len(self.met_score)),self.shadow_rand_score)
    pylab.xlabel('Iterations')
    pylab.ylabel('Score')
    pylab.title('Random Shadow overlap Score')
    figfile = os.path.join(outdir, 'shadow_rand_score.png')
    pylab.savefig(figfile, bbox_inches='tight')
    pylab.clf()

  @staticmethod
  def color_mask_img(img, original_img, mask, rect, idx, figfile = None, show_image = 0, hist_overlay = 0):

    alpha = 0.6

    img = img_as_float(img)
    rows, cols = img.shape

    classes = rect.keys()
    num_classes = len(classes) + 1

    # Construct a colour image to superimpose
    colors = [(1.0,1.0,1.0,1.0)]
    colors.extend(cm.jet(N.linspace(0,1,num_classes-1)[::-1]))
    norm = mpl.colors.Normalize(vmin=0, vmax=num_classes - 1)
    cmap = mpl.colors.ListedColormap(colors)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    color_mask = m.to_rgba(mask)
    color_mask = color_mask[:,:,0:3] 

    # Construct RGB version of grey-level image
    img_color = N.dstack((img, img, img))

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)

    ## Replace the hue and saturation of the original image
    ## with that of the color mask

    img_masked = color.hsv2rgb(img_hsv)
    img_masked_copy = img_masked.copy()
    if not hist_overlay:
      for i,keys in enumerate(rect):
        (left_top, bottom_right) = rect[keys]
        cv2.rectangle(img_masked, left_top, bottom_right,color=colors[i+1],thickness=3) 
    else:
      color_mask_hsv = color.rgb2hsv(color_mask)
      img_hsv[..., 0] = color_mask_hsv[..., 0]
      img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    (left_top, bottom_right) = rect[idx]
    cv2.rectangle(img_masked_copy, left_top, bottom_right,color=colors[-1],thickness=3) 

    # Width ratio is uneven because of the colorbar - image with colorbar seemed to be smaller othewise
    gs = gridspec.GridSpec(1, 2,width_ratios=[1.12,1])

    # Display image with overlayed demud output
    fig = plt.figure()
    a = fig.add_subplot(gs[0])
    a.set_title('Demud Output')
    img_disp = plt.imshow(img_masked, cmap = cmap, vmin=0, vmax=num_classes) 
    plt.setp( a.get_yticklabels(), visible=False)
    plt.setp( a.get_xticklabels(), visible=False)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("left", "8%", pad="5%")
    cax = plt.colorbar(img_disp, ticks = N.linspace(0.5,num_classes-.5, num_classes), cax = cax)
    cax.set_ticklabels(range(0,num_classes) )
    cax.ax.tick_params(labelsize=5)

    # Display original image as well
    a = fig.add_subplot(gs[1])
    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    a.set_title('Original Image')
    img_disp = plt.imshow(original_img)
    plt.setp( a.get_yticklabels(), visible=False)
    plt.setp( a.get_xticklabels(), visible=False)

    if not (figfile is None):
        plt.savefig(figfile, bbox_inches='tight')
        print('Wrote selection priority plot to %s' % figfile)

    # Display the output
    if show_image:
        plt.show()

    plt.close('all')
    return (img_masked_copy, cmap, num_classes)
