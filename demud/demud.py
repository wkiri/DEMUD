#!/usr/bin/env python
# File: demud.py
# Author: Kiri Wagstaff, 2/28/13; James Bedell, summer 2013
#
# Implementation of DEMUD (Discovery through Eigenbasis Modeling of
# Uninteresting Data).  See Wagstaff et al., AAAI 2013.
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

import sys, os
import numpy as np
from numpy import linalg
from numpy import nanmean
import math
import copy, base64, time
import csv
import pylab

from dataset_uci_classes import GlassData, IrisData, EcoliData, AbaloneData, IsoletData
from dataset_float import FloatDataset
from dataset_float_classes import *
#from dataset_decals import DECaLSData
from dataset_des import DESData
#from dataset_gbtfil import GBTFilterbankData
#from dataset_misr import MISRDataTime
#from dataset_libs import LIBSData
#from dataset_finesse import FINESSEData
from dataset_envi import ENVIData
#from dataset_envi import SegENVIData
#from dataset_irs  import IRSData
#from dataset_kepler import KeplerData
#from dataset_mastcam import MastcamData
#from dataset_tc import TCData
#from dataset_navcam import NavcamData
#from dataset_images import ImageData
#from exoplanet_lookup import ExoplanetLookup
#import kepler_lookup
import log
from log import printt

#from PIL import Image
import pickle

import optparse
from optparse import *

__VERSION__ = "1.7" # Adds compatibility for Windows filenames

default_k_values = {}
default_n_value = 10
use_max_n = False


#______________________________score_items_missing__________________________
def  compute_error_with_missing(X, U, mu):
  """compute_error_with_missing(X, U, mu, missingmethod):

  Calculate the score (reconstruction error) for every item in X,
  with respect to the SVD model in U and mean mu for uninteresting items,
  when there could be missing values in X (indicated with NaN).

  If an item contains entirely NaNs (no good values),
  its error will be 0.  Maybe should be NaN instead?

  Return an array of item reconstruction errors and their reprojections.
  """

  # We want to ignore (work around) NaNs, without imputing.
  # This is less efficient than with no NaNs:
  # we have to process each item individually
  # since they might have different missing values.
  #diagS  = np.diag(S).reshape(len(S), len(S))
  reproj = np.zeros(X.shape) * np.nan
  err    = np.zeros(X.shape)
  
  for i in range(X.shape[1]):
    x        = X[:,i].reshape(-1, 1)
    # Usable features are not NaN in x nor in mu
    isgood   = ~np.logical_or(np.isnan(x), np.isnan(mu))
    goodinds = np.where(isgood)[0]
    numgood  = len(goodinds)

    if numgood == 0:  # No good data!  Do nothing (err for this item is 0)
      pass

    elif numgood == x.shape[0]:  # All good -- normal processing.
      proj        = np.dot(U.T, x - mu)
      reproj[:,i] = (np.dot(U, proj) + mu).squeeze()
      err[:,i]    = x.squeeze() - reproj[:,i]

    else:
      # Imputation/modeling method from Brand 2002
      # X = U*(S*(((U*S)+)*X)) (eqn 11)
      # Should we be using S?  We aren't at the moment.
      # Selectively use/fill only goodinds:
      proj = np.dot(U[goodinds,:].T,
                    x[goodinds,0] - mu[goodinds,0])
      reproj[goodinds,i] = np.dot(U[goodinds,:], proj) + mu[goodinds,0]
      err[goodinds,i]    = x[goodinds,0] - reproj[goodinds,i]

  return (err, reproj)


#______________________________score_items_________________________________
def  score_items(X, U, mu,
                 scoremethod='lowhigh',
                 missingmethod='none',
                 feature_weights=[]):
  """score_items(X, U, scoremethod, missingmethod, feature_weights)

  Calculate the score (reconstruction error) for every item in X,
  with respect to the SVD model in U and mean mu for uninteresting items.

  'scoremethod' indicates which residual values count towards
  the interestingness score of each item:
  - 'low': negative residuals
  - 'high': positive residuals
  - 'lowhigh': both

  'missingmethod' indicates how to handle missing (NaN) values:
  - 'zero': set missing values to zero
  - 'ignore': ignore missing values following Brand (2002)
  - 'none': assert nothing is missing (NaN).  Die horribly if not true.

  'feature_weights' influence how much each feature contributes to the score.

  Return an array of item reconstruction scores and their reprojections.
  """

  # Use U to model and then reconstruct the data in X.
  # 1. Project all data in X into space defined by U,
  #    then reconstruct it.
  if missingmethod.lower() != 'ignore':
    # All missing values should have been replaced with 0,
    # or non-existent.
    # 1a. Subtract the mean and project onto U
    proj   = np.dot(U.T, (X - mu))
    # 1b. Reconstruct by projecting back up and adding mean
    reproj = np.dot(U, proj) + mu
    # 1c. Compute the residual
    err    = X - reproj
    
  else:
    # Missing method must be 'ignore' (Brand 2002)
    (err, reproj) = compute_error_with_missing(X, U, mu)

  # 2. Compute reconstruction error
  if scoremethod == 'low':    # Blank out all errors > 0
    err[err>0] = 0
  elif scoremethod == 'high': # Blank out all errors < 0
    err[err<0] = 0
  else: # default, count everything
    pass
  
  # Weight features if requested
  if feature_weights != []:
    for i in range(len(feature_weights)):
      err[i,:] = err[i,:] * feature_weights[i]

  if missingmethod.lower() == 'ignore':
    # Only tally error for observed features.
    # This means that items with missing values are not penalized
    # for those features, which is probably the best we can do.
    scores = np.nansum(np.array(np.power(err, 2)), axis=0)
  else:
    scores = np.sum(np.array(np.power(err, 2)), axis=0)

  return (scores, reproj)
  

#______________________________select_next_________________________________
def  select_next(X, U, mu,
                 scoremethod='lowhigh',
                 missingmethod='none',
                 feature_weights=[],
                 oldscores=[], oldreproj=[]):
  """select_next(X, U, mu, scoremethod, missingmethod, feature_weights)

  Select the next most-interesting item in X,
  given model U, singular values S, and mean mu for uninteresting items.

  'scoremethod' indicates which residual values count towards
  the interestingness score of each item:
  - 'low': negative residuals
  - 'high': positive residuals
  - 'lowhigh': both

  'missingmethod' indicates how to handle missing (NaN) values:
  - 'zero': set missing values to zero
  - 'ignore': ignore missing values following Brand (2002)
  - 'none': assert nothing is missing (NaN).  Die horribly if not true.

  'feature_weights' influence how much each feature contributes to the score.

  'oldscores' provides the scores calculated in the previous iteration;
  if not empty, skip scoring and just return the next best.
  Likewise, 'oldreproj' is needed if we do this shortcut.

  Return the index of the selected item, its reconstruction,
    its reconstruction score, and all items' reconstruction scores.
  """

  print "------------ SELECTING --------------"
  if U == []:
    printt("Empty DEMUD model: selecting item number %d from data set" % \
             (log.opts['iitem']))
    return log.opts['iitem'], [], []

  if X.shape[1] < 1 or U == [] or mu == []:
    printt("Error: No data in X and/or U and/or mu.")
    return None, [], []

  if X.shape[0] != U.shape[0] or X.shape[0] != mu.shape[0]:
    printt("Mismatch in dimensions; must have X mxn, U mxk, mu mx1.")
    return None, [], []

  # If oldscores is empty, compute the score for each item
  if oldscores == []:
    (scores, reproj) = score_items(X, U, mu, scoremethod, missingmethod)
  elif oldreproj == []:
    printt("Error: oldscores provided, but not oldreproj.")
    return None, [], []
  else: # both are valid, so use them here
    (scores, reproj) = (oldscores, oldreproj)

  # Select and return index of item with max reconstruction error,
  # plus the updated scores and reproj
  m = scores.argmax()

  return m, scores, reproj


#______________________________select_next_NN______________________________
def  select_next_NN(X, x):
  """select_next_NN(X, x)

  Select the nearest neighbor to x in X.

  Return the index of the selected item.
  """

  if X == [] or x == []:
    printt("Error: No data in X and/or x.")
    return None
  if X.shape[0] != x.shape[0]:
    printt("Mismatch in dimensions; must have X mxn, x mx1.")
    return None

  # Compute the (Euclidean) distance from x to all items in X
  scores = np.apply_along_axis(linalg.norm, 0, X - x[:,np.newaxis])

  # Select and return item with min distance to x
  m = scores.argmin()

  return m


#______________________________update_model________________________________
def  update_model(X, U, S, k, n, mu,
                  svdmethod='full',
                  missingmethod='zero'):
  """update_model(X, U, S, k, n, mu, svdmethod, missingmethod)

  Update SVD model U,S (dimensionality k)
  by either adding items in X to it,
  or regenerating a new model from X,
  assuming U already models n items with mean mu.
  Technically we should have V as well, but it's not needed.

  'svdmethod' indicates type of update to do:
  - 'full': Recompute SVD from scratch.  Discards current U, S.
  - 'increm-ross': Ross et al.'s method for incremental update,
    with mean tracking.
  - 'increm-brand': Brand's incremental SVD method

  'missingmethod' indicates how to handle missing (NaN) values:
  - 'zero': set missing values to zero
  - 'ignore': ignore missing values inspired by Brand (2002)
  - 'none': assert nothing is missing (NaN).  Die horribly if not true.

  Return new U, S, mu, n, and percent variances.
  """

  if X == []:
    printt("Error: No data in X.")
    return None, None, None, -1, None

  # If there is no previous U, and we just got a single item in X,
  # then create a U the same size, first value 1 (rest 0),
  # and return it with mu.
  if U == [] and X.shape[1] == 1:
    mu   = X
    # Do this no matter what.  Let mu get NaNs in it as needed.
    U    = np.zeros_like(mu)
    U[0] = 1
    S    = np.array([0])
    n    = 1
    pcts = [1.0]
    return U, S, mu, n, pcts

  ###########################################################################
  # Do full SVD of X if this is requested, regardless of what is in U 
  # Also, if n = 0 or U is empty, start from scratch
  output_k = False
  if svdmethod == 'full' or U == [] or n == 0:
    if n == 0:
      if U == []:
        printt("----- initial SVD -----")
        output_k = True
      else:
        # Reshape so we don't have an empty dimension (yay python)
        U = U.reshape(-1, 1)
    elif U == []:
      printt("WARNING: N (number of items modeled by U) is %d, not zero, but U is empty!" % n)

    # Bootstrap
    if missingmethod == 'ignore':
      printt("ERROR: ignore with full is not possible under ordinary circumstances.")
      printt("Use --increm-brand to impute for NaNs.")
      printt("For now, we are filling NaNs with 0.")
      X    = copy.deepcopy(X)
      z    = np.where(np.isnan(X))
      X[z] = 0

    mu      = np.mean(X, axis=1).reshape(-1,1)
    X       = X - mu
    U, S, V = linalg.svd(X, full_matrices=False)
    
    # Keep only the first k components
    S_full = S
    S = S[0:k]
    U = U[:,0:k]

    # Update n to number of new items in X
    n = X.shape[1]
    
  ###########################################################################
  # Incremental SVD from Ross
  elif svdmethod == 'increm-ross':
    # Incremental SVD from Ross et al. 2008
    # "Incremental Learning for Robust Visual Tracking"
    # based on Lim and Ross's sklm.m implementation in MATLAB.

    # This method DOES NOT handle missing values.
    if missingmethod == 'ignore':
      print 'ERROR: increm-ross cannot handle missing values.'
      print 'If they are present, try svdmethod=increm-brand'
      print '  or use missingmethod=zero to zero-fill.'
      print 'If there are no missing values, specify missingmethod=none.'
      sys.exit(1)

    n_new  = X.shape[1]
    
    # Compute mean
    # Weirdly, the later 'X-mu_new' is MUCH faster if you reshape as shown.
    #  This is because of differences in the way numpy treats a 1d array versus a 2d column.
    mu_new = np.mean(X, axis=1).reshape(-1,1)

    # Subtract the mean, append it as a column vector, and update mu
    # X - mu_new will be zero if X has only 1 item
    mu_old = mu
    # New mu is a weighted sum of old and new mus
    mu     = (n * mu_old + n_new * mu_new) / (n + n_new)
    B      = np.hstack((X - mu,
                           math.sqrt(n_new * n/float(n_new+n)) * \
                           (mu_old - mu_new)))
    printt("Now tracking mean for %d -> %d items; mu.min %f, mu.max %f " % \
        (n, n+n_new, np.nanmin(mu), np.nanmax(mu)))
    n      = n + n_new

    if S.all() == 0:
      npcs  = U.shape[1]
      diagS = np.zeros((npcs, npcs))
    else:
      diagS = np.diag(S)

    # I don't think this is right.  At this point B is the augmented
    # matrix rather than the single observation.
    proj       = np.dot(U.T, B)
    reproj_err = B - np.dot(U, proj)

    # to get orthogonal form of reproj_err
    #  This should return q with dimensions [d(X) by n_new+1], square
    q, dummy   = linalg.qr(reproj_err, mode='full')
    # print 'q.shape should be 7x2: ', q.shape
    Q = np.hstack((U, q))

    # From Ross and Lim, 2008
    # R = [ [ Sigma, U.T * X ] [ 0, orthog. component of reproj error ] ]
    k_now = diagS.shape[0]
    new_dim = k_now + n_new + 1
    R = np.zeros((new_dim, new_dim))
    R[0:k_now,0:k_now] = diagS
    R[0:k_now,k_now:] = proj
    orthog_reproj_err = np.dot(q.T, reproj_err)
    R[k_now:, k_now:] = orthog_reproj_err
    
    # Perform SVD of R.  Then finally update U.
    U, S, V = linalg.svd(R, full_matrices=False)

    U = np.dot(Q, U)
    
    # Keep only the first k components
    U = U[:,0:min([n,k])]
    S_full = S
    S = S[0:min([n,k])]

  ###########################################################################
  # Incremental SVD from Brand
  elif svdmethod == 'increm-brand':
    # Pulled out James's attempt to handle NaNs into
    # increm-brand-james.py.  Starting over from scratch here.
    n_new  = X.shape[1]

    if n_new != 1:
      print "WARNING: increm-brand will probably only work by adding one item at a time."
      raw_input('\nPress enter to continue or ^C/EOF to exit. ')

    if missingmethod == 'ignore':
      # 1. Update mu
      mu_old = mu
      mu_new = X

      # Be careful!  For any pre-existing NaNs in mu,
      # let mu_new fill them in.  Can't get any worse!
      naninds = np.where(np.isnan(mu_old))[0]
      if naninds.size > 0:
        mu_old[naninds,0] = mu_new[naninds,0]
      # And likewise for mu_new -- fill with good values from mu_old.
      naninds = np.where(np.isnan(mu_new))[0]
      if naninds.size > 0:
        mu_new[naninds,0] = mu_old[naninds,0]
      # At this point, the only NaNs that should appear are
      # values that were NaN for both mu and X to start with.
      # They will stay NaN and that's okay.
      
      # New mu is a weighted sum of old and new mus
      mu     = (n * mu_old + n_new * mu_new) / (n + n_new)
      printt("Now tracking mean for %d -> %d items; mu.min %f, mu.max %f " % \
             (n, n+n_new, np.nanmin(mu), np.nanmax(mu)))
      n      = n + n_new

      # 2. Subtract off the mean
      X = X - mu

      # 3. Compute L, the projection of X onto U
      # Note: this will only work for a single item in X
      goodinds = np.where(~np.isnan(X))[0]
      #print 'X: %d of %d are good.' % (len(goodinds), X.shape[0])

      diagS    = np.diag(S)
      # This is Brand's method, which involves S:
      L = np.dot(diagS,
                 np.dot(np.linalg.pinv(np.dot(U[goodinds,:],
                                              diagS)),
                        X[goodinds,:]))
      # Simplified version that does not use S (but is probably wrong):
      #L = np.dot(U[goodinds,:].T,
      #           X[goodinds,:])
      # Top row of the Q matrix (eqn 12, Brand 2002)
      Q1 = np.hstack([diagS, L])

      # 4. Compute J, the orthogonal basis of H, which is
      #    the component of X orthog to U (i.e., unrepresentable direction)
      # 5. Compute K, the projection of X onto J (i.e., unrep. content)
      K = linalg.norm(X[goodinds,:] - np.dot(U[goodinds,:],
                                             np.dot(U[goodinds,:].T,
                                                    X[goodinds,:])))
      # H = X - UL
      J = np.zeros((U.shape[0], 1))
      J[goodinds] = np.dot(K,
                           np.linalg.pinv(X[goodinds,:] -
                                          np.dot(U[goodinds,:],
                                                 L))).T
      
      # Bottom row of Q matrix (eqn 12, Brand 2002)
      Q2 = np.hstack([np.zeros([1, len(S)]), np.array(K).reshape(1,1)])
      Q = np.vstack([Q1, Q2])

      # 6. Take the SVD of Q
      Uq, Sq, Vq = linalg.svd(Q, full_matrices=False)

      # 7. Update U and S (eqn 4, Brand 2002)
      # Note: Since J is zero-filled for badinds, now U is too.
      # Alternatively, we give J NaNs and let them get into U as well.
      # I think that is a worse idea though.
      U = np.dot(np.hstack([U, J]), Uq)
      S = Sq
      # Updating V requires knowing old V,
      # but we don't need the new one either so it's okay to skip.
      
      ############# end ###########
      
    else: # No missing values (or not 'ignore')
      # 1. Update mu
      mu_old = mu
      mu_new = X
      # New mu is a weighted sum of old and new mus
      mu     = (n * mu_old + n_new * mu_new) / (n + n_new)
      n      = n + n_new

      # 2. Subtract off the mean
      X = X - mu

      # 3. Compute L, the projection of X onto U
      L = np.dot(U.T, X)
      Q1 = np.hstack([np.diag(S), L])

      # 4. Compute J, the orthogonal basis of H, which is
      #    the component of X orthog to U (i.e., unrepresentable direction)
      # 5. Compute K, the projection of X onto J (i.e., unrep. content)
      JK = X - np.dot(U, L)
      (J, K) = linalg.qr(JK)

      Q2 = np.hstack([np.zeros([1, len(S)]), np.array(K).reshape(1,1)])
      Q = np.vstack([Q1, Q2])

      # 6. Take the SVD of Q
      Uq, Sq, Vq = linalg.svd(Q, full_matrices=False)

      # 7. Update U and S (eqn 4, Brand 2002)
      U = np.dot(np.hstack([U, J]), Uq)
      S = Sq
      # V requires knowing old V,
      # but we don't need the new one either so it's okay.
    

    # Keep only the first k components
    U = U[:,0:min([n,k])]
    S = S[0:min([n,k])]

    Usum = U.sum(1)


  ###########################################################################
  # We have a bad svdmethod, but somehow didn't catch it earlier.
  else:
    printt("504: Bad Gateway in protocol <Skynet_authentication.exe>")
    return None, None, None, None, None

  indivpcts = None

  # This only works if a full SVD was done
  if (svdmethod == 'full' and output_k and log.opts['k_var'] == -773038.0):
    # Calculate percent variance captured by each 
    cumsum      = np.cumsum(S_full)
    #print cumsum.shape
    if cumsum[-1] != 0:
      indivpcts   = S / cumsum[-1]
      indivpcts   = indivpcts[0:k]  # truncate to first k
      cumpercents = cumsum / cumsum[-1]
    else:
      indivpcts   = []

    # Calculate percent variance captured
    if k >= cumsum.shape[0]:
      printt('Cannot estimate data variance; specified k (%d) exceeds the number of SVs (%d).' % (k, cumsum.shape[0]))
    else:
      printt("Selected value of k=%d captures %5.2f%% of the data variance" % \
             (k, cumpercents[k-1] * 100))
    if log.opts['pause']: raw_input("Press enter to continue\n")

  return U, S, mu, n, indivpcts

#______________________________demud_______________________________________
def  demud(ds, k, nsel, scoremethod='lowhigh', svdmethod='full', 
           missingmethod='none', feature_weights=[], 
           start_sol=None, end_sol=None, flush_parameters=False):
  """demud(ds, k, nsel, scoremethod, svdmethod, missingmethod, feature_weights):

  Iteratively select nsel items from data set ds,
  using an incremental SVD model of already-seen items
  with dimensionality k.

  'scoremethod' indicates which residual values count towards
  the interestingness score of each item:
  - 'low': negative residuals
  - 'high': positive residuals
  - 'lowhigh': both

  'svdmethod' indicates type of update to do:
  - 'full': Recompute SVD from scratch.
  - 'increm-ross': Ross et al.'s method for incremental update,
    with mean tracking.

  'missingmethod' indicates how to handle missing (NaN) values:
  - 'zero': set missing values to zero
  - 'ignore': ignore missing values following Brand (2002)
  - 'none': assert nothing is missing (NaN).  Die horribly if not true. (Default)
  """

  # Sanity-check/fix nsel
  if nsel > ds.data.shape[1]:
    nsel = ds.data.shape[1]

  printt("Running DEMUD version %s for %d iterations using k=%d" %
         (__VERSION__, nsel, k))

  ###########################################################################
  # Check to ensure that parameters are valid
  if ds.data == []:
    printt("Error: No data in ds.data.")
    return 
  if k < 1:
    printt("Error: k must be at least 1.")
    return 
  if nsel < 1:
    printt("Error: nsel must be at least 1.")
    return
  elif nsel == 0:
    printt("Warning: nsel = 0.  This means demud will do nothing, slowly.")

  if 'iitem' not in log.opts or flush_parameters:
    # Temporary hack to allow the demud() method to be called from external scripts.
    #   Better long-term support for this should probably exist.
    log.opts['iitem'] = 'mean'
    log.opts['shotfilt'] = 0
    log.opts['fft'] = False
    log.opts['static'] = False
    log.opts['coi'] = None
    log.opts['note'] = None
    log.opts['coiaction'] = None
    log.opts['plotvariance'] = False
    log.opts['kepler'] = False
    log.opts['plot'] = True
    log.opts['mastcam'] = False
    log.opts['interactive'] = False
    log.opts['alwaysupdate'] = False
    log.opts['start_sol'] = start_sol
    log.opts['end_sol'] = end_sol
    log.opts['log'] = True
    log.opts['k'] = k
    log.opts['k_var'] = False
    log.opts['svdmethod'] = svdmethod
    log.opts['missingdatamethod'] = missingmethod
    log.opts['svd_print'] = False
    log.opts['md_print'] = False
    log.opts['clean'] = False
    log.opts['printk'] = False
    log.opts['score_print'] = False
    log.opts['fw_print'] = False
    log.opts['fun'] = False # no fun for you!
    log.opts['pause'] = False

    print "No method of initializing the dataset was chosen.  Defaulting to mean."

  log.opts['start_sol'] = start_sol
  log.opts['end_sol']   = end_sol

  ###############################################
  # Add experiment information to dataset name
  # TODO: store this information in a text file within the directory instead,
  # and find another way to usefully create distinct directory names (maybe nested)
  ds.name += '-k=' + str(k)
  ds.name += '-dim=' + str(ds.data.shape[0])
  ds.name += '-' + svdmethod
  if scoremethod != 'lowhigh': ds.name += '-score=' + scoremethod
  if missingmethod != "none": ds.name += '-missing=' + missingmethod
  if feature_weights != []: ds.name += '-featureweight=' + os.path.basename(log.opts['fw'])
  if log.opts['sol'] != -1: ds.name += '-sol%d' % log.opts['sol']
  if log.opts['sols'] != None: ds.name += '-sol%d' % log.opts['start_sol']
  if log.opts['sols'] != None: ds.name += '-%d' % log.opts['end_sol']
  if ds.initfilename != '' and ds.initfilename != None:
    ds.name += '-init-file'
  if log.opts['init_prior_sols'] == True:
    ds.name += '-init-prior'
  else:
    ds.name += '-init_item=' + str(log.opts['iitem'])
  if log.opts['shotfilt'] != 0: ds.name += '-shotfilt=%d' % log.opts['shotfilt']
  if log.opts['fft']: ds.name += "-fft"
  if log.opts['static']: ds.name += '-static' 
  if log.opts['coi'] != None: ds.name += '-coi-' + log.opts['coiaction'] \
      + '=' + log.opts['coi']
  if log.opts['note'] != None: ds.name += '-' + log.opts['note']
  
  ###############################################
  # Set up output directories for plots and logging
  if not os.path.exists('results'):
    os.mkdir('results')
  outdir = os.path.join('results', ds.name)
  if not os.path.exists(outdir):
    os.mkdir(outdir)
  log.logfilename = os.path.join(outdir, 'demud.log')
  log.logfile     = open(log.logfilename, 'w')
  
  ###############################################
  # Print dataset info
  printt("Dataset: " + ds.name)
  printt(" Read from " + ds.filename)
  printt(" Dimensions (features x items): %d x %d" % ds.data.shape)
 
  ###############################################
  # Plot variance
  if log.opts['plotvariance']:
    plot_variance(ds)

  ###############################################
  # Check once and for all if the input data contains NaN's.
  
  X = ds.data
  
  # Warn user what to do if it's true and s/he aren't ready for it
  nans = np.isnan(X)
  if np.sum(nans) > 0:

    #If no missingmethod selected, code will break. Prevent that noisily.
    if missingmethod == 'none':
      printt('WARNING! Input data contains NaNs but no handling method has been chosen.')
      printt('Please use one of the following, knowing ignore is MUCH slower:')
      printt('--missingdatamethod=zero/ignore')
      sys.exit(-1)

    #If we are going to handle NaN's with zeroing, just do it now
    if missingmethod == 'zero':
      z = np.where(nans)
      if len(z) > 0:
        printt("Filling NaNs with 0: %d / %d = %0.2f%%" % (np.sum(nans), X.shape[0]*X.shape[1], np.sum(nans)/float(X.shape[0]*X.shape[1])*100))
        X[z] = 0

    #Let user know we're in ignore NaN mode
    if missingmethod == 'ignore':
      printt('Missing data (NaNs) will be ignored. This is a very slow operation.')
  else:
    printt('No NaNs in this data file, rest easy.')

  ###############################################
  # Initialize the model
  
  U = []
  # S = np.array([0])
  S = np.array([1])
  mu = []
  n = 0
  pcts = []
  
  # Initial dataset is supplied
  if ds.initfilename != '' and ds.initfilename != None:
    log.opts['iitem'] = -1  # not really, but default is 0
    printt('Initializing model with data from %s' % ds.initfilename)
    U, S, mu, n, pcts = update_model(ds.initdata, U, S, k, n=0, mu=[],
                                     svdmethod=svdmethod,
                                     missingmethod=missingmethod)
    
  # Doing a full SVD
  elif log.opts['static'] or log.opts['iitem'] in ('-1','svd','SVD'):
    log.opts['iitem'] = -1
    printt('Doing initial SVD to get started.')
    U, S, mu, n, pcts = update_model(X, U, S, k, n=0, mu=[],
                                     svdmethod=svdmethod, 
                                     missingmethod=missingmethod)
    
  # Select random item
  elif log.opts['iitem'] in ('r','R','random','RANDOM'):
    randitem = np.random.randint(X.shape[1])
    printt('Selecting random item = %d to get started.'%randitem)
    log.opts['iitem'] = randitem
  
  # Use dataset mean
  elif log.opts['iitem'] in ('mean', 'average', 'mu', '-2', -2):
    printt('Initializing model to mean and skipping to selection 1')
    mu = nanmean(X, axis=1).reshape(-1,1)
    # if we have missingmethod set to 'ignore', and some features are entirely NaNs,
    #   then there will still be NaNs in mu.  We can safely set them to zero for now
    #   because 'ignore' will mean that these values don't ever get looked at again.
    #mu[np.isnan(mu)] = 0
    # print "Just calculated mean."
    #U, S, V = linalg.svd(mu, full_matrices=False)

    # Instead, treat this like we do for first selection normally.
    U    = np.zeros_like(mu)
    U[0] = 1
    S    = np.array([0])

    pcts = [1.0]
    log.opts['iitem'] = -2

  log.opts['iitem'] = int(log.opts['iitem'])
  
  printt('')
  
  ###############################################
  # Initialize all of the counters, sums, etc for demud
  n_items  = X.shape[1]
  orig_ind = np.arange(n_items)
  seen = ds.initdata
  ncois = 0
  whencoiswerefound = []
  oldscoresum = -1
  maxscoresum = -1
  sels = []
  sels_idx = []
    
  # Create the 'results' directory, if needed
  if not os.path.exists('results'):
    os.mkdir('results')

  ###########################################################################
  ## MAIN ITERATIVE DISCOVERY LOOP

  scores = []
  reproj = []
  for i in range(nsel):
    printt("Time elapsed at start of iteration %d/%d:" % (i, nsel-1),
           time.clock())

    ###############################################
    # If we just found a COI (class of interest) in the previous round,
    # and the coi-action is 'seek',
    #   then don't use the model to select the next item.
    # Instead, do a nearest-neighbor search through X based on sels[-1].
    
    if whencoiswerefound != [] and (whencoiswerefound[-1] == i-1) and \
        (log.opts['coiaction'] == 'seek'):
      printt("Actively searching for the next COI based on item %d"
             % sels[-1])
      ind = select_next_NN(X, ds.data[:,sels[-1]])
      # There is no reconstruction, r, of the new item.
      # But we want something to generate explanations.
      # So use the previous selection as the reconstruction.
      # Warning: this may be confusing, since this item
      # was picked to be SIMILAR to r, not different from it.
      r = ds.data[:,sels[-1]]
      # We update scores simply by removing this item.
      # I think we don't want to do this since scores gets updated 
      # after plotting info for this choice.
      #scores = np.delete(scores, ind)
      
    ###############################################
    # Get the selection, according to model U
    else:
      # If using a static model, pass in oldscores and oldreproj
      # to avoid re-calculating scores
      if log.opts['static']:
        ind, scores, reproj = select_next(X, U, mu, scoremethod,
                                          missingmethod, feature_weights,
                                          oldscores=scores, oldreproj=reproj)
      else:
        ind, scores, reproj = select_next(X, U, mu, scoremethod,
                                          missingmethod, feature_weights)
      # If initializing with a specific item, 
      # then scores and reproj will be empty
      if scores == []:
        score = 0.0
        r     = X[:,ind] # reproj is same as item itself
      else:
        score = scores[ind]
        r     = reproj[:,ind]

    # Update selections
    sels += [orig_ind[ind]]
    sels_idx += [ind]

    printt("%d) Selected item %d (%s), score %g." % \
        (i, orig_ind[ind], ds.labels[orig_ind[ind]], score))

    ###############################################
    # Report the fractional change in sum of reconstruction scores
    scoresum = sum(scores)
    if i > 1:
      printt(" Total data set reconstruction error: %g (of %g is %.2f%%)"
             % (scoresum, maxscoresum, scoresum/maxscoresum*100))
    else:
      printt("  Initial total data set reconstruction error: %g" % scoresum)
      maxscoresum = scoresum
    oldscoresum = scoresum

    ###############################################
    # This selection is x.  Print its characteristics
    x = X[:,ind]

    printt("  Min/max x values: %.2e, %.2e" % (np.nanmin(x),
                                              np.nanmax(x)))
    printt("  Min/max r values: %.2e, %.2e" % (np.nanmin(r),
                                              np.nanmax(r)))
    diff = x-r
    printt("  Min/max/mean residual: %.2f, %.2f, %.2f" % (np.nanmin(diff),
                                                         np.nanmax(diff),
                                 np.mean(diff[np.where(~np.isnan(diff))])))
    printt("    Class = %s" % ds.labels[orig_ind[ind]])
    printt('')
    
    ###############################################
    # Report on how many nans are in this selection.
    if (len(np.where(np.isnan(X))[0]) > 0 or 
        len(np.where(np.isnan(seen))[0]) > 0):
      goodinds = np.where(~np.isnan(x))[0]
      print '  Sel. %d: %d (%.2f%%) good indices (not NaN)' % \
          (i, len(goodinds), 100*len(goodinds) / float(len(x)))
    
    ###############################################
    # Plot item using dataset's plotting method.
    label = ds.labels[orig_ind[ind]]
    if log.opts['kepler']:
      dc = log.opts['static'] or (U != [] and U.shape[1] > 1)
      dsvd = (log.opts['static'] and i is 0) or (U != [] and U.shape[1] > 1)
      if log.opts['plot']:
        ds.plot_item(i, orig_ind[ind], x, r, k, label,
                     U, mu, S, X, pcts, scores, drawsvd=dsvd, drawcloud=dc)
    elif log.opts['navcam']:
      if log.opts['plot']:
        ds.save_rec(r, orig_ind[ind], X[:,ind], k)
        ds.plot_item(i, orig_ind[ind], x, r, k, label)
    else:
      if log.opts['plot']:
        ds.plot_item(i, orig_ind[ind], x, r, k, label,
                     U, scores, feature_weights)
    
    ds.write_selections_csv(i, k, orig_ind[ind], label, ind, scores)


    if log.opts['decals']:
      #####################################################
      # Write a list of selections that are similar to this selection (x).
      # First, score all items with respect to a single-item model of x.
      # Create a U the same size as x, first value 1 (rest 0),
      # and set mu to be x.
      this_U    = np.zeros_like(x)
      this_U[0] = 1
      this_U    = this_U.reshape(-1, 1)
      this_mu   = x
      this_mu   = this_mu.reshape(-1, 1)
      (this_item_scores, reproj) = score_items(X, this_U, this_mu,
                                               scoremethod, missingmethod, 
                                               feature_weights)
      ds.write_similar_html(10, i, k, ind, this_item_scores)

    ###############################################
    # Setup for checking if to update or not.
    do_update = True

    ###############################################
    # Don't update if class of interest is found in item label
    if log.opts['coi'] is not None:
      if log.opts['coi'] in ds.labels[orig_ind[ind]]: 
        do_update = False
        printt("Not updating, it's the class of interest!")
        ncois = ncois + 1
        whencoiswerefound += [i]
        
        # If the coi-action is 'keep', proceed normally (do_update is false).
        # If the coi-action is 'seek', next selection should be
        #  the nearest neighbor to the last one.  This check is done
        #  in select_next().

        # If we've hit n_conts, exit
        if ncois == log.opts['n_coi']:
          printt("\nFound class of interest on these picks:\n")
          if not log.opts['kepler']:
            for ff in whencoiswerefound: printt(ff)
          else:    
            el = ExoplanetLookup()
            for fff in whencoiswerefound: 
              kid = int(ds.labels[sels[fff]].split(':')[0])
              kidmin = el.min_period(kid)
              if kidmin == "Unknown":
                printt("%3d) %8d: min period %7s days" % (fff, kid, kidmin))
              else:
                printt("%3d) %8d: min period %7.4f days" % (fff, kid, float(kidmin)))
            return
          return

    ###############################################
    # Be interactive!
    if log.opts['interactive']:
      userpref = raw_input("Do you want to add this to the model of "
                           "uninteresting data?\n(Y/N) ")
      printt(userpref)
      if userpref.lower() in ('y', 'yes', 'yes please'):
        printt("You said yes!")
        pass
      elif userpref.lower() in ('n', 'no', 'no thank you'):
        printt("You said no!")
        do_update = False
      else:
        bad_input = True
        while bad_input:
          printt("Sorry, I don't recognize that input.  Please choose yes or no.")
          userpref = raw_input("(Y/N) ")
          if userpref.lower() == 'y' or 'yes' or 'yes please':
            bad_input = False
            pass
          elif userpref.lower() == 'n' or 'no' or 'no thank you':
            bad_input = False
            do_update = False
    
    ###############################################
    # Check for static SVD
    if log.opts['alwaysupdate'] and do_update == False: 
      do_update = True
      printt("Doing an update!")
      
    if log.opts['static']: do_update = False
          
    ###############################################
    # Update the model
    if do_update:
      # We are doing this check because 'full' will only model what's in seen
      #   increm-ross will add what's in seen to the model U, S
      if svdmethod == 'full':
        if seen == []:
          seen = x.reshape(-1,1)
        else:
          seen = np.hstack((seen,x.reshape(-1,1)))
      else:
        seen = x.reshape(-1,1)
        # Reset U to empty if this is the first iteration
        # in case an SVD was used to select the first item,
        # UNLESS an initial data set was specified.
        if (i == 0 and ds.initdata == []): 
          U = []

      U, S, mu, n, pcts = update_model(seen, U, S, k, n, mu,
                                       svdmethod=svdmethod,
                                       missingmethod=missingmethod)
    else:
      printt("Skipped updating model U because data was interesting.")

    ###############################################
    # Remove this item from X and other variables
    keep     = range(X.shape[1])
    keep.remove(ind)
    X        = X[:,keep]
    orig_ind = orig_ind[keep]
    if scores != []:
      scores = scores[keep]
      reproj = reproj[:,keep]

    printt()   # spacing

    ###############################################
    # Plot the top 4 principal components of the new model
    if U != [] and log.opts['plot'] and log.opts['dan']:
      ds.plot_pcs(i, U, mu, k, S)
    # if log.opts['misr']:
    #   pylab.clf()
    #   pylab.imshow(U.reshape([ds.along_track, -1]))
    #   pylab.show()

    # End loop over selections

  ###############################################
  # Report on when observations from the class of interest were found (if any)
  if len(whencoiswerefound) > 0:
    printt("\nFound class of interest on these picks:\n")
    if not log.opts['kepler']:
      for ff in whencoiswerefound: printt(ff)
    else:    
      el = ExoplanetLookup()
      for fff in whencoiswerefound: 
        kid = int(ds.labels[sels[fff]].split(':')[0])
        minper = el.min_period(kid)
        if minper != 'Unknown':
          printt("%3d) %8d: min period %7.4f days" % (fff, kid, float(minper)))
        else:
          printt("%3d) %8d: min period unknown" % (fff, kid))
  
  ###############################################
  # Return
  return (sels, sels_idx)
    
#______________________________generate_feature_weights____________________
def  generate_feature_weights(d, xvals):
  """generate_feature_weights(d, xvals):
  
  Generate feature weights for a dataset with d items based on the 
    value of the optparse variable 'fw' specified by --featureweightmethod=
  
  This is the place in the code to add more feature weighting methods.
    You do not need to worry about modifying the check_opts() method below,
    but you may want to edit fw_print() (--featureweightmethods).
  
  """
  
  if log.opts['fw'] == 'flat' or log.opts['fw'] == '':
    return []
  elif log.opts['fw'] == 'boostlow':  
    return [(1. / (i ** (1. / math.log(d, 2)))) for i in range(1, d+1)]
  elif log.opts['fw'] == 'boosthigh':
    ocelot = [(1. / (i ** (1. / math.log(d, 2)))) for i in range(1, d+1)]
    ocelot.reverse()
    return ocelot
  else:
    # Assume it's a filename and attempt to read weights from the file
    return read_feature_weights(log.opts['fw'], xvals)

  return []


def  read_feature_weights(fwfile, xvals):
  """read_feature_weights(fwfile, xvals)
  
  Read feature weights from the specified file.
  The data set's xvals must be provided so this method can 
    sanity-check the number of weights (and match to wavelengths,
    if specified).
  """

  # If no file was specified, skip this step.
  if fwfile == '': # default
    return []

  if not os.path.exists(fwfile):
    printt(' Could not find feature weight file %s, skipping.' % fwfile)
    return []

  printt('Treating feature-weight argument as a file name (%s).' % fwfile)
  # Read in the feature-weight file
  f = open(fwfile)
  lines = f.readlines()
  if len(lines) != len(xvals):
    raise ValueError('Looking for %d weights, but got %d.' % (len(xvals), len(lines)))

  wolf = [-1] * len(xvals)
  for (i,line) in enumerate(lines):
    values = line.split()

    # Sanity check that we have one of two cases:
    # 1. One weight per line
    if len(values) == 1:
      wolf[i] = float(values[0])

    # 2. Two values per line (feature - matches xvals, weight)
    elif len(values) == 2:
      values = map(float, values)
      feat_ind = np.where(xvals == values[0])[0]
      if len(feat_ind) == 0:
        raise ValueError('Could not find feature %s.' % values[0])
      wolf[feat_ind[0]] = values[1]

  # Check that feature weights are valid
  for weight in wolf:
    if weight < 0 or weight > 1:
      raise ValueError("Weights must be between 0 and 1.")
    
  return wolf

#______________________________finish_initialization_______________________
# Print out data shape, and check to set N to max.
def  finish_initialization(ds, action='reading in dataset'):
  global use_max_n, default_n_value

  printt("Time elapsed after " + action + ":", time.clock())

  if use_max_n or default_n_value > ds.data.shape[1]:
    printt("Iterating through all data vectors.")
    default_n_value = ds.data.shape[1]

  fw = generate_feature_weights(ds.data.shape[0], ds.xvals)
  
  return fw
  
#______________________________check_if_files_exist________________________
# Takes a list of files and determines if any of them do not exist.
# If so, exits in disgrace.
def  check_if_files_exist(files, ftype='input'):
  # First file is the data file and must exist
  if files[0] == '':
    printt("%s file not specified." % ftype)
    return False
  
  for f in files:
    # Allow .pkl files to not exist, since they will be generated.
    if f == '' or f.endswith('.pkl'):
      continue
    if not os.path.exists(f):
      printt("Error: specified %s file %s does not exist" % (ftype, f))
      return False
  return True
     
#______________________________report_classes______________________________
# Reports upon classes found.  Suppressed with --no-report option.
def report_classes(ds, sels, sels_idx, data_choice):
  # print a list of all classes found in first nsel selections
  if not (data_choice is 'navcam'):
    found = []
    printt("CLASSES FOUND:\n")
    for (i, s) in enumerate(sels):
      if not ds.labels[s] in found:
        printt('Class found on selection %d: %s' % (i, ds.labels[s]))
        found.append(ds.labels[s])
    printt("\nNumber of classes found: %d\n" % len(found))
  else:
    file_sels = {};

    for files in ds.datafiles:
      file_sels[files] = [];

    for i,idx in enumerate(sels_idx):
      file_idx = np.where(ds.img_label_split > idx)[0][0]-1;
      file_sels[ds.datafiles[file_idx]] += [sels[i]];

    for key in file_sels:
      if file_sels[key]: 
        print("File: %s" %(key));
        for csels in file_sels[key]:
          i = sels.index(csels) 
          printt('Class found on selection %d: %s' % (i,ds.labels[csels]))

      
#______________________________svd_print___________________________________
# print out SVD options and exit.
def  svd_print():
  printt("'svdmethod' indicates type of update to do:")
  printt("- 'full': Recompute SVD from scratch.  (Default)")
  printt("- 'increm-ross': Ross et al.'s method for incremental update,")
  printt("         with mean tracking. Does not handle missing values.")
  printt("- 'increm-brand': Brand's method for incremental update,")
  printt("         with mean tracking. Can handle missing values.")
  printt("")
  printt("--increm is a shortcut for --svdmethod=increm-ross.")
  printt("")
  exit()

#______________________________score_print_________________________________
# Print out scoring options and exit.
def  score_print():
  printt("'scoremethod' indicates how to score sources by reconstruction error:")
  printt("- 'low': negative residuals")
  printt("- 'high': positive residuals")
  printt("- 'lowhigh': both (Default)")
  exit()
  
#______________________________md_print____________________________________
# Print out missing data options and exit.
def  md_print():
  printt("'missingdatamethod' indicates how to handle missing (NaN) values:")
  printt("- 'zero': set missing values to zero (Default)")
  printt("- 'ignore': ignore missing values following Brand (2002)")
  printt("- 'none': assert nothing is missing (NaN).  Die horribly if not true.")
  exit()
  
#______________________________fw_print____________________________________
# Print out feature weight options and exit.
def  fw_print():
  printt("'featureweightmethod' indicates how to weight features given:")
  printt("- 'flat': all features weighted 1.0 (default)")
  printt("- 'boostlow': boost earlier features more on a sliding scale")
  printt("- 'boosthigh': boost higher features the same way")
  printt("")
  printt("Any other argument will be interpreted as a file name to be read from.")
  printt("  The file must contain one weight per line as a float or int.")
  printt("  Does not currently accept negative weights, since it would be")
  printt("    confusing: weights are applied multiplicatively to residuals")
  printt("    immediately before scoring, which uses absolute value.")
  printt("   This could however be developed further to interact with scoring")
  printt("     methods, such that features would be scored differently.")
  printt("")
  printt("The function used for the boost methods is as follows:")
  printt("  For a dataset with n features:")
  printt("  [(1.0 / (i ** (1.0 / math.log(n, 2)))) for i in range(1, n+1)]")
  printt("    ^ Reverse for boosthigh")
  printt("  This will weight the most important feature at 1.0")
  printt("    and the least important at 0.5, with an exponential curve between.")
  exit()

#______________________________make_config_________________________________
# Remake demud.config. 
def  clean():

  #global log.opts

  # Figure out if we're clobbering an existing config file
  if os.path.exists(os.path.join(os.getcwd(), 'demud.config')):

    printt("WARNING: This will overwrite your current demud.config file.")
    printt("  Do you really want to continue?")
    y = raw_input("  Enter Y to continue, anything else to abort: ")

    if y.lower() != 'y' and y.lower() != 'yes' and y.lower() != 'yes please':
      printt("Aborting.")
      exit()
    
    if y.lower() == 'yes please': 
      if log.opts['fun']:
        printt("Certainly!  Thank you for saying please!")
    
  printt("\nWriting to demud.config\n")


  # Do the actual output
  outputfile = open(os.path.join(os.getcwd(), 'demud.config'), 'w+')

  outputfile.write("Demud.config\nJames Bedell\n06/26/2013\n"
                   "########### README #################\n\n"
                   "Each dependency line takes the format:\n"
                   "mydatafile = /home/jbedell/research/data/sample.data\n"
                   "(either relative or absolute path)\n"
                   "Single and double quotes are okay but not necessary.\n\n"
                   "Lines which begin with a # like Python comments are "
                   "ignored (leading whitespace okay)\n"
                   "(comments which begin in the middle of "
                   "lines may produce unexpected results)\n\n"
                   "** Only the last (unignored) assignment to any "
                   "variable is kept. **\n\n"
                   "Below are the data sets and their dependencies.  "
                   "Headers are of the format:\n"
                   "----- Sample data classification set: "
                   "sampledatafile sampledatamap\n"
                   " -h --help\n\n"
                   "Do not delete the headers, or demud.py as is "
                   "may suffer an ignominious demise by IndexError\n\n")
  
  # this is the part in
  # the code where a haiku is
  # lurking silently

  outputfile.write("############ DATASETS #################\n\n"
                   "----- Glass classification data set: ucidatafile\n"
                   " -g --glass\n"
                   "----- Iris classification data set: ucidatafile\n"
                   " -i --iris\n"
                   "----- E. Coli classification data set: ucidatafile\n"
                   " -e --ecoli\n"
                   "----- Abalone classification data set: ucidatafile\n"
                   " -o --abalone\n"
                   "----- ISOLET letter classification: ucidatafile\n"
                   " -z --isolet\n"
                   "ucidatafile = \n\n"
                   "----- Test data set: floatdatafile\n"
                   " -x --testdata\n"
                   "----- Pancam spectra data set: floatdatafile\n"
                   " -p --pancam\n"
                   "----- APF spectra data set: floatdatafile\n"
                   " -b --apf\n"
                   "----- CNN feature data set: floatdatafile\n"
                   " -v --cnn\n"
                   "----- DAN spectra data set: floatdatafile\n"
                   " --dan\n"
                   "floatdatafile = \n\n"
                   "----- GBT filterbank data set: gbtdirname, catalogfile\n"
                   " --gbtfil\n"
                   "gbtdirname  = \n\n"
                   "catalogfile = \n\n"
                   "----- DECaLS FITS data set: decalsfilename\n"
                   " --decals\n"
                   "decalsfilename  = \n\n"
                   "----- DES FITS data set: desfilename\n"
                   " --des\n"
                   "desfilename  = \n\n"
                   "---- ChemCam: libsdatafile libsinitdatafile\n"
                   " -c --chemcam\n"
                   "libsdatafile = \n"
                   "libsinitdatafile = \n\n"
                   "----- FINESSE: finessedirname\n"
                   " -f --finesse\n"
                   "finessedirname = \n\n"
                   "----- MISR aerosol data: misrAODdirname, misrrawdirname, misrdatafile\n"
                   " -m --misr\n"
                   "misrAODdirname  = \n"
                   "misrrawdirname  = \n"
                   "misrdatafile = \n\n"
                   "----- AVIRIS data: avirisdirname, avirisrawfile, avirissegmap\n"
                   " -a --aviris\n"
                   "avirisdirname = \n"
                   "avirisrawfile = \n"
                   "avirissegmap  = \n\n"
                   "----- IRS Spitzer exoplanet atmosphere data: irsdatafile, irsdatefile, irsmodelfile, irswavecalfile\n"
                   " -s --spitzer\n"
                   "irsdatafile  = \n"
                   "irsdatefile  = \n"
                   "irsmodelfile = \n"
                   "irswavecalfile = \n\n"
                   "----- Kepler light curve data: keplerdatafolder, keplerdataextension\n"
                   " -k --kepler\n"
                   "keplerdatafolder = \n"
                   "keplerdataextension = \n\n"
                   "----- TextureCam image data: tcfilename, tcpklname\n"
                   " -t --texturecam\n"
                   "tcfilename = \n"
                   "tcpklname = \n\n"
                   "----- UCIS hyperspectral image cube: ucisrawfile\n"
                   " -u --ucis\n"
                   "ucisrawfile = \n\n"
                   "----- Mastcam images: mastcamdatafolder\n"
                   " -j --mastcam\n"
                   "mastcamdatafolder = \n\n"
                   "----- Images: imagedatafolder, imageinitdatafolder\n"
                   " -I --images\n"
                   "imagedatafolder = \n"
                   "imageinitdatafolder = \n\n"
                   "----- Image Sequence data: datafolder, solnumber, initdatasols\n"
                   "-q --navcam\n"
                   "datafolder = \n"
                   "solnumber = \n"
                   "scaleInvariant = \n"
                   "initdatasols = \n")
    
  outputfile.close()
    
  exit()
  
#______________________________parse_args__________________________________
# Set up option parser and parse command-line args
def  parse_args():
  ###########################################################################
  # Add command-line options.  Store their values.
  #
  global __VERSION__
  vers = __VERSION__
  parser = optparse.OptionParser(usage="python %prog [-gecmasktofuzj] [options]", 
                                  version=vers)
  
  # Datatype options
  dtypes = OptionGroup(parser, "Datatype Options", 
                          "Exactly one must be selected.")
  
  dtypes.add_option('-g', '--glass', help='Glass classification', 
                      default=False, action='store_true', dest='glass')
  dtypes.add_option('--iris', help='Iris classification', 
                      default=False, action='store_true', dest='iris')
  dtypes.add_option('-e', '--ecoli', help='E. coli classification', 
                      default=False, action='store_true', dest='ecoli')
  dtypes.add_option('-o', '--abalone', help='Abalone classification', 
                      default=False, action='store_true', dest='abalone')
  dtypes.add_option('-p', '--pancam', help='Pancam spectra', 
                      default=False, action='store_true', dest='pancam')
  dtypes.add_option('-b', '--apf', help='APF spectra',
                      default=False, action='store_true', dest='apf')
  dtypes.add_option('-v', '--cnn', help='CNN feature vectors', 
                      default=False, action='store_true', dest='cnn')
  dtypes.add_option('--dan', help='DAN spectra',
                      default=False, action='store_true', dest='dan')
  dtypes.add_option('--gbt', help='GBT spectra',
                      default=False, action='store_true', dest='gbt')
  dtypes.add_option('--gbtfil', help='GBT filterbank',
                      default=False, action='store_true', dest='gbtfil')
  dtypes.add_option('--decals', help='DECaLS FITS file',
                      default=False, action='store_true', dest='decals')
  dtypes.add_option('--des', help='DES FITS file',
                      default=False, action='store_true', dest='des')
  dtypes.add_option('-x', '--testdata', help='Test data', 
                      default=False, action='store_true', dest='testdata')
  dtypes.add_option('-c', '--chemcam', help='ChemCam data', default=False, 
                      action='store_true', dest='chemcam')
  dtypes.add_option('-f', '--finesse', help='FINESSE data', default=False, 
                      action='store_true', dest='finesse')
  dtypes.add_option('-m', '--misr', help='MISR aerosol data', default=False, 
                      action='store_true', dest='misr')
  dtypes.add_option('-a', '--aviris', help='AVIRIS data', default=False, 
                      action='store_true', dest='aviris')
  dtypes.add_option('-s', '--spitzer', help='Spitzer IRS exoplanet atmosphere data',
                      default=False, action='store_true', dest='irs')
  dtypes.add_option('-k', '--kepler', help='Kepler exoplanet data', 
                      default=False, action='store_true', dest='kepler')
  dtypes.add_option('-t', '--texturecam', help='TextureCam image data', 
                      default=False, action='store_true', dest='texturecam')
  dtypes.add_option('-u', '--ucis', help='UCIS hyperspectral data',
                      default=False, action='store_true', dest='ucis')
  dtypes.add_option('-z', '--isolet', help='ISOLET letter recognition data', 
                      default=False, action='store_true', dest='isolet')
  dtypes.add_option('-q', '--navcam', help='Run for sequence of Images (MER purpose)', 
                      default=False, action='store_true', dest='navcam')
  dtypes.add_option('-j', '--mastcam', help='MSL Mastcam image data', 
                      default=False, action='store_true', dest='mastcam')
  dtypes.add_option('-I', '--images', help='Image data in a directory', 
                      default=False, action='store_true', dest='images')

  parser.add_option_group(dtypes)
  
  # Parameter options
  params = OptionGroup(parser, "Parameter Options",
                          "Specify DEMUD parameters."
                          "  Override the defaults in demud.config.")
                          
  params.add_option('--k', help='Number of principal components for reconstructing data',
                      default=-773038, type=int, action='store', dest='k')
  params.add_option('--n', '--iters', help='Number of iterations of SVD and selection; default 10',
                      default=-773038, type=int, action='store', dest='n')
  params.add_option('--all', help="Iterate through all data items",
                      default=False, action='store_true', dest='all') 

  params.add_option('--svdmethod', help="SVD method to use on each iteration (see --svdmethods for a list)",
                      default='default', type=str, action='store', dest='svdmethod')
  params.add_option('--increm', help="Same as --svdmethod=increm-ross",
                      default=False, action='store_true', dest='increm')
  params.add_option('--missingdatamethod', help="How to handle missing data (see --missingdatamethods for a list)",
                      default='none', type=str, action='store', dest='missingdatamethod')    
  params.add_option('--scoremethod', help="How to score data residuals (see --scoremethods for a list)",
                      default='lowhigh', type=str, action='store', dest='scoremethod')             
  params.add_option('--featureweightmethod', help="How to weight features for scoring (see --featureweightmethods for a list)",
                      default='', type=str, action='store', dest='fw')  
  params.add_option('--static', help='Static SVD: truncate to k vectors and never update again',
                      default=False, action='store_true', dest='static')

  params.add_option('--sol', help="Analyze data from this sol.  Use with -c.",
                    default=-1, type=int, action='store', dest='sol')
  params.add_option('--sols', help="Analyze data from this sol range (<int>-<int>).  Use with -c.",
                    default=None, type=str, action='store', dest='sols')
  params.add_option('--initpriorsols', help="Initialize with data from sols prior to the specified sol or sol range.  Use with -c.",
                    default=False, action='store_true', dest='init_prior_sols')
  params.add_option('--note', help='Note to append to output directory name',
                      default=None, action='store', dest='note')                   
  
  parser.add_option_group(params)              
                   
  # Data processing and output options
  dataops = OptionGroup(parser, "Data processing and output options",
                          "Specify additional preprocessing or postprocessing options.")
                      
  dataops.add_option('--init-item', help='Index of initialization item (default: 0; -1 or svd for full-data SVD; r for random; mean for mean)',
                      default=0, type=str, action='store', dest='iitem')
  dataops.add_option('-i', '--interactive', help='Ask for feedback on adding selection to U', 
                      default=False, action='store_true', dest='interactive')
  dataops.add_option('--variance', help="Optimize --k to capture this much data variance\n Range: [0.0 1.0]",
                      default=-773038.0, type=float, action='store', dest='k_var')

  dataops.add_option('--shotnoisefilter', help='Apply median filter of specified width. Used with [-cu].',
                      default=0, type=int, action='store', dest='shotfilt')
  dataops.add_option('--fft', help='Perform FFT on data first. Only supported by [-k].', 
                      default=False, action='store_true', dest='fft')
  dataops.add_option('--lookup', help="Look up sources' status tags from MAST.  Used with [-k].",
                      default=False, action='store_true', dest='lookup')

  dataops.add_option('--no-report', help="Don't report on classes found.  Used with [-ego].",
                      default=True, action='store_false', dest='report')
  dataops.add_option('--no-plot', help="Don't plot any output.",
                      default=True, action='store_false', dest='plot')
  dataops.add_option('--no-log', help="Don't log text output.",
                      default=True, action='store_false', dest='log')
  dataops.add_option('--pause', help='Pause after calculating k-variance',
                      default=False, action='store_true', dest='pause')
  # When specifying a class of interest (COI)
  dataops.add_option('--coi', help='Class of interest.',
                      default=None, type=str, action='store', dest='coi')
  dataops.add_option('--coi-action', help='What to do when a COI is found (keep or seek).',
                      default=None, type=str, action='store', dest='coiaction')
  dataops.add_option('--n-coi', help='Exit after n items of class of interest found. \nUsed with coi-keep or coi-seek.',
                      default=-773038, type=int, action='store', dest='n_coi')
  dataops.add_option('--always-update', help='Always update model, ignoring COI.  Still use COI for output.  Total hack.',
                      default=False, action='store_true', dest='alwaysupdate')
                                     
  parser.add_option_group(dataops)
  
  # Other options                
  parser.add_option('--config', help='Specify config file other than demud.config', 
                      default=None, type='string', action='store', dest='config')
  parser.add_option('--make-config', help='Re-create demud.config with empty variables; exit.',
                      default=False, action='store_true', dest='clean')
  parser.add_option('--easter-eggs', '--fun', help=optparse.SUPPRESS_HELP, default=False, 
                      action='store_true', dest='fun')

  parser.add_option('--svdmethods', help='Provide details on available SVD methods and exit',
                      default=False, action='store_true', dest='svd_print')
  parser.add_option('--scoremethods', help='Provide details on available scoring methods and exit',
                      default=False, action='store_true', dest='score_print')
  parser.add_option('--missingdatamethods', help='Provide details on missing data methods and exit',
                      default=False, action='store_true', dest='md_print')
  parser.add_option('--featureweightmethods', help='Provide details on feature weight methods and exit',
                      default=False, action='store_true', dest='fw_print')

  parser.add_option('--print-default-k', help='Provide details on default values for --k and exit',
                      default=False, action='store_true', dest='printk')
  parser.add_option('--plot-variance', help='Make a plot of k vs. variance explained.',
                      default=False, action='store_true', dest='plotvariance')
                      
  (options, args) = parser.parse_args()

  return vars(options)
  
#______________________________check_opts__________________________________
# Ensure that the arguments supplied make sense
def  check_opts(datatypes):

  # Check if a function argument was supplied
  global opts, use_max_n
  if (log.opts['svd_print'] or log.opts['md_print'] or log.opts['clean'] 
      or log.opts['printk'] or log.opts['score_print'] or log.opts['fw_print']):
    if len(sys.argv) == 3 and log.opts['fun']:
        pass
    elif len(sys.argv) > 2:
      printt("Error: conflicting arguments.  Use -h for help.")
      exit()
    if log.opts['svd_print']:
      svd_print()
    elif log.opts['printk']:
      print_default_k_values()
    elif log.opts['md_print']:
      md_print()
    elif log.opts['fw_print']:
      fw_print()
    elif log.opts['clean']:
      clean()
    elif log.opts['score_print']:
      score_print()
    else:
      printt("Python is tired now.  Go bother somebody else.")
      exit()

  # Check to make sure that exactly one datatype argument was supplied.
  sum = 0
  selected = None
           
  for k in log.opts: 
    if log.opts[k] == True and k in datatypes: 
      sum += 1
      selected = k
  
  if sum != 1:
    printt("Error: Exactly one datatype argument must be supplied.  Use -h for help.")
    exit()
    
  # Check to make sure that --k and --variance are not both specified
  if log.opts['k'] != -773038 and log.opts['k_var'] != -773038.0:
    printt("Error: conflicting arguments: --k and --variance.  Use -h for help.")
    exit()
    
  # Check to make sure that --missingdatamethod has an appropriate argument
  mdmethods = ('none', 'ignore', 'zero')
  if (log.opts['missingdatamethod'] != None):
    if (log.opts['missingdatamethod'] not in mdmethods):
      printt("Error: missing data method %s not supported." % 
             log.opts['missingdatamethod'])
      printt("Choose between 'zero', 'ignore', and 'none'.")
      printt("Use --missingdatamethods for more info.")
      exit()
          
  # Check to make sure that --svdmethod has an appropriate argument
  if log.opts['svdmethod'] != 'increm-ross' and log.opts['svdmethod'] != 'default' and log.opts['increm']:
    printt("Error: cannot specify --increm along with different svdmethod.")
    printt("Use --svdmethods for more info.")
    exit()
  
  if log.opts['svdmethod'] == 'default': log.opts['svdmethod'] = 'full'    
  
  if log.opts['increm']: 
    log.opts['svdmethod'] = 'increm-ross'
    printt("Using increm")
    
  svdmethods = ('full', 'increm-ross', 'increm-brand')
  if (log.opts['svdmethod'] != None):
    if (log.opts['svdmethod'] not in svdmethods):
      printt("Error: SVD method %s not supported." % log.opts['svdmethod'])
      printt("Choose between 'full', 'increm-ross', and 'increm-brand'.")
      printt("Use --svdmethods for more info.")
      exit()
      
  # Check to make sure that --scoremethod has an appropriate argument
  scoremethods = ('low', 'high', 'lowhigh')
  if (log.opts['scoremethod'] != None):
    if (log.opts['scoremethod'] not in scoremethods):
      printt("Error: score method %s not supported." % log.opts['scoremethod'])
      printt("Choose between 'low', 'high', and 'lowhigh'.")
      printt("Use --scoremethods for more info.")
      exit()
  
  # Check to make sure that --shotnoisefilt was supplied only with a valid argument
  if log.opts['shotfilt'] > 0:
    if log.opts['shotfilt'] < 3:
      printt('Error: Shot noise filter is only meaningful for values >= 3.  Odd values are best.')
      exit()
    if not log.opts['chemcam'] and not log.opts['ucis']:
      printt('Error: Shot noise filter is only used for ChemCam (-c) or UCIS (-u) data.')
      exit()

  # Check to make sure that --fft was supplied only with a valid argument
  if (log.opts['fft']):
    if not (log.opts['kepler']):
      printt("Error: FFT not supported with datatype %s" % selected)
      exit()
      
  # Check to make sure that --lookup was only supplied with a valid argument
  if (log.opts['lookup']):
    if not (log.opts['kepler']):
      printt("Error: --lookup supplied with invalid datatype.  Use -h for help.")
      exit()
      
  # Check to make sure that --no-report was only supplied with a valid argument
  if not (log.opts['report']):
    if not (log.opts['glass'] or log.opts['iris'] or log.opts['ecoli'] or 
            log.opts['abalone'] or log.opts['isolet']):
      printt("Error: --no-report supplied with invalid datatype.  Use -h for help.")
      exit()
  if selected not in ['glass', 'iris', 'ecoli', 'abalone', 'isolet']:
    log.opts['report'] = False
      
  # Check to make sure that a valid value of k or k_var and n were given
  if (log.opts['k'] != -773038 and log.opts['k'] < 1):
    printt("Error: bad argument to --k.  Number of PCs must be at least 1.")
    exit()
  if (log.opts['k_var'] != -773038.0 and (log.opts['k_var'] < 0 or log.opts['k_var'] > 1)):
    printt("Error: bad argument to --variance.  Must be between 0.0 and 1.0.")
    exit()
  if (log.opts['n'] != -773038 and log.opts['n'] < 1):
    printt("Error: bad argument to --n.  Number of iterations must be at least 1.")
    exit()

  # Check specified sol number for nonnegative and appropriate data type
  if log.opts['sol'] > -1 and log.opts['sols'] != None:
    printt("Error: Can only use either -sol or -sols, not both.")
    exit()
  elif log.opts['sol'] > -1:
    if not log.opts['chemcam']:
      printt("Error: Sol number specification is only supported for ChemCam (-c).")
      exit()
    else:
      log.opts['start_sol'] = log.opts['sol']
      log.opts['end_sol']   = log.opts['sol']
  # If a sol range was specified, use that
  elif log.opts['sols'] != None:
    svals = log.opts['sols'].split('-')
    if len(svals) != 2:
      printt("Error parsing start and end sols from %s (format: s1-s2)." % log.opts['sols'])
      exit()
    (start, end) = map(int, svals)
    if start >= 0 and end >= 0 and start <= end:
      printt("Analyzing data from sols %d-%d, inclusive." % (start, end))
      log.opts['start_sol'] = start
      log.opts['end_sol']   = end
    else:
      printt("Error parsing start and end sols from %s." % log.opts['sols'])
      exit()
   
  # Check to see if n-coi was given
  if (log.opts['n_coi'] != -773038 and log.opts['coi'] == None):
    printt("Error: cannot supply --n-coi without specifying --coi.")
    exit()
  if (log.opts['n_coi'] > 0 and (log.opts['n_coi'] >= log.opts['n'])):
    use_max_n = True
  # Check the coiaction
  if (log.opts['coiaction'] and log.opts['coi'] == None):
    printt("Eror: cannot specify --coi-action without specifying --coi.")
    exit()
  if (log.opts['coiaction'] not in [None, 'keep', 'seek']):
    printt("Error: --coi-action must be specified as 'keep' or 'seek'.")
    exit()
    
  # Check to see if all was given
  if (log.opts['all']):
    printt("Using the whole data set.")
    use_max_n = True

  return selected
  
#______________________________parse_config_term___________________________
def  parse_config_term(config, term):
  """parse_config_term(config, term)
  Search for term in config content and return its value (after = sign).
  - config: result returned by readlines() applied to config file
  - term: a string
  """

  # Matching lines
  lines = [line for line in config if line.startswith(term)]

  # This term may not be defined in the config file
  if lines == []:
    return ''

  # If the term is used multiple times, it uses the last one
  return lines[-1].split('=')[-1].strip().replace("'", "").replace('"', '')
  
#______________________________parse_config________________________________
def  parse_config(config, data_choice):
  """parse_config(config, data_choice):
  Parse out the filenames needed for the data set of choice.
  - config: result returned by readlines() applied to config file
  - data_choice: string such as 'glass' or 'kepler'
    (assume already validated; returned by check_opts())
  """

  # UCI data
  ucidatafile     = parse_config_term(config, 'ucidatafile')

  # Floating point data (or Pancam, APF, GBT, CNN, or DAN)
  floatdatafile   = parse_config_term(config, 'floatdatafile')

  # GBT filterbank
  gbtdirname      = parse_config_term(config, 'gbtdirname')
  catalogfile     = parse_config_term(config, 'catalogfile')

  # DECaLS
  decalsfilename  = parse_config_term(config, 'decalsfilename')

  # DES
  desfilename     = parse_config_term(config, 'desfilename')

  # ChemCam
  libsdatafile     = parse_config_term(config, 'libsdatafile')
  libsinitdatafile = parse_config_term(config, 'libsinitdatafile')

  # FINESSE
  finessedirname = parse_config_term(config, 'finessedirname')

  # MISR
  misrrawdirname = parse_config_term(config, 'misrrawdirname')
  misrAODdirname = parse_config_term(config, 'misrAODdirname')
  misrdatafile   = parse_config_term(config, 'misrdatafile')

  # AVIRIS
  avirisdirname  = parse_config_term(config, 'avirisdirname')
  avirisrawfile  = parse_config_term(config, 'avirisrawfile')
  avirissegmap   = parse_config_term(config, 'avirissegmap')

  # IRS (Spitzer)
  irsdatafile    = parse_config_term(config, 'irsdatafile')
  irsdatefile    = parse_config_term(config, 'irsdatefile')
  irsmodelfile   = parse_config_term(config, 'irsmodelfile')
  irswavecalfile = parse_config_term(config, '  irswavecalfile')

  # Kepler
  keplerdatafolder    = parse_config_term(config, 'keplerdatafolder')
  keplerdataextension = parse_config_term(config, 'keplerdataextension')
  
  # Mastcam
  mastcamdatafolder   = parse_config_term(config, 'mastcamdatafolder')

  # Images
  imagedatafolder       = parse_config_term(config, 'imagedatafolder')
  imageinitdatafolder   = parse_config_term(config, 'imageinitdatafolder')
  
  # Texturecam (image)
  tcfilename     = parse_config_term(config, 'tcfilename')
  tcpklname      = parse_config_term(config, 'tcpklname')

  # Navcam
  datafolder     = parse_config_term(config, 'datafolder')
  solnumber      = parse_config_term(config, 'solnumber')
  initdatasols   = parse_config_term(config, 'initdatasols')
  scaleInvariant = parse_config_term(config, 'scaleInvariant')
  if scaleInvariant != '':
    scaleInvariant = int(scaleInvariant)

  # UCIS
  ucisrawfile    = parse_config_term(config, 'ucisrawfile')
  
  if (data_choice == 'glass' or
      data_choice == 'iris' or
      data_choice == 'abalone' or
      data_choice == 'isolet' or
      data_choice == 'ecoli'):
    return ([ucidatafile],'')
  elif data_choice in ['pancam', 'testdata', 'apf', 'dan', 'gbt', 'cnn']:
    return ([floatdatafile],'')
  elif data_choice == 'gbtfil':
    return ([gbtdirname, catalogfile],'')
  elif data_choice == 'decals':
    return ([decalsfilename],'')
  elif data_choice == 'des':
    return ([desfilename],'')
  elif data_choice == 'chemcam' or data_choice.startswith('libs'):
    return ([libsdatafile, libsinitdatafile],'')
  elif data_choice == 'finesse':
    return ([finessedirname],'')
  elif data_choice == 'misr':
    return ([misrrawdirname, misrAODdirname, misrdatafile],'')
  elif data_choice == 'aviris':
    return ([avirisdirname, avirisrawfile, avirissegmap],'')
  elif data_choice == 'irs':
    return ([irsdatafile, irsdatefile, ismodelfile, irswavecalfile],'')
  elif data_choice == 'kepler':
    return ([keplerdatafolder], keplerdataextension)
  elif data_choice == 'texturecam':
    return ([tcfilename, tcpklname],'')
  elif data_choice == 'mastcam':
    return ([mastcamdatafolder],'')
  elif data_choice == 'images':
    return ([imagedatafolder, imageinitdatafolder],'')
  elif data_choice == 'navcam':
    # Parse through initdatasols to convert it into a tuple
    initsols = [];
    start_sols = [];
    end_sols = [];
    if len(initdatasols):
        for sols in initdatasols.split(','):
            sols.replace(" ", "")
            if(len(start_sols) > len(end_sols)):
                if not (sols[-1] == ')'):
                    printt('Error: unrecognized data set %s.' % data_choice)
                    printt('Incorrect initdatasols format.')
                    printt("Example initdatasols: '(1950,1955),1959'")
                end_sols.append(int(sols[:4]));
                continue;
            
            if sols[0] == '(':
                start_sols.append(int(sols[1:]));
            else:
                initsols.append(int(sols));

        for start,end in zip(start_sols, end_sols):
            initsols.extend(range(start,end + 1));
    
    return ([datafolder], (int(solnumber), initsols, scaleInvariant))

  elif data_choice == 'ucis':
    return ([ucisrawfile],'')

  printt('Error: unrecognized data set %s.' % data_choice)
  return ()

#______________________________optimize_k__________________________________
def  optimize_k(ds, v):
  """optimize_k(ds, v):
  choose k intelligently to capture v% of the data variance.
  Does a full SVD (inefficient; redundant with first call to update_model).
  Assumes missingmethod = zero.
  """
  assert v >= 0.0 and v <= 1.0
  
  # If initialization data is present, optimize k using that data
  if len(ds.initdata) > 0:
    X = ds.initdata
  # Otherwise analyze the new data 
  else:
    X = ds.data
  
  if X == []:
    printt("Error: No data in input.")
    exit()

  # Handle NaNs with zeros for this lookup
  z = np.where(np.isnan(X))
  if z[0] != []:
    printt("Filling NaNs with 0: %d of %d total." % \
          (z[0].shape[1], X.shape[0] * X.shape[1]))
    X = copy.deepcopy(X)
    X[z] = 0
  
  # Set up svd so we can operate on S
  mu      = np.mean(X, axis=1).reshape(-1,1)
  X       = X - mu
  U, S, V = linalg.svd(X, full_matrices=False)

  # Calculate percent variance captured
  cumsum = np.cumsum(S)
  cumpercents = cumsum / cumsum[-1]
  if len(cumpercents) < 22:
    printt("Cumulative percents:\n%s" % cumpercents)
  else:
    printt("Cumulative percents:\n%s" % cumpercents[:20])
  percents = S / cumsum[-1]
  #minind = np.argmin(abs(cumpercents - v))
  minind = [i for i, x in enumerate(cumpercents) if x > v][0]
  
  printt("\nChose k = %d, capturing %5.2f%% of the data variance\n" % \
        (minind + 1, 100 * cumpercents[minind]))
  
  if log.opts['pause']: raw_input("Press enter to continue\n")
  
  return minind + 1
  
#______________________________plot_variance_______________________________
def plot_variance(ds):
  X = ds.data
  U, S, V = linalg.svd(X, full_matrices=False)
  pylab.plot([qq+1 for qq in range(S.shape[0])], [sum(S[:a+1]) / float(sum(S)) for a in range(len(S))], '-r')
  pylab.xlabel('Number of PCs')
  pylab.ylabel('Percentage of variance explained')
  pylab.title('PCs vs. variance for dataset %s' % ds.name.split('-')[0])

  outdir = os.path.join('results', ds.name)
  if not os.path.exists(outdir):
    os.mkdir(outdir)
  pylab.savefig(os.path.join(outdir, '__variance.pdf'))
  
#______________________________init_default_k_values_______________________
def  init_default_k_values():
  global default_k_values

  default_k_values = {
    'glass'       :  5,
    'iris'        :  3,
    'ecoli'       :  6,
    'pancam'      :  2,
    'apf'         :  2,
    'cnn'         : 50,
    'dan'         :  2,
    'gbt'         : 10,
    'gbtfil'      : 10,
    'decals'      : 10,
    'des'         : 10,
    'testdata'    :  2,
    'chemcam'     : 10,
    'finesse'     : 10,
    'misr'        : 2, #10,
    'abalone'     : 4,
    'isolet'      : 20,
    'aviris'      : 10,
    'irs'         : 10,
    'kepler'      : 50,
    'texturecam'  : 10,
    'navcam'      : 10,
    'mastcam'     : 2,
    'images'      : 10,
    'ucis'        : 10,
  }
  
#______________________________print_default_k_values______________________
def  print_default_k_values():
  init_default_k_values()
  global default_k_values
  printt(default_k_values)
  exit()
  
#______________________________load_data___________________________________
def load_data(data_choice, data_files, sol_number = None, initsols = None, scaleInvariant = None):

  ###########################################################################
  ## GLASS DATA SET (classification)
  if data_choice == 'glass': 
    ds = GlassData(data_files[0])
  ## IRIS DATA SET (classification)
  elif data_choice == 'iris': 
    ds = IrisData(data_files[0])
  ## ABALONE DATA SET (classification)
  elif data_choice == 'abalone': 
    ds = AbaloneData(data_files[0])
  ## E. COLI DATA SET (classification)
  elif data_choice == 'ecoli':
    ds = EcoliData(data_files[0])
  ## ISOLET DATA SET (classification)
  elif data_choice == 'isolet':
    ds = IsoletData(data_files[0])
  ## PANCAM SPECTRA DATA SET
  elif data_choice == 'pancam':
    ds = PancamSpectra(data_files[0])
  ## APF SPECTRA DATA SET
  elif data_choice == 'apf':
    ds = APFSpectra(data_files[0])
  ## CNN FEATURE DATA SET
  elif data_choice == 'cnn':
    ds = CNNFeat(data_files[0])
  ## DAN SPECTRA DATA SET
  elif data_choice == 'dan':
    ds = DANSpectra(data_files[0])
  ## GBT SPECTRA DATA SET
  elif data_choice == 'gbt':
    ds = GBTSpectra(data_files[0])
  ## GBT FILTERBANK DATA SET
  elif data_choice == 'gbtfil':
    ds = GBTFilterbankData(data_files[0], data_files[1])
  ## DECALS FITS DATA SET
  elif data_choice == 'decals':
    ds = DECaLSData(data_files[0])
  ## DES FITS DATA SET
  elif data_choice == 'des':
    ds = DESData(data_files[0])
  ## TEST DATA SET
  elif data_choice == 'testdata':
    ds = Floats(data_files[0])
  ## CHEMCAM DATA SET
  elif data_choice == 'chemcam' or data_choice.startswith('libs'):
    ds = LIBSData(data_files[0], data_files[1],
                  startsol       = log.opts['start_sol'],
                  endsol         = log.opts['end_sol'],
                  initpriorsols  = log.opts['init_prior_sols'],
                  shotnoisefilt  = log.opts['shotfilt'])
  ## FINESSE DATA SET
  elif data_choice == 'finesse':
    ds = FINESSEData(data_files[0])
  ## MISR AEROSOL DATA SET
  elif data_choice == 'misr':
    printt("I see that you want to look at MISR data!  Very cool.")
    printt("I highly recommend using the following parameters:")
    printt("  --missingdatamethod=ignore")
    printt("  --svdmethod=increm-brand")
    printt("  --init-item=mean")
    printt("Using all defaults (zero, full, 0) will also work with worse results.")
    printt("Behavior on other parameter combinations is not predictable.")
    printt("")
    printt("Continuing with execution...")
    printt("")
    # raw_input('Press enter to continue or enter ^C/EOF to abort.')


    ds = MISRDataTime(data_files[0], data_files[1], data_files[2])
    
    '''
    #
    # THIS IS ALL OLD STUFF FROM KIRI WHICH IS NOW DEAD.
    #   keeping it for posterity and because things might be informative.
    #

    # The following calls bypass command-line options given
    origname = ds.name
    #sels = demud(ds, k=k, nsel=10)
    # Selections (for k=10) should be:
    # [168, 128, 24, 127, 153, 108, 188, 103, 0, 64]
    ds.name = origname

    sels , sels_idx = demud(ds, k=k, nsel=1, svdmethod='increm-ross')
    # Selections should be:
    # k=10 [168, 128, 24, 159, 127, 153, 47, 108, 188, 64]
    # k=5  [128, 159, 24, 127, 47, 153, 188, 108, 64, 0]
    ds.name = origname

    sels , sels_idx = demud(ds, k=k, nsel=1,
                 svdmethod='increm-ross',
                 missingmethod='ignore')
    # Selections should be:
    # k=10 [128, 169, 130, 150, 40, 195, 84, 70, 194, 175]
    # k=5  [128, 169, 24, 40, 135, 185, 139, 127, 16, 36]
    # [7, 0, 3, 2, 4, 5, 1, 6, 9, 8] # for 10
    ds.name = origname

    sels, sels_idx  = demud(ds, k=k, nsel=1,
                 scoremethod='high',
                 svdmethod='increm-ross',
                 missingmethod='ignore')
    # Selections should be:
    # k=10 
    # k=5  [128, 24, 127, 0, 152, 153, 159, 120, 46, 52]
    ds.name = origname
    '''

  ## AVIRIS DATA SET
  elif data_choice == 'aviris':
    #ds = ENVIData(avirisrawfile)
    ds = SegENVIData(data_files[1], data_files[2])
    ds.write_RGB(data_files[0] + 'f970619t01p02_r02_sc04.png')    
  ## SPITZER IRS DATA SET
  elif data_choice == 'irs':
    ds = IRSData(data_files[0], data_files[1], data_files[2], data_files[3])
  ## KEPLER DATA SET (accepts fft)
  elif data_choice == 'kepler':
    ds = KeplerData(data_files[0], etc)
  ## MASTCAM DATA SET (accepts fft)
  elif data_choice == 'mastcam':
    ds = MastcamData(data_files[0])
  ## IMAGE DATA SET
  elif data_choice == 'images':
    ds = ImageData(data_files[0], data_files[1])
  ## TEXTURECAM DATA SET
  elif data_choice == 'texturecam':
    ds = TCData(data_files[0], data_files[1])
    ds.selections = np.zeros((ds.height, ds.width))
  ## NAVCAM
  elif data_choice == 'navcam':
    ds = NavcamData(data_files[0], sol_number, initsols, scaleInvariant); 
  ## UCIS (ENVI) DATA SET
  elif data_choice == 'ucis':
    ds = ENVIData(data_files[0], 
                  shotnoisefilt = log.opts['shotfilt'],
                  fwfile        = log.opts['fw'])
    ds.write_RGB(os.path.join(os.path.dirname(data_files[0]),
                              'envi-rgb.png'))
  else:
    ## should never get here
    printt("Invalid data set choice.")
    exit()

  printt("datatype ", type(ds.data))

  if ds.data.shape[1] != len(ds.labels):
    printt("Error: %d items but %d labels!" % (ds.data.shape[1],
                                               len(ds.labels)))
    exit()

  return ds
  
#______________________________main________________________________________
# Main execution
def  main():

  printt("DEMUD version " + __VERSION__ + "\n")
  
  log.opts = parse_args()
  log.opts['start_sol'] = None
  log.opts['end_sol']   = None
  
  ###########################################################################
  ## Check to ensure a valid set of arguments was given.
  
  datatypes = ('glass', 'iris', 'ecoli',  'abalone', 'isolet',
               'chemcam', 'finesse', 'misr', 'aviris',
               'irs', 'kepler', 'texturecam', 'navcam',
               'pancam', 'apf', 'dan', 'gbt', 'gbtfil', 'decals',
               'mastcam', 'images', 'ucis', 'testdata', 'cnn', 'des')
  
  data_choice = check_opts(datatypes)
  
  (config, fft) = (log.opts['config'], log.opts['fft'])
  (lookup, report) = (log.opts['lookup'], log.opts['report'])   
  (q, sm, mm) = (log.opts['svdmethod'], 
      log.opts['scoremethod'], log.opts['missingdatamethod'])
  
  ###########################################################################
  ## Check for config file and read it in

  # Read in config file
  config = 'demud.config' if log.opts['config'] == None else log.opts['config']
  
  if check_if_files_exist([config], 'config') == False:
    sys.exit(1)
  
  with open(config) as config_file:
    content        = config_file.readlines()
    data_files,etc = parse_config(content, data_choice)

  printt("Elapsed time after parsing args and reading config file:", time.clock())
  
  ###########################################################################
  ## Now we are moving on to the cases which handle each data set.
  ###########################################################################
  
  init_default_k_values()
  
  global ds

  if check_if_files_exist(data_files) == False:
    sys.exit(1)

  if data_choice == 'navcam':  
    sol_number, initsols, scaleInvariant = etc
    ds = load_data(data_choice, data_files, sol_number, initsols, scaleInvariant)
  else:
    ds = load_data(data_choice, data_files)

  # Proceed with final setup
  fw = finish_initialization(ds)

  if data_choice == 'kepler' and fft:
    ds.fftransform()
    fw = finish_initialization(ds, action="performing FFT")

  k = log.opts['k'] if log.opts['k'] != -773038 else default_k_values[data_choice]
  if log.opts['k_var'] != -773038.0: k = optimize_k(ds, log.opts['k_var'])
  n = log.opts['n'] if log.opts['n'] != -773038 else default_n_value
  if log.opts['n'] == -1: n = ds.data.shape[1]

  # Run DEMUD!
  sels, sels_idx = demud(ds, k=k, nsel=n, scoremethod=sm, svdmethod=q,
                         missingmethod=mm, feature_weights=fw,
                         start_sol=log.opts['start_sol'],
                         end_sol=log.opts['end_sol'])

  # Report the results
  if report:
    report_classes(ds, sels, sels_idx, data_choice)

  if (data_choice == 'decals' or 
      data_choice == 'des'):
    # Perform final cleanup of HTML selections file
    outdir = os.path.join('results', ds.name)
    htmlselfile = os.path.join(outdir, 'selections-k%d.html' % k)
    htmlfid = open(htmlselfile, 'a')
    htmlfid.write('</body>\n</html>\n')
    htmlfid.close()

  if data_choice == 'mastcam':
    for i in ds.segmentation:
      seg = ds.segmentation[i]
      break

    image = len(sels) * np.ones(seg.shape)
    for i in range(len(sels)):
      image[seg == sels[i]+1] = i

    pylab.imshow(image)
    pylab.colorbar()
    pylab.savefig(os.path.join('results',
                               '%s-n=%d-segmentation.pdf' % 
                               (ds.name, len(sels))))

    for l in ds.labels:
      img = ds.fullimages[l.split('_')[0]]
      break

    with open(os.path.join('results',
                           '%s-n=%d-segmentation.pkl' % 
                           (ds.name, len(sels))), 'w') as f:
      pickle.dump((sels, seg, img), f)

    Image.fromarray(img).save(os.path.join('results', 
                                           '%s-n=%d-segmentation.png' % 
                                           (ds.name, len(sels))), 'PNG')

      
  if data_choice == 'kepler' and lookup:
    # Output a list of selections, and then lookup their labels
    found = [int(ds.labels[x].split('-')[0].split(':')[0]) for x in sels]
      
    printt(found)
      
    flags = kepler_lookup.lookup(found)
    
    for i in range(len(found)):
      printt("%3d) %d: %s" % (i, found[i], flags[str(found[i])]))
      
  if data_choice == 'navcam':
    # Saves the scores to the output folder
    outdir  = os.path.join('results', ds.name)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    ds.plot_score(outdir)
      
  ###########################################################################

  printt("Total elapsed processor time:", time.clock()) 

  log.logfile.close()
  print "Wrote log to %s\n" % log.logfilename

  if (log.opts['fun']):
    print base64.b64decode('VGhhbmsgeW91LCBjb21lIGFnYWlu')
    print


if __name__ == "__main__":
  main()
  

#####
# CHANGELOG
#  
#  1.2: Interactive feedback and class of interest
#  1.3: First element chosen is data[:,0] unless static SVD
#  1.4: Incremental SVD fully functional; choice of first element
#  1.5: Feature weighting included; full SVD option for init-item=-1 back in
#  1.6: Start of summer 2014, added to include Mastcam support
#  1.7: [inprogress] implementing image processing w/ CNN
#
#####

#####
# To add a new dataset:
#
#  - add the appropriate case in the main method
#  - add argument to list of datatypes at beginning of main method
#  - in the same place, make the local boolean variable
#  - add command-line flag
#  - update check_opts as appropriate (ie, for lookup, fft, etc)
#  - add import
#  - add default k value
#  - add required files to make-config
#
#####



