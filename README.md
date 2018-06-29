DEMUD: Discovery via Eigenbasis Modeling of Uninteresting Data
==============================================================
Contact author: Kiri Wagstaff, kiri.wagstaff@jpl.nasa.gov

Contributors: James Bedell, Jake Lee

DEMUD is a data analysis algorithm that incrementally selects the most
interesting or novel item from a data set.  In addition, it provides
explanations for why each item is chosen.  Its incremental approach
minimizes redundancy in selected items; unlike many anomaly detection
systems, it will highlight a particular anomaly only the first time it
is encountered.

Under the hood, DEMUD uses an SVD-based model of the items it selects
and incrementally (1) selects the item that is least well represented
by the current model (i.e., contains the most unexpected information)
and (2) updates its SVD model to "learn" about that item and avoid
selecting similar items in the future.

Installation
------------

You should be able to install demud by simply doing

   `$ python setup.py install`

That will make the `demud` console script available system-wide.

If this doesn't work, use 

   `$ python demud.py [options]` 

in all of the examples below.


To get started
--------------

1. DEMUD has an extensive help message.  Start by running (from `src/`)

   `$ demud -h`

2. Create an empty `demud.config` file by running

   `$ demud --make-config`

3. DEMUD supports a variety of different input data types.  See the
   "Datatype Options:" section of the help message and select the
   appropriate option for your data.  

   **Please do not check your demud.config file back in to this 
   repository.  It is a local configuration file for your system.**

   **Example 1: Images.**

   To run DEMUD on a collection of images using the pixel representation, specify the directory
   containing the images on the `imagedatafolder` line in `demud.config`.
   Then run

   `$ demud -I`

   The results will appear in the `results/` directory under your 
   current directory.  You will also find a log file (`demud.log`)
   and a list of the selections (`selections.csv`).

   **Example 2: UCI data sets (included test cases).** 

   Several UCI data sets are already supported, and the `glass` and
   `ecoli` data sets are provided in the `data/` directory.  You can
   try them out by specifying the appropriate pathname for the data
   files in `demud.config` (see the `glassdatafile` and
   `ecolidatafile` variables), then running

   `$ demud -g`

   `$ demud -e`

   Note: UCI data sets were obtained from

   Lichman, M. (2013). UCI Machine Learning Repository
   [http://archive.ics.uci.edu/ml](http://archive.ics.uci.edu/ml). 
   Irvine, CA: University of California, 
   School of Information and Computer Science.

4. Other data types.

   

   If your data type is not yet supported, consider adding it by 
   (1) adding a new command-line option
   (2) adding parsing support for this option in `demud.py`
   (3) adding a new file called `dataset_yourtype.py` that inherits
   from the `Dataset` class and implements `__init__()`, `readin()`,
   and `plot_item()`.  See existing classes for examples.

Other relevant options
----------------------

There are many other options you can specify for DEMUD, which are
detailed in the help message.  Here are some of the most commonly
used: 

Model options:

* `--k=K`:                 Number of principal components for SVD model; default is specific to data set (demud.py)
* `--variance=K_VAR`:      Optimize --k to capture this much data variance
                           Range: [0.0 1.0]
* `--increm`:              Use an incremental SVD update; usually faster.

Selection and output options:

* `--n=N`, `--iters=N`:    Number of iterations of SVD and selection; default 10
* `--all`:                 Iterate through all data items
* `--init-item=IITEM`:     Index of initialization item (default: 0; -1 or svd
                           for full-data SVD; r for random)

By default, DEMUD starts by selecting the first item in the data set.  You may get more interesting results by using an initial SVD to select the "most anomalous" item from the data set as a starting point, e.g.:

`$ demud -g --init-item=-1`

By default, DEMUD recomputes a full SVD (of the previously selected
items) at each iteration.  If you will be selecting a lot of items,
you may get faster results using an incremental SVD.  See: 

`$ demud --svdmethods`

By default, DEMUD sets any missing values to 0.  You can try different
methods; see: 

`$ demud --missingdatamethods`

By default, DEMUD treats all features equally.  You can specify
different feature weighting methods; see: 

`$ demud --featureweightmethods`

Additional scripts
------------------

`scripts/` provides additional scripts for preprocessing data for use with DEMUD. Usage instructions are included within each subdirectory.

This includes an image feature extraction script used for experiments presented at WHI 2018.

References
----------

1. "Guiding Scientific Discovery with Explanations using DEMUD."
   Kiri L. Wagstaff, Nina L. Lanza, David R. Thompson, Thomas
   G. Dietterich, and Martha S. Gilmore. 
   Proceedings of the Twenty-Seventh Conference on Artificial
   Intelligence (AAAI-13), 2013. 
   [http://wkiri.com/research/papers/wagstaff-demud-13.pdf](http://wkiri.com/research/papers/wagstaff-demud-13.pdf) 

   This paper describes the non-interactive DEMUD algorithm; it
   identifies diverse items within a larger data set for your review.
   The paper reports on results from CRISM and (laboratory) ChemCam
   data analysis. 

2. "Unusual ChemCam Targets Discovered Automatically in Curiosity's
   First Ninety Sols in Gale Crater, Mars." 
   Kiri L. Wagstaff, Nina L. Lanza, and Roger C. Wiens.
   45th Lunar and Planetary Science Conference, March 2014. 
   [http://www.hou.usra.edu/meetings/lpsc2014/pdf/1575.pdf](http://www.hou.usra.edu/meetings/lpsc2014/pdf/1575.pdf) 

   This abstract reports on DEMUD results when applied to Mars data
   collected by ChemCam. 

3. DEMUD was created as part of the IMBUE project.  You can read more
   about IMBUE and access relevant publications and data sets at the 
   IMBUE website:    
   [http://imbue.jpl.nasa.gov/](http://imbue.jpl.nasa.gov/)

