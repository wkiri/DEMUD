#!/usr/bin/env python
# File: montage.py
# When DEMUD has been applied to images, 
# make a 'montage' plot of the first N selections made by DEMUD
#
# Author: Kiri Wagstaff, wkiri@jpl.nasa.gov, 7/28/18
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

import os
import subprocess

# Borrowed from https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


# Create a montage of the first numsel images in selectionsfile,
# reading images from imagedir, and save montage to outdir
def make_montage(selectionsfile, imagedir, numsel, outdir):

    print('Reading image selections from %s' % selectionsfile)

    # Get the number of selections present in selectionsfile
    # Subtract 1 for the header
    sels = file_len(selectionsfile)-1
    # If numsel is not -1 ("all"), adjust sels accordingly
    if numsel != -1 and numsel > 0 and numsel <= sels:
        sels = numsel

    # Name the file montage-k*.jpg (according to selectionsfile)
    outfn = 'montage-%s.jpg' % selectionsfile.split('-')[-1].split('.')[0]

    # Create the montage
    outfile = os.path.join(outdir, outfn)
    os.system("montage `head -%d %s | tail -%d | cut -f3 -d',' | sed 's,^,%s/&,'` %s" % \
                  (sels+1, selectionsfile, sels, imagedir, outfile))

    print('Wrote montage to %s' % outfile)


# Check arguments and call make_montage()
def main(selectionsfile, imagedir, numsel, outdir):

    # Check arguments
    if not os.path.exists(selectionsfile):
        print('Selections file not found: %s' % selectionsfile)
        sys.exit(1)

    if not os.path.isdir(imagedir):
        print('Image directory not found: %s' % imagedir)
        sys.exit(1)

    if not os.path.isdir(outdir):
        print('Creating output directory: %s' % outdir)
        
    make_montage(selectionsfile, imagedir, numsel, outdir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument('selectionsfile', help='DEMUD selections file')
    parser.add_argument('imagedir',       help='Directory containing original images')
    parser.add_argument('-n', '--numsel', default=-1, type=int,
                        help='Number of selections to show (default: all)')
    parser.add_argument('-o', '--outdir', default='.', 
                        help='Output directory (default: . )')
    args = parser.parse_args()
    
    main(**vars(args))
