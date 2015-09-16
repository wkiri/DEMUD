#!/usr/bin/env python
# File: log.py
# Author: Kiri Wagstaff, using logging code written by James Bedell, 9/9/13
#
# Support log file creation and printing
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

opts = {}
logfile = None
logfilename = None

#_______________________________________________________________________________
#____________________________________printt_____________________________________
#

# USE THIS FOR ANY AND ALL OUTPUT THAT YOU WANT TO GET WRITTEN TO THE LOG FILE
def printt(*args):
  """printt(*args)
  
  Same exact functionality as print *args, except output will be written to log
    file as well as stdout.  Similar to Unix command "tee", hence the extra t.
    
  If the logfile has not been initialized, same as print.
  """
  
  printed = ''
  for s in args:
    printed += str(s) + ' '
  
  printed = printed[:-1]
  print printed
  
  if opts != {} and opts['log']:
    if logfile != None:
      logfile.write(printed + '\n')
      logfile.flush()

