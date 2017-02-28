#!/usr/bin/env python
# Interactive DEMUD GUI for experiments
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

import Tkinter as tk
from PIL import Image, ImageTk
import tkFont

# iDEMUD GUI inherits from tk.Frame
class iDEMUD_GUI(tk.Frame):
    sel_ind = 0
    
    def __init__(self, master=None):
        tk.Frame.__init__(self, master, class_='iDEMUD_GUI')

        self.sel_var = tk.StringVar(self)
        self.sel_var.set('Selection %d' % self.sel_ind)

        # Grid layout
        self.grid()  
        # Make GUI appear on screen
        self.createWidgets()

    def chooseInteresting(self):
        print 'Interesting'
        self.nextSelection()

    def chooseMaybe(self):
        print 'Maybe'
        self.nextSelection()

    def chooseUninteresting(self):
        print 'Uninteresting'
        self.nextSelection()

    def nextSelection(self):
        self.sel_ind = self.sel_ind + 1
        self.sel_var.set('Selection %d' % self.sel_ind)
        
    def createWidgets(self):
        # Show a title
        self.title = tk.Label(self,
                              text = 'Interactive DEMUD',
                              font = ('Helvetica', 24),
                              bg   = '#ccf')
        self.title.grid(ipadx = 100)
        
        # Show selection information
        self.selection = tk.Label(self,
                                  textvariable = self.sel_var)
        self.selection.grid()
        
        # Show an image
        self.image = Image.open('/Users/wkiri/Research/IMBUE/data/mastcam/multispectral_drcl/mastcam-034.jpg')
        self.photo = ImageTk.PhotoImage(self.image)
        self.imglbl = tk.Label(self, image = self.photo)
        self.imglbl.grid()

        # Show the user feedback options:
        # 1. Interesting
        # 2. Maybe
        # 3. Uninteresting
        self.fd = tk.LabelFrame(self,
                                text = 'Select one')
        self.fd.grid(padx = 10)
        self.thumbsup   = ImageTk.PhotoImage(Image.open('/Users/wkiri/Research/IMBUE/git/src/demud/fig-thumbs-up.png'))
        self.thumbsdown = ImageTk.PhotoImage(Image.open('/Users/wkiri/Research/IMBUE/git/src/demud/fig-thumbs-down.png'))
        self.interButton = tk.Button(self.fd,
                                     text    = 'Interesting',
                                     compound = tk.LEFT,
                                     image = self.thumbsup,
                                     command = self.chooseInteresting)
        self.interButton.grid(row=0, column = 0)
        self.maybeButton = tk.Button(self.fd,
                                     text = 'Maybe',
                                     command = self.chooseMaybe)
        self.maybeButton.grid(row=0, column = 1)
        self.unintButton = tk.Button(self.fd,
                                     text = 'Uninteresting',
                                     compound = tk.RIGHT,
                                     image = self.thumbsdown,
                                     command = self.chooseUninteresting)
        self.unintButton.grid(row=0, column = 2)
        
        # Create the quit button
        self.quitButton = tk.Button(self,
                                    text    = 'Quit',
                                    command = self.quit,
                                    background      = '#fcc')
        self.quitButton.grid(ipadx = 30,
                             pady  = 10)

# Main script
gui = iDEMUD_GUI()
gui.master.title('Interactive DEMUD GUI')
# Wait for and process events
gui.mainloop()
