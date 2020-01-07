# -*- coding: utf-8 -*-
"""
Created on Tue Dec 2 2015

@author: tsaad

This script will update the Wasatch <Density> specification to conform to the newly
adopted interface for compressible flows. We have recently added an attribute called 'method' to 
the <Density> block. This attribute allows one to specify whether the simulation is for
a constant density, low Mach, or compressible flow.

This script will convert existing input files by adding the 'method' attribute.

!!!WARNING: THIS SPEC ASSUMES THAT ALL INPUT FILES ARE EITHER CONSTANT DENSITY OR LOW MACH
PROBLEMS. 

***************
USAGE:
convert-bc-value-to-attribute.py -i <inputfile> -d <path>

This script can be used to convert a single ups file OR all ups files in a given
directory, specified by -d. You cannot specify BOTH -i and -d.

***************
EXAMPLE:
convert-bc-value-to-attribute -d /Users/me/development/uintah/src/StandAlone/inputs/Wasatch
will process all ups files in the Wasatch inputs directory

"""
import lxml.etree as et
import sys, getopt
import glob
import re
# Function that indents elements in a xml tree. 
# Thanks to: http://effbot.org/zone/element-lib.htm#prettyprint
def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def help():
  print 'Usage: convert-bc-value-to-attribute -i <inputfile>'

# Commandline argument parser
def main(argv):
   inputfile = ''
   inputdir  = ''
   try:
      opts, args = getopt.getopt(argv,"hi:d:",["ifile=","dir="])
   except getopt.GetoptError:
       help()
       sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         help()
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-d", "--dir"):
         inputdir = arg
   print inputfile
   print inputdir      
   if inputfile=='' and inputdir=='':
     print 'Please specify a valid input file or directory!'
     help()
     sys.exit(2)

   if (inputfile !='' and inputdir!=''):
     print 'You cannot specify both inputfile and inputdir!'
     help()
     sys.exit(2)

   return (inputdir, inputfile)

def update_density_spec(fname):
    print 'Processing file: ', fname
    tree = et.parse(fname)
    root = tree.getroot()
    hasDensity = True

    # check if this is a wasatch input file
    if root.find('Wasatch') is None:
        return

    # check if this file has density specification
    if (root.find('Wasatch')).find('Density') is None:
        hasDensity=False
        return
    
    # get the density xml spec block
    densitySpec = (root.find('Wasatch')).find('Density')

    # check if this density block already has a method attribute    
    if densitySpec.get('method','NONE') != 'NONE':
        print 'found method'
        return

    # find out if this density block has a nametag child. If it does, then (at the time of writing)
    # this means that the input file is a LOWMACH simulation. Make the necessary changes
    if densitySpec.find('NameTag') is None:
        densitySpec.set('method','CONSTANT')
    else:
        densitySpec.set('method','LOWMACH')
                    
    # only write the file when it has a density
    if hasDensity is True:
        tree.write(fname,pretty_print=True)
        print '  done!'

# start here
if __name__ == "__main__":
   inputdir, fname = main(sys.argv[1:])
   if inputdir != '':
     for fname in glob.glob(inputdir + "/*.ups"):
          update_density_spec(fname)
   elif fname != '':
     update_density_spec(fname)
        