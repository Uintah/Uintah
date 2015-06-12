# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 16:01:29 2014

@author: tsaad

This script will convert the specification of Convective and Diffusive flux 
expressions into the newer, attribute-based format. The attribute based format
is shorter and more concise.
For example:
      <ConvectiveFlux>
        <Method>KOREN</Method> 
        <Direction>X</Direction>
        <AdvectiveVelocity>
            <NameTag name="u" />
        </AdvectiveVelocity>
      </ConvectiveFlux>
will be converted to
      <ConvectiveFlux direction="X" method="KOREN">
        <AdvectiveVelocity>
            <NameTag name="u" />
        </AdvectiveVelocity>
      </ConvectiveFlux>

and
	<DiffusiveFlux>
		<Direction>X</Direction>
		<ConstantDiffusivity>0.1</ConstantDiffusivity>
	</DiffusiveFlux>
 will be converted to
 	<DiffusiveFlux direction="X" coefficient="0.1"/>

***************
USAGE:
update-flux-expressions.py -i <inputfile> -d <path>

This script can be used to convert a single ups file OR all ups files in a given
directory, specified by -d. You cannot specify BOTH -i and -d.

***************
EXAMPLE:
update-flux-expressions -d /Users/me/development/uintah/src/StandAlone/inputs/Wasatch
will process all ups files in the Wasatch inputs directory

"""
import lxml.etree as et
import sys, getopt
import glob

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
  print 'Usage: update-flux-expressions -i <inputfile>'

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

# converts <value> to an attribute
def update_flux_expressions(fname):
    print 'Processing file: ', fname
    tree = et.parse(fname)
    root = tree.getroot()
              
    for diffFlux in root.iter('DiffusiveFlux'):
      dirBlock = diffFlux.find('Direction')
      if dirBlock is not None:
        mydir = dirBlock.text
        diffFlux.remove(dirBlock)        
        diffFlux.set('direction',mydir)
      
      constCoefBlock = diffFlux.find('ConstantDiffusivity')
      if constCoefBlock is not None:
        diffFlux.set('coefficient',constCoefBlock.text)
        diffFlux.remove(constCoefBlock)
        parent = diffFlux.find('..')
        temp = et.SubElement(parent, 'temp',diffFlux.attrib)
        temp.tail='\n'
        indent(temp,3)
        parent.remove(diffFlux)
        indent(parent,2)
        
    for temp in root.iter('temp'):
      temp.tag = 'DiffusiveFlux'
    
    for convFlux in root.iter('ConvectiveFlux'):
      dirBlock = convFlux.find('Direction')
      if dirBlock is not None:
        mydir = dirBlock.text
        convFlux.remove(dirBlock)        
        convFlux.set('direction',mydir)
      
      methodBlock = convFlux.find('Method')
      if methodBlock is not None:
        convFlux.set('method',methodBlock.text)
        convFlux.remove(methodBlock)

    tree.write(fname,pretty_print=True)

# start here
if __name__ == "__main__":
   inputdir, fname = main(sys.argv[1:])
   if inputdir != '':
     for fname in glob.glob(inputdir + "/*.ups"):
          update_flux_expressions(fname)
   elif fname != '':
     update_flux_expressions(fname)
        