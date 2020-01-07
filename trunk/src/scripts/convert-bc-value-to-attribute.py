# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:26:33 2013

@author: tsaad

This script will convert the <value> xml specification in Uintah input files
to an attribute. For example:
        <BCType var="Dirichlet" label="phi_central" id="all">
          <value> 1.0 </value>
        </BCType>
will be converted to
        <BCType var="Dirichlet" label="phi_central" value="1.0" id="all"/>
This will reduce the number of lines in the BC specification by up to 66.6%.
Certain boundary conditions have more than one child associated with them. For
example
        <BCType var="Dirichlet" label="phi_central" id="all">
          <value> 1.0 </value>
          <SomeOtherTag> something </SomeOtherTag>
        </BCType>
This will be converted to:
        <BCType var="Dirichlet" label="phi_central" value="1.0" id="all">
          <SomeOtherTag> something </SomeOtherTag>
        </BCType>
In this case, we were able to remove one line only.

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

# converts <value> to an attribute
def convert_bc_value(fname):
    print 'Processing file: ', fname
    tree = et.parse(fname)
    root = tree.getroot()
    hasBCs = True
    # check if this file has boundary conditions
    if (root.find('Grid')).find('BoundaryConditions') is None:
        hasBCs=False
        return

    hasValue = True
    for face in root.iter('Face'):
        for bc in face.iter('BCType'):
            
            if bc.find('value') is None:
                hasValue=False
                continue

            #count the number of children
            count = 0
            for child in bc:
                count += 1;
            
            if count == 1 and hasValue:
                temp = et.SubElement(face, 'temp', bc.attrib)
                val = bc.find('value').text
                val = val.strip()
                face.remove(bc)
                temp.set('value',val)    
                temp.tail = '\n'
                indent(temp,4)
                indent(face,3)                
            elif count > 1 and hasValue:
                valSpec = bc.find('value')
                val = valSpec.text
                bc.set('value',val)
                bc.remove(valSpec)
                
        for bc in face.iter('temp'):
            bc.tag = 'BCType'
            
    # only write the file when it has boundary conditions
    if hasBCs is True:
        tree.write(fname,pretty_print=True)
        print '  done!'

# start here
if __name__ == "__main__":
   inputdir, fname = main(sys.argv[1:])
   if inputdir != '':
     for fname in glob.glob(inputdir + "/*.ups"):
          convert_bc_value(fname)
   elif fname != '':
     convert_bc_value(fname)
        