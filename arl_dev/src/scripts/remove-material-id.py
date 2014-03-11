# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:26:33 2013

@author: tsaad

This script will remove the "id" attribute from BCType and will add a 
<DefaultMaterial> to be used for all BCs that do NOT have an id.
You can specify which default material you want to use via the -m commandline
argument. If -m is not specified, then 'all' is used as the default material.

***************
USAGE:
remove-material-id.py -i <inputfile> -d <path> -m <default_material_id>

This script can be used to convert a single ups file OR all ups files in a given
directory, specified by -d. You cannot specify BOTH -i and -d.

***************
EXAMPLE:
remove-material-id.py -d /Users/me/development/uintah/src/StandAlone/inputs/Wasatch
will process all ups files in the Wasatch inputs directory


"""
import lxml.etree as et
import sys, getopt
import glob

def help():
  print 'usage: remove-material-id.py -i <inputfile> -d <directory> -m <materialID>'
  print 'The material id (-m) argument is optional. If not specified, it will default to all.'

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

# Commandline argument parser
def main(argv):
  inputfile = ''
  inputdir = ''
  defaultmat='all'
  try:
    opts, args = getopt.getopt(argv,"hi:d:m:",["ifile=","dir=","matid="])
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
    elif opt in ("-m", "--matid"):
      defaultmat = arg         

   # error trapping
  if (inputfile=='' and inputdir==''):
    print 'Please specify a valid input file or directory!'
    help()
    sys.exit(2)

  if (inputfile !='' and inputdir!=''):
    print 'You cannot specify both inputfile and inputdir!'
    help()
    sys.exit(2)

  return (inputfile, inputdir, defaultmat)

# Removes the material id and adds a default material
def remove_material_id(fname, defmat):
  print 'now processing: ', fname
  print 'using default material: ', defmat
  
  tree = et.parse(fname)
  root = tree.getroot()
  hasBCs = True
  needsSave = False
  # check if this file has boundary conditions
  if (root.find('Grid')).find('BoundaryConditions') is None:
      hasBCs=False
      return
  
  if hasBCs is True:
      bcSpec = (root.find('Grid')).find('BoundaryConditions')
      defaultMatSpec = et.Element('DefaultMaterial')
      defaultMatSpec.text = defmat
      defaultMatSpec.tail = '\n'
      bcSpec.insert(0,defaultMatSpec)
      indent(defaultMatSpec,3)

  hasID = False
  for face in root.iter('Face'):
    for bc in face.iter('BCType'):
      if bc.get('id') is None:
        hasID = False
      else:
        hasID = True
        if needsSave == False:
          needsSave = True
      if hasID:
        del bc.attrib['id']            
        
  # only write the file when it has ids specified
  if needsSave == True:
    tree.write(fname,pretty_print=True)
    print '  done!'  

if __name__ == "__main__":
   fname, inputdir, defmat = main(sys.argv[1:])
   if inputdir != '':
     for fname in glob.glob(inputdir + "/*.ups"):
          remove_material_id(fname, defmat)
   elif fname != '':
     remove_material_id(fname, defmat)