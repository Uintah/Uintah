# -*- coding: utf-8 -*-
"""
Created on Monday Oct 13 2014

@author: Tony Saad

This script will update all initialization BasicExpressions in Wasatch to use STATE_NONE.

***************
USAGE:
change-initialization-state.py -i <inputfile> -d <path>

This script can be used to convert a single ups file OR all ups files in a given
directory, specified by -d. You cannot specify BOTH -i and -d.

***************
EXAMPLE:
change-initialization-state.py -d /Users/me/development/uintah/src/StandAlone/inputs/Wasatch
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
    opts, args = getopt.getopt(argv,"hi:d:m:",["ifile=","dir="])
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

# update all initialization expression to STATE_NONE
def change_initialization_state(fname):
  print 'now processing: ', fname
  
  tree = et.parse(fname)
  root = tree.getroot()
  hasBCs = True
  needsSave = False
  
  for expr in root.iter('BasicExpression'):
    tasklist = expr.find('TaskList')
    if tasklist.text == 'initialization':
      nametag = expr.find('NameTag')
      if nametag.get('state') == 'STATE_DYNAMIC':
        nametag.attrib['state'] = 'STATE_NONE'
        needsSave = True
        print 'changing state for ', nametag.get('name')
    else:
      continue
        
  # only write the file when it has ids specified
  if needsSave == True:
    tree.write(fname,pretty_print=True)
    print '  done!'  

if __name__ == "__main__":
   fname, inputdir = main(sys.argv[1:])
   if inputdir != '':
     for fname in glob.glob(inputdir + "/*.ups"):
          change_initialization_state(fname)
   elif fname != '':
     change_initialization_state(fname)