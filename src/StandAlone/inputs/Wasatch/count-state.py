# -*- coding: utf-8 -*-
"""
Created on Monday Oct  13 2014

@author: Tony Saad

This script will count the number of expression states that exist in wasatch-like xml input files.

***************
USAGE:
count-state.py -i <inputfile> -d <path>

This script can be used to count the Expr::State in a single ups file OR all ups files in a given
directory, specified by -d. You cannot specify BOTH -i and -d.

***************
EXAMPLE:
count-state.py -d /Users/me/development/uintah/src/StandAlone/inputs/Wasatch
will process all ups files in the Wasatch inputs directory


"""
from __future__ import division # float division
import lxml.etree as et
import sys, getopt
import glob

def help():
  print 'usage: count-state.py -i <inputfile> -d <directory>'

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

  return (inputfile, inputdir)

# goes through all xml blocks in an input file and counts the number of different expression states
def count_state(fname):
  print 'now processing: ', fname
  
  tree = et.parse(fname)
  root = tree.getroot()
  statenone = 0
  statedynamic=0
  staten=0
  stateforward=0
  for nametag in root.iter('NameTag'):
    state = nametag.get('state')
    if state == 'STATE_NONE':
      statenone += 1
    elif state == 'STATE_DYNAMIC':
      statedynamic += 1
    elif state == 'STATE_N':
      staten +=1
    elif state == 'CARRY_FORWARD':
      stateforward +=1
  
  allstates = statedynamic + statenone + staten + stateforward
  if allstates == 0:
    allstates = 1
  print 'STATE_DYNAMIC: ', statedynamic/allstates*100, '%'
  print 'STATE_NONE: ', statenone/allstates*100, '%'
  print 'STATE_N: ', staten/allstates*100, '%'
  print 'CARRY_FORWARD: ', stateforward/allstates*100, '%'
  print '--------------------------'
  return statedynamic, statenone, staten, stateforward
  # only write the file when it has ids specified
  #if needsSave == True:
    #tree.write(fname,pretty_print=True)
    #print '  done!'  

if __name__ == "__main__":
   fname, inputdir = main(sys.argv[1:])
   if inputdir != '':
     statedynamic = 0
     statenone = 0
     staten = 0
     stateforward  = 0
     for fname in glob.glob(inputdir + "/*.ups"):
          statedynamic_, statenone_,staten_, stateforward_ = count_state(fname)
          statedynamic += statedynamic_
          statenone += statenone_
          staten += staten_
          stateforward += stateforward_
      
     allstates = statedynamic + statenone + staten + stateforward
     print '************************'
     print 'OVERALL STATS '
     print '************************'     
     print 'STATE_DYNAMIC: ', statedynamic/allstates*100, '%'
     print 'STATE_NONE: ', statenone/allstates*100, '%'
     print 'STATE_N: ', staten/allstates*100, '%'
     print 'CARRY_FORWARD: ', stateforward/allstates*100, '%'      
     print '************************'          
   elif fname != '':
     count_state(fname)