#! /usr/bin/env python3
from runSusTests import getTestName, getUpsFile, setInputsDir
import os
import sys
import importlib

#______________________________________________________________________
#  This script prints out the path to the ups files 
#  This is useful if you want to run sed/grep on all the ups files
#______________________________________________________________________

components = [ "ARCHES", "ICE", "IMPM", "Examples", "Models", "MPMICE", "MPM", "UCF", "Wasatch" ]
components = [ "Wasatch" ]

whichTests = "NIGHTLYTESTS"

# define paths
srcDir     =  os.path.abspath( "/home/rt/Linux/last_ran.lock/src" )
#srcDir     = os.path.abspath( "/home/harman/Builds/Fresh/03-08-19/src" )
inputDir   =  os.path.abspath( "%s/StandAlone/inputs" % srcDir )
RT_Dir     =  os.path.abspath( "%s/R_Tester" % srcDir)

# add path when searching for component.py
sys.path.append(RT_Dir)

# hack so the component.py module can find the inputs dir
sys.argv.append('null')
sys.argv.append(inputDir)

#__________________________________
for component in components :
  print("__________________________________________%s" % (component), file=sys.stderr)


  # Read tests from the <component.py> file
  THE_COMPONENT = importlib.import_module( component )

  RT_tests = THE_COMPONENT.getTestList( whichTests )
  
  #__________________________________    
  # print the path to the ups files
  warnUser = False
  for test in RT_tests :
      testname = getTestName( test )
      
      ups = getUpsFile( test )
      
      print("%s/%s/%s " % (inputDir, component, ups))
      
      if ( ups.find('tmp') == -1 ):
        warnUser = True;
        
      
  if ( warnUser ):
      print("\n*** Some of the input files listed have been modified.", file=sys.stderr)
