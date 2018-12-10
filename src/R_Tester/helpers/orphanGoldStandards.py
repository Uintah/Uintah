#! /usr/bin/env python3
from .runSusTests import getTestName
import os
import sys
import importlib

#______________________________________________________________________
#  This script searches the gold standards directories for gold standards
#  that don't have a corresponding entry in the <component.py> file
#______________________________________________________________________

components = [ "ARCHES", "ICE", "IMPM", "Examples", "Models", "MPMICE", "MPM", "UCF", "Wasatch" ]
#components = [ "ICE" ]

whichTests = "NIGHTLYTESTS"
dbgOpt     = "opt"       # dbg or opt

# define paths
GS_dir     =  os.path.abspath( "/home/rt/Linux/TestData/%s" % dbgOpt )
srcDir     =  os.path.abspath( "/home/rt/Linux/last_ran.lock/src" )
inputDir   =  os.path.abspath( "%s/StandAlone/inputs" % srcDir )
RT_Dir     =  os.path.abspath( "%s/R_Tester" % srcDir)

# add path when searching for component.py
sys.path.append(RT_Dir)

# hack so the component.py module can find the inputs dir
sys.argv.append('null')
sys.argv.append(inputDir)

#__________________________________
for component in components :
  print("__________________________________________%s:%s" % (dbgOpt, component))
  print(" The following gold standards are orphans")

  #define gold standard path
  gs_path = "%s/%s" % (GS_dir, component)
  
  # get a list of all gold standards
  gs_dirs = os.listdir( os.path.abspath( gs_path ) )

  # pull tests from the <component.py> file
  THE_COMPONENT = importlib.import_module( component )

  RT_tests = THE_COMPONENT.getTestList( whichTests )
  
  #__________________________________    
  # search for orphan gold standards
  for target in gs_dirs:
    isOrphan = True
    
    for test in RT_tests :
      testname = getTestName( test )
      
      if testname == target:
        isOrphan = False
        continue
    
    if( isOrphan ):
      print("  rm -rf %s/%s " % (gs_path, target))
