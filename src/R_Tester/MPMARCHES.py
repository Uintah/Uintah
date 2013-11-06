#!/usr/bin/env python

from os import symlink,environ
from sys import argv,exit,platform
from helpers.runSusTests import runSusTests, inputs_root
from helpers.modUPS import modUPS

#______________________________________________________________________
#  Test syntax: ( "folder name", "input file", # processors, "OS", ["flags1","flag2"])
#  flags:
#       gpu:                    - run test if machine is gpu enabled 
#       no_uda_comparison:      - skip the uda comparisons
#       no_memoryTest:          - skip all memory checks
#       no_restart:             - skip the restart tests
#       no_dbg:                 - skip all debug compilation tests
#       no_opt:                 - skip all optimized compilation tests
#       do_performance_test:    - Run the performance test, log and plot simulation runtime.
#                                 (You cannot perform uda comparsions with this flag set)
#       doesTestRun:            - Checks if a test successfully runs
#       abs_tolerance=[double]  - absolute tolerance used in comparisons
#       rel_tolerance=[double]  - relative tolerance used in comparisons
#       exactComparison         - set absolute/relative tolerance = 0  for uda comparisons
#       startFromCheckpoint     - start test from checkpoint. (/home/csafe-tester/CheckPoints/..../testname.uda.000)
#       sus_options="string"    - Additional command line options for sus command
#
#  Notes: 
#  1) The "folder name" must be the same as input file without the extension.
#  2) If the processors is > 1.0 then an mpirun command will be used
#  3) Performance_tests are not run on a debug build.
#______________________________________________________________________

NIGHTLYTESTS = [  
#                  ("mpmpipe_test"          , "mpmpipe_test.ups"          , 8   , "Linux" , ["exactComparison"]) , 
#                  ("methaneFireWContainer" , "methaneFireWContainer.ups" , 1.1 , "Linux" , ["exactComparison", "no_restart"]), 
#                  ("hot_block"             , "hot_block.ups"             , 1.1 , "Linux" , ["exactComparison", "no_restart"]),
                  ("intrusion_test"        , "intrusion_test.ups"        , 1.1 , "Linux" , ["exactComparison", "no_restart"])
               ]
               
LOCALTESTS =   [  
#                  ("mpmpipe_test"          , "mpmpipe_test.ups"          , 8   , "All" , ["exactComparison"]) , 
#                  ("methaneFireWContainer" , "methaneFireWContainer.ups" , 1.1 , "All" , ["exactComparison", "no_restart"]), 
#                  ("hot_block"             , "hot_block.ups"             , 1.1 , "All" , ["exactComparison", "no_restart"]),
                  ("intrusion_test"        , "intrusion_test.ups"        , 1.1 , "All" , ["exactComparison", "no_restart"])
               ]  
DEBUGTESTS =[]
#__________________________________
# The following list is parsed by the local RT script
# and allows the user to select the tests to run
#LIST: LOCALTESTS DEBUGTESTS NIGHTLYTESTS
#__________________________________

# returns the list  
def getTestList(me) :
  if me == "LOCALTESTS":
    TESTS = LOCALTESTS
  elif me == "DEBUGTESTS":
    TESTS = DEBUGTESTS
  elif me == "NIGHTLYTESTS":
    TESTS = NIGHTLYTESTS
  else:
    print "\nERROR:MPMARCHES.py  getTestList:  The test list (%s) does not exist!\n\n" % me
    exit(1)
  return TESTS
#__________________________________

if __name__ == "__main__":

  TESTS = getTestList( environ['WHICH_TESTS'] )

  result = runSusTests(argv, TESTS, "MPMARCHES")
  exit( result )

