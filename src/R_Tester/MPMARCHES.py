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
#       do_performance_test:    - Run the performance test
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
                  ("mpmpipe_test"          , "mpmpipe_test.ups"          , 8   , "Linux" , ["exactComparison"]) , 
                  ("methaneFireWContainer" , "methaneFireWContainer.ups" , 1.1 , "Linux" , ["exactComparison", "no_restart"]), 
                  ("heptane_pipe"          , "heptane_pipe.ups"          , 1.1 , "Linux" , ["exactComparison", "no_restart", "no_memoryTest"]), 
                  ("coal_table_pipe"       , "coal_table_pipe.ups"       , 1.1 , "Linux" , ["exactComparison", "no_restart", "no_memoryTest"]),
                  ("hot_block"             , "hot_block.ups"             , 1.1 , "Linux" , ["exactComparison", "no_restart"]),
                  ("intrusion_test"        , "intrusion_test.ups"        , 1.1 , "Linux" , ["exactComparison"])
               ]
               
LOCALTESTS =   [  
                  ("mpmpipe_test"          , "mpmpipe_test.ups"          , 8   , "All" , ["exactComparison"]) , 
                  ("methaneFireWContainer" , "methaneFireWContainer.ups" , 1.1 , "All" , ["exactComparison", "no_restart"]), 
                  ("heptane_pipe"          , "heptane_pipe.ups"          , 1.1 , "All" , ["exactComparison", "no_restart", "no_memoryTest"]),
                  ("coal_table_pipe"       , "coal_table_pipe.ups"       , 1.1 , "All" , ["exactComparison", "no_restart", "no_memoryTest"]),
                  ("hot_block"             , "hot_block.ups"             , 1.1 , "All" , ["exactComparison", "no_restart"]),
                  ("intrusion_test"        , "intrusion_test.ups"        , 1.1 , "All" , ["exactComparison"])
               ]  

#__________________________________
                     
def getNightlyTests() :
  return NIGHTLYTESTS

def getLocalTests() :
  return LOCALTESTS

#__________________________________

if __name__ == "__main__":

  if environ['WHICH_TESTS'] == "local":
    TESTS = LOCALTESTS
  else:
    TESTS = NIGHTLYTESTS

  result = runSusTests(argv, TESTS, "MPMARCHES")
  exit( result )

