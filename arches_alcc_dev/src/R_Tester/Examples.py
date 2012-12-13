#!/usr/bin/env python

from sys import argv,exit
from os import environ
from helpers.runSusTests import runSusTests

#______________________________________________________________________
#  Test syntax: ( "folder name", "input file", # processors, "OS",["flags1","flag2"])
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
NIGHTLYTESTS = [   ("poisson1",         "poisson1.ups",         1, "ALL"),
                   ("RMCRT_test_1L",    "RMCRT_test_1L.ups",    1, "ALL", ["exactComparison"]),
                   ("RMCRT_bm1_DO",     "RMCRT_bm1_DO.ups",     1, "ALL",["exactComparison"]),
                   ("RMCRT_ML",         "RMCRT_ML.ups",         1, "ALL", ["exactComparison"]),
                   ("RMCRT_VR",         "RMCRT_VR.ups",         1, "ALL", ["exactComparison"]),
                   ("RMCRT_1L_reflect", "RMCRT_1L_reflect.ups", 1, "ALL", ["exactComparison"])
               ]

# Tests that are run during local regression testing
LOCALTESTS   = [   ("RMCRT_test_1L",    "RMCRT_test_1L.ups",    1, "ALL", ["exactComparison"]),
                   ("RMCRT_bm1_DO",     "RMCRT_bm1_DO.ups",     1 , "ALL",["exactComparison"]),
                   ("RMCRT_ML",         "RMCRT_ML.ups",         1, "ALL", ["exactComparison"]),
                   ("RMCRT_VR",         "RMCRT_VR.ups",         1, "ALL", ["exactComparison"]),
                   ("RMCRT_1L_reflect", "RMCRT_1L_reflect.ups", 1, "ALL", ["exactComparison"]), 
               ]
                 #  ("RMCRT_bm1_DO",     "RMCRT_bm1_DO.ups",     1, "ALL", ["exactComparison"])

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
  result = runSusTests(argv, TESTS, "Examples")
  exit( result )
