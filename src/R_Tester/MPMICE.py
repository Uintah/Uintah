#!/usr/bin/env python

from sys import argv,exit
from os import environ
from helpers.runSusTests import runSusTests

#______________________________________________________________________
#  Test syntax: ( "folder name", "input file", # processors, "OS", ["flags1","flag2",...])
#  flags: 
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
#
#  Notes: 
#  1) The "folder name" must be the same as input file without the extension.
#  2) If the processors is > 1.0 then an mpirun command will be used
#  3) Performance_tests are not run on a debug build.
#______________________________________________________________________

NIGHTLYTESTS = [   ("massX",                 "massX.ups",                 1,  "Linux", ["exactComparison"]),   \
                   ("guni2dRT",              "guni2dRT.ups",              4,  "Linux", ["exactComparison"]),   \
                   ("SteadyBurn_2dRT",       "SteadyBurn_2dRT.ups",       4,  "Linux", ["exactComparison"]),   \
                   ("TBurner_2dRT",          "TBurner_2dRT.ups",          4,  "Linux", ["exactComparison"]),   \
                   ("TRWnoz",                "TRWnoz.ups",                1,  "Linux", ["exactComparison"]),   \
                   ("testConvertMPMICEAdd",  "testConvertMPMICEAdd.ups",  1,  "Linux", ["exactComparison"]),   \
                   ("advect_2L_MI",          "advect_2L_MI.ups",          1,  "Linux", ["exactComparison"]),   \
                   ("explode2D_amr",         "explode2D_amr",             8,  "Linux", ["startFromCheckpoint","exactComparison"]),   \
                   ("advect",                "advect.ups",                1,  "Darwin", ["doesTestRun"]),  \
                   ("massX",                 "massX.ups",                 1,  "Darwin", ["doesTestRun"]),  \
                   ("guni2dRT",              "guni2dRT.ups",              4,  "Darwin", ["doesTestRun"]),  \
                   ("SteadyBurn_2dRT",       "SteadyBurn_2dRT.ups",       4,  "Darwin", ["doesTestRun"]),  \
                   ("TBurner_2dRT",          "TBurner_2dRT.ups",          4,  "Darwin", ["doesTestRun"]),  \
                   ("TRWnoz",                "TRWnoz.ups",                1,  "Darwin", ["doesTestRun"]),  \
                   ("testConvertMPMICEAdd",  "testConvertMPMICEAdd.ups",  1,  "Darwin", ["doesTestRun"]),  \
                   ("advect_2L_MI",          "advect_2L_MI.ups",          1,  "Darwin", ["doesTestRun"]),  \
    	       ]
              
# Tests that are run during local regression testing
LOCALTESTS =  [    ("massX",                 "massX.ups",                 1,  "Linux", ["exactComparison"]),   \
                   ("guni2dRT",              "guni2dRT.ups",              4,  "Linux", ["exactComparison"])
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

  result = runSusTests(argv, TESTS, "MPMICE")
  exit( result )
