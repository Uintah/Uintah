#!/usr/bin/python

from sys import argv,exit
from os import environ
from helpers.runSusTests import runSusTests

#______________________________________________________________________
#  Test syntax: ( "folder name", "input file", # processors, "OS", ["flags1","flag2"])
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
#       startFromCheckpoint     - start test from checkpoint. (/usr/local/home/csafe-tester/CheckPoints/..../testname.uda.000)
#
#  Note: the "folder name" must be the same as input file without the extension.
#______________________________________________________________________

NIGHTLYTESTS = [   ("HePlume",       "HePlume.ups",     1.1, "Linux",  ["exactComparison"]), \
                   ("HePlume",       "HePlume.ups",     1.1, "Darwin", ["doesTestRun"]), \
                   ("JP8_Radiation", "JP8_Radiation.ups", 4, "Linux",  ["exactComparison"])
    	        ]
               
               
# Tests that are run during local regression testing               
LOCALTESTS   = [   ("HePlume",       "HePlume.ups",     1.1, "Linux",  ["exactComparison"]), \
                   ("HePlume",       "HePlume.ups",     1.1, "Darwin", ["doesTestRun"]), \
                   ("JP8_Radiation", "JP8_Radiation.ups", 4, "Linux",  ["exactComparison"])
    	        ]

#__________________________________

def getNightlyTests() :
  return TESTS

def getLocalTests() :
  return TESTS

#__________________________________

if __name__ == "__main__":

  if environ['LOCAL_OR_NIGHTLY_TEST'] == "local":
    TESTS = LOCALTESTS
  else:
    TESTS = NIGHTLYTESTS

  result = runSusTests(argv, TESTS, "Models")
  exit( result )
