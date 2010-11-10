#!/usr/bin/python
 
from os import symlink,environ
from sys import argv,exit,platform
from helpers.runSusTests import runSusTests, inputs_root
from helpers.modUPS import modUPS

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
#  Notes: 
#  The "folder name" must be the same as input file without the extension.
#  If the processors is > 1.0 then an mpirun command will be used
#______________________________________________________________________

UNUSED_TESTS = []

NIGHTLYTESTS = [
  ("BasicScalarTransportEquation",      "BasicScalarTransportEquation.ups",     1,      "Linux",	["exactComparison","no_restart","no_memoryTest","do_performance_test"] )
  ("TabPropsInterface",	      		"TabPropsInterface.ups",     		1,      "Linux",        ["exactComparison","no_restart","no_memoryTest","do_performance_test"] )
]


# Tests that are run during local regression testing
LOCALTESTS = [
  ("BasicScalarTransportEquation",      "BasicScalarTransportEquation.ups",     1,      "Linux",        ["exactComparison","no_restart","no_memoryTest"] )
  ("TabPropsInterface",	      		"TabPropsInterface.ups",     		1,      "Linux",        ["exactComparison","no_restart","no_memoryTest","do_performance_test"] )
]

#__________________________________

def getNightlyTests() :
  return NIGHTLYTESTS

def getLocalTests() :
  return LOCALTESTS

#__________________________________

if __name__ == "__main__":

  if environ['LOCAL_OR_NIGHTLY_TEST'] == "local":
    TESTS = LOCALTESTS
  else:
    TESTS = NIGHTLYTESTS

  result = runSusTests(argv, TESTS, "Wasatch")
  exit( result )

