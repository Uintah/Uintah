#!/usr/bin/env python

from sys import argv,exit
from os import environ
from helpers.runSusTests import runSusTests

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
NIGHTLYTESTS = [   ("poissonGPU1",       "poissonGPU1.ups",         1, "Linux", ["gpu"]),
                   ("gpuSchedulerTest",  "gpuSchedulerTest.ups",  1.1, "Linux", ["gpu", "no_restart", "no_uda_comparison", "sus_options=-nthreads 4 -gpu"]) ]

# Tests that are run during local regression testing
LOCALTESTS   = [   ("poissonGPU1",       "poissonGPU1.ups",         1, "Linux", ["gpu"]),
                   ("gpuSchedulerTest",  "gpuSchedulerTest.ups",  1.1, "Linux", ["gpu", "no_restart", "no_uda_comparison", "sus_options=-nthreads 4 -gpu"]) ]



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
  result = runSusTests(argv, TESTS, "GPU")
  exit( result )
