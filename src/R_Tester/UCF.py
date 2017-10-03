#!/usr/bin/env python

from os import chdir,getcwd,mkdir,system,environ
from sys import argv,exit,platform
from helpers.runSusTests import runSusTests, inputs_root
from helpers.modUPS import modUPS

#______________________________________________________________________
#Performance and other UCF related tests

#  Test syntax: ( "folder name", "input file", # processors, "OS", ["flags1","flag2"])
#  flags: 
#       gpu:                    - run test if machine is gpu enabled
#       no_uda_comparison:      - skip the uda comparisons
#       no_memoryTest:          - skip all memory checks
#       no_restart:             - skip the restart tests
#       no_dbg:                 - skip all debug compilation tests
#       no_opt:                 - skip all optimized compilation tests
#       no_cuda:                - skip test if this is a cuda enable build
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
#  2) Performance_tests are not run on a debug build.
#______________________________________________________________________


NIGHTLYTESTS = [ ("ice_perf_32KPatches",  "icePerf_32KPatches.ups",            10, "All", ["do_performance_test"]),
               ]

LOCALTESTS = [ ("switchExample_impm_mpm", "Switcher/switchExample_impm_mpm.ups",1, "All", ["no_memoryTest"]), \
               ("switchExample3",         "Switcher/switchExample3.ups",        1, "All", ["no_restart","no_memoryTest"]), \
               ("ice_perf_test",          "icePerformanceTest.ups",             1, "All", ["do_performance_test"]),  \
               ("mpmice_perf_test",       "mpmicePerformanceTest.ups",          1, "All", ["do_performance_test"]), \
               ("LBwoRegrid",             "LBwoRegrid.ups",                     2, "All", ["exactComparison"])
             ]
DEBUGTESTS =[]
#__________________________________
# The following list is parsed by the local RT script
# and allows the user to select the tests to run
#LIST: LOCALTESTS DEBUGTESTS NIGHTLYTESTS
#___________________________________

# returns the list  
def getTestList(me) :
  if me == "LOCALTESTS":
    TESTS = LOCALTESTS
  elif me == "DEBUGTESTS":
    TESTS = DEBUGTESTS
  elif me == "NIGHTLYTESTS":
    TESTS = LOCALTESTS + NIGHTLYTESTS
  else:
    print "\nERROR:UCF.py  getTestList:  The test list (%s) does not exist!\n\n" % me
    exit(1)
  return TESTS
#__________________________________

if __name__ == "__main__":

  TESTS = getTestList( environ['WHICH_TESTS'] )

  result = runSusTests(argv, TESTS, "UCF")
  exit( result )
  
