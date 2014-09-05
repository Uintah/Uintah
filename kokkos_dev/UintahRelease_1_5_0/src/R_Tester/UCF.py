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

UNUSED = [ ("dlb_mpm",        "mpmDLBTest.ups",     8, "NONE"), \
           ("dlb_mpmice",     "mpmiceDLBTest.ups",  8, "NONE") 
]

NIGHTLYTESTS = [ ("ice_perf_test",          "icePerformanceTest.ups",             1, "Linux", ["do_performance_test"]),  \
                 ("mpmice_perf_test",       "mpmicePerformanceTest.ups",          1, "Linux", ["do_performance_test"]), \
                 ("LBwoRegrid",             "LBwoRegrid.ups",                     2, "Linux", ["exactComparison"]), \
                 ("switchExample_impm_mpm", "Switcher/switchExample_impm_mpm.ups",1, "Linux", ["no_memoryTest"]), \
                 ("switchExample3",         "Switcher/switchExample3.ups",        1, "Linux", ["no_restart","no_memoryTest"]), \
               ]

LOCALTESTS = [ ("switchExample_impm_mpm", "Switcher/switchExample_impm_mpm.ups",1, "Linux", ["no_memoryTest"]), \
               ("switchExample3",         "Switcher/switchExample3.ups",        1, "Linux", ["no_restart","no_memoryTest"]), \
               ("ice_perf_test",          "icePerformanceTest.ups",             1, "Linux", ["do_performance_test"]),  \
               ("mpmice_perf_test",       "mpmicePerformanceTest.ups",          1, "Linux", ["do_performance_test"]), \
               ("LBwoRegrid",             "LBwoRegrid.ups",                     2, "Linux", ["exactComparison"])
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

  result = runSusTests(argv, TESTS, "UCF")
  exit( result )
  
