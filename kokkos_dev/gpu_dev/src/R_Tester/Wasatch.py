#!/usr/bin/env python
 
from sys import argv, exit
from os import environ
from helpers.runSusTests import runSusTests, inputs_root, generatingGoldStandards
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
#       startFromCheckpoint     - start test from checkpoint. (/home/csafe-tester/CheckPoints/..../testname.uda.000)
#
#  Notes: 
#  1) The "folder name" must be the same as input file without the extension.
#  2) If the processors is > 1.0 then an mpirun command will be used
#  3) Performance_tests are not run on a debug build.
#______________________________________________________________________

UNUSED_TESTS = []

NIGHTLYTESTS = [
  ("BasicScalarTransportEquation",  "BasicScalarTransportEquation.ups",1,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("BasicScalarTransportEq_2L",     "BasicScalarTransportEq_2L.ups",   1,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ), \
  ("TabPropsInterface",             "TabPropsInterface.ups",           1,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ), \
  ("convection-test2",              "convection-test2.ups",            2,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("convection-test",               "convection-test.ups",             3,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("convection-test3",              "convection-test3.ups",            1,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("convection-superbee",           "convection-test-superbee.ups",    1,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("bc-test-mixed",                 "bc-test-mixed.ups",               4,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("ScalarTransportEquation",       "ScalarTransportEquation.ups",     1,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("momentum-test-mms-xy",          "momentum-test-mms-xy.ups",        4,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("momentum-test-mms-xz",          "momentum-test-mms-xz.ups",        4,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("momentum-test-mms-yz",          "momentum-test-mms-yz.ups",        4,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("scalability-test",              "scalability-test.ups",            1,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("momentum-test-mms-3D",          "momentum-test-mms-3D.ups",        8,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("bc-test-svol-xdir",             "bc-test-svol-xdir.ups",           4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("bc-test-svol-ydir",             "bc-test-svol-ydir.ups",           4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("bc-test-svol-zdir",             "bc-test-svol-zdir.ups",           4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("bc-test-xvol-xdir",             "bc-test-xvol-xdir.ups",           4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("bc-test-xvol-ydir",             "bc-test-xvol-ydir.ups",           4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("bc-test-xvol-zdir",             "bc-test-xvol-zdir.ups",           4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("bc-test-yvol-xdir",             "bc-test-yvol-xdir.ups",           4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("bc-test-yvol-ydir",             "bc-test-yvol-ydir.ups",           4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("bc-test-yvol-zdir",             "bc-test-yvol-zdir.ups",           4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("bc-test-zvol-xdir",             "bc-test-zvol-xdir.ups",           4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("bc-test-zvol-ydir",             "bc-test-zvol-ydir.ups",           4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("bc-test-zvol-zdir",             "bc-test-zvol-zdir.ups",           4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("qmom-test",                     "qmom-test.ups",                   4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("convection-test-svol-xdir",     "convection-test-svol-xdir.ups",   4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("convection-test-svol-ydir",     "convection-test-svol-ydir.ups",   4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("convection-test-svol-zdir",     "convection-test-svol-zdir.ups",   4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("force-on-graph-postprocessing-test",     "force-on-graph-postprocessing-test.ups",   4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] )
]


# Tests that are run during local regression testing
LOCALTESTS = [
  ("BasicScalarTransportEquation", "BasicScalarTransportEquation.ups", 1, "All", ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("BasicScalarTransportEq_2L",     "BasicScalarTransportEq_2L.ups",   1, "All", ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("TabPropsInterface",             "TabPropsInterface.ups",           1,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("convection-test2",              "convection-test2.ups",            2,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("convection-test",               "convection-test.ups",             3,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("convection-test3",              "convection-test3.ups",            1,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("convection-superbee",           "convection-test-superbee.ups",    1,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("bc-test-mixed",                 "bc-test-mixed.ups",               4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("ScalarTransportEquation",       "ScalarTransportEquation.ups",     1,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("momentum-test-mms-xy",          "momentum-test-mms-xy.ups",        4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("momentum-test-mms-xz",          "momentum-test-mms-xz.ups",        4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("momentum-test-mms-yz",          "momentum-test-mms-yz.ups",        4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("scalability-test",              "scalability-test.ups",            1,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("momentum-test-mms-3D",          "momentum-test-mms-3D.ups",        8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("bc-test-svol-xdir",             "bc-test-svol-xdir.ups",           4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("bc-test-svol-ydir",             "bc-test-svol-ydir.ups",           4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("bc-test-svol-zdir",             "bc-test-svol-zdir.ups",           4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("bc-test-xvol-xdir",             "bc-test-xvol-xdir.ups",           4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("bc-test-xvol-ydir",             "bc-test-xvol-ydir.ups",           4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("bc-test-xvol-zdir",             "bc-test-xvol-zdir.ups",           4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("bc-test-yvol-xdir",             "bc-test-yvol-xdir.ups",           4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("bc-test-yvol-ydir",             "bc-test-yvol-ydir.ups",           4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("bc-test-yvol-zdir",             "bc-test-yvol-zdir.ups",           4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("bc-test-zvol-xdir",             "bc-test-zvol-xdir.ups",           4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("bc-test-zvol-ydir",             "bc-test-zvol-ydir.ups",           4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("bc-test-zvol-zdir",             "bc-test-zvol-zdir.ups",           4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("qmom-test",                     "qmom-test.ups",                   4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("convection-test-svol-xdir",     "convection-test-svol-xdir.ups",   4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("convection-test-svol-ydir",     "convection-test-svol-ydir.ups",   4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("convection-test-svol-zdir",     "convection-test-svol-zdir.ups",   4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("force-on-graph-postprocessing-test",     "force-on-graph-postprocessing-test.ups",   4,  "All",  ["exactComparison","no_restart","no_memoryTest"] )        
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

  result = runSusTests(argv, TESTS, "Wasatch")
  exit( result )

