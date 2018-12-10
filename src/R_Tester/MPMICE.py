#!/usr/bin/env python3

from sys import argv,exit
from os import environ
from helpers.runSusTests import runSusTests, ignorePerformanceTests

#______________________________________________________________________
#  Test syntax: ( "folder name", "input file", # processors, "OS", ["flags1","flag2",...])
#
#  OS:  Linux, Darwin, or ALL
#
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
#       postProcessRun          - start test from an existing uda in the checkpoints directory.  Compute new quantities and save them in a new uda
#       startFromCheckpoint     - start test from checkpoint. (/home/rt/CheckPoints/..../testname.uda.000)
#       sus_options="string"    - Additional command line options for sus command
#
#  Notes:
#  1) The "folder name" must be the same as input file without the extension.
#  2) Performance_tests are not run on a debug build.
#______________________________________________________________________

NIGHTLYTESTS = [
                  ("BurnRate",                "BurnRate.ups",                1, "ALL", ["startFromCheckpoint"]),
    	       ]

AMRTESTS   = [
                  ("advect_2L_MI",            "advect_2L_MI.ups",            8,  "ALL", ["exactComparison"]),
                  # these tests are non-deterministic
                  ("advect_+_amr",            "advect_+_amr.ups",            8,  "ALL", []),
                  ("advect_HollowSphere_amr", "advect_HollowSphere_amr.ups", 8,  "ALL", ["no_dbg"])
             ]

LOCALTESTS = [   ("massX",                    "massX.ups",                 1,  "ALL", ["exactComparison"]),
                 ("pistonVal",                "pistonValidation.ups",      2,  "ALL", ["exactComparison"]),
                 ("pistonVal_mks",            "pistonValidation.SI.Cu.ups",2,  "ALL", ["exactComparison"]),
                 ("guni2dRT",                 "guni2dRT.ups",              4,  "ALL", ["exactComparison"]),
                 ("SteadyBurn_2dRT",          "SteadyBurn_2dRT.ups",       4,  "ALL", ["exactComparison"]),
                 ("TBurner_2dRT",             "TBurner_2dRT.ups",          4,  "ALL", ["exactComparison"]),
                 ("TRWnoz",                   "TRWnoz.ups",                4,  "ALL", ["exactComparison"]),
                 ("JWLppCuRS2d",              "JWLppCuRS2d.ups",           10, "ALL", ["exactComparison"]),
                 ("DDT",                      "DDT.ups",                   1,  "ALL", ["exactComparison","no_dbg"]),
                 ("InductionTime",            "InductionTime.ups",         1  ,"ALL", ["exactComparison","no_dbg"]),
                 ("InductionPropagation",     "InductionPropagation.ups",  1  ,"ALL", ["exactComparison","no_dbg"]),
                 ("PBX_Cylinder_Ext_Load",    "PBX_array/oneCylinder.ups", 4  ,"ALL", ["exactComparison","no_restart","no_dbg"])
    	       ]
DEBUGTESTS =[]
#__________________________________
# The following list is parsed by the local RT script
# and allows the user to select the tests to run
#LIST: AMRTESTS DEBUGTESTS LOCALTESTS NIGHTLYTESTS BUILDBOTTESTS
#__________________________________


# returns the list
def getTestList(me) :
  if me == "AMRTESTS":
    TESTS = AMRTESTS
  elif me == "LOCALTESTS":
    TESTS = LOCALTESTS + AMRTESTS
  elif me == "DEBUGTESTS":
    TESTS = DEBUGTESTS
  elif me == "NIGHTLYTESTS":
    TESTS = LOCALTESTS + NIGHTLYTESTS + AMRTESTS
  elif me == "BUILDBOTTESTS":
    TESTS = ignorePerformanceTests( LOCALTESTS + NIGHTLYTESTS + AMRTESTS )
  else:
    print("\nERROR:MPMICE.py  getTestList:  The test list (%s) does not exist!\n\n" % me)
    exit(1)
  return TESTS
#__________________________________

if __name__ == "__main__":

  TESTS = getTestList( environ['WHICH_TESTS'] )

  result = runSusTests(argv, TESTS, "MPMICE")
  exit( result )
