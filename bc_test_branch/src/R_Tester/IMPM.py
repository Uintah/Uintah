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
#       do_performance_test:    - Run the performance test, log and plot simulation runtime.
#                                 (You cannot perform uda comparsions with this flag set)
#       doesTestRun:            - Checks if a test successfully runs
#       abs_tolerance=[double]  - absolute tolerance used in comparison
#       rel_tolerance=[double]  - relative tolerance used in comparison
#       exactComparison         - set absolute/relative tolerance = 0  for uda comparisons
#       startFromCheckpoint     - start test from checkpoint. (/home/csafe-tester/CheckPoints/..../testname.uda.000)
#       sus_options="string"    - Additional command line options for sus command
#
#  Notes: 
#  1) The "folder name" must be the same as input file without the extension.
#  2) If the processors is > 1.0 then an mpirun command will be used
#  3) Performance_tests are not run on a debug build.
#______________________________________________________________________

UNUSED = [ ("4disks2matsv", "4disks_2d.2matsv.ups", 4, "Linux"), \
    	]

NIGHTLYTESTS = [  ("4disks_2d.1mat",   "4disks_2d.1mat.ups", 1,   "None", ["exactComparison"]), \
	           ("billet.static",    "billet.static.ups",  2,   "ALL", ["exactComparison"]), \
	           ("adiCuJC01s296K",   "adiCuJC01s296K.ups", 1.1, "ALL", ["exactComparison"]), \
	           ("adiCuMTS01s296K",  "adiCuMTS01s296K.ups",1.1, "ALL", ["exactComparison"]), \
	           ("adiCuPTW01s296K",  "adiCuPTW01s296K.ups",1.1, "ALL", ["exactComparison"]), \
	           ("adiCuSCG01s296K",  "adiCuSCG01s296K.ups",1.1, "ALL", ["exactComparison"]), \
	           ("adiCuZA01s296K",   "adiCuZA01s296K.ups", 1.1, "ALL", ["exactComparison"])
    	         ]
                
# Tests that are run during local regression testing       
LOCALTESTS = [    ("4disks_2d.1mat",   "4disks_2d.1mat.ups", 1,   "None"), \
	           ("billet.static",    "billet.static.ups",  2,   "ALL"), \
	           ("adiCuJC01s296K",   "adiCuJC01s296K.ups", 1.1, "ALL"), \
	           ("adiCuMTS01s296K",  "adiCuMTS01s296K.ups",1.1, "ALL"), \
	           ("adiCuPTW01s296K",  "adiCuPTW01s296K.ups",1.1, "ALL"), \
	           ("adiCuSCG01s296K",  "adiCuSCG01s296K.ups",1.1, "ALL"), \
	           ("adiCuZA01s296K",   "adiCuZA01s296K.ups", 1.1, "ALL")
    	       ]       

#__________________________________

def getNightlyTests() :
  return NIGHTLYTESTS

def getLocalTests() :
  return LOCALTESTS

if __name__ == "__main__":

  if environ['WHICH_TESTS'] == "local":
    TESTS = LOCALTESTS
  else:
    TESTS = NIGHTLYTESTS

  result = runSusTests(argv, TESTS, "IMPM")
  exit( result )

