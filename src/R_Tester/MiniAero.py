#!/usr/bin/env python

from sys import argv, exit
from os import environ
from helpers.runSusTests import runSusTests, inputs_root, generatingGoldStandards
from helpers.modUPS import modUPS

the_dir = generatingGoldStandards()

if the_dir == "" :
  the_dir = "%s/MiniAero" % inputs_root()
else :
  the_dir = the_dir + "/MiniAero"
  
#__________________________________
#  modifications to the input file
riemann3D_ups = modUPS( the_dir, "riemann3D.ups" ,  \
                           ["<maxTime>            0.0025     </maxTime>",          \
                            "<resolution>       [80,80,80]   </resolution>",       \
                            "<outputInterval>     0.00125    </outputInterval>",   \
                            '<checkpoint interval="0.001"  cycle="2"/>'])


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

LOCALTESTS = [   ("advect",           "advect.ups",           1, "ALL", ["no_dbg","exactComparison"]),    \
                 ("sod_miniaero_ref", "sod_miniaero_ref.ups", 1, "ALL", ["no_dbg","exactComparison"]),    \
                 ("riemann3D",        riemann3D_ups,          2, "ALL", ["no_dbg","exactComparison"]),                 
                 ("riemann3D_thread", riemann3D_ups,          1.1, "ALL", ["no_dbg","exactComparison", "sus_options=-nthreads 8"])
              ]
#__________________________________


#__________________________________
# The following line is parsed by the local RT script
# and allows the user to select the different subsets
#LIST: LOCALTESTS
#__________________________________
# returns the list  
def getTestList(me) :
  if me == "LOCALTESTS":
    TESTS = LOCALTESTS
  else:
    print "\nERROR:MiniAero.py  getTestList:  The test list (%s) does not exist!\n\n" % me
    exit(1)
  return TESTS

#__________________________________
if __name__ == "__main__":

  TESTS = getTestList( environ['WHICH_TESTS'] )

  result = runSusTests(argv, TESTS, "MiniAero")
  exit( result )

