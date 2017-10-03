#!/usr/bin/env python

from sys import argv, exit
from os import environ, system
from helpers.runSusTests import runSusTests, inputs_root, generatingGoldStandards
from helpers.modUPS import modUPS

the_dir = generatingGoldStandards()

if the_dir == "" :
  the_dir = "%s/ICE" % inputs_root()
else :
  the_dir = the_dir + "/ICE"

riemann_1L_ups    = modUPS( the_dir,                       \
                             "riemann_sm.ups" ,            \
                             ["<maxTime>            0.0001      </maxTime>", \
                              "<outputInterval> 0.000025 </outputInterval>"])


riemann_AMR_3L_ups = modUPS( the_dir,                       \
                             "riemann_AMR.ups" ,            \
                             ["<maxTime>            0.0001      </maxTime>", \
                              "<outputInterval> 0.000025 </outputInterval>"])

hotBlob_AMR_3L_ups = modUPS( the_dir,                       \
                             "hotBlob_AMR.ups",             \
                             ["<max_levels>3</max_levels>", \
                              "<filebase>AMR_HotBlob_3L.uda</filebase>"])

#______________________________________________________________________
#  Test syntax: ( "folder name", "input file", # processors, "OS",["flags1","flag2"])
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
#       startFromCheckpoint     - start test from checkpoint. (/home/csafe-tester/CheckPoints/..../testname.uda.000)
#       sus_options="string"    - Additional command line options for sus command
#
#  Notes:
#  1) The "folder name" must be the same as input file without the extension.
#  2) Performance_tests are not run on a debug build.
#______________________________________________________________________

NIGHTLYTESTS = [   ("advect",             "advect.ups",              1, "All", ["exactComparison"]),
                   ("riemann_1L",         riemann_1L_ups,            1, "All", ["exactComparison"]),
                   ("CouettePoiseuille",  "CouettePoiseuille.ups",   1, "All", ["exactComparison"]),
                   ("hotBlob2mat",        "hotBlob2mat.ups",         1, "All", ["exactComparison"]),
                   ("hotBlob2mat_sym",    "hotBlob2mat_sym.ups",     1, "All", ["exactComparison"]),
                   ("impHotBlob",         "impHotBlob.ups",          1, "All", ["exactComparison"]),
                   ("hotBlob2mat8patch",  "hotBlob2mat8patch.ups",   8, "All", ["exactComparison"]),
                   ("waterAirOscillator", "waterAirOscillator.ups",  4, "All", ["exactComparison"])
              ]

AMRTESTS =    [
                  ("riemann_AMR_3L",      riemann_AMR_3L_ups,       8, "All", ["exactComparison"]),
                  ("advect2matAMR",      "advect2matAMR.ups",       1, "All", ["exactComparison"]),
                  ("hotBlob_AMR",        "hotBlob_AMR.ups",         4, "All", ["exactComparison"]),
                  ("hotBlob_AMR_3L",      hotBlob_AMR_3L_ups,       4, "All", ["exactComparison"]),
                  ("impAdvectAMR",       "impAdvectAMR.ups",        1, "All", ["exactComparison"]),
              ]

DEBUGGING =   [   ("advect",           "advect.ups",           1, "All", ["exactComparison"]),
                  ("riemann_sm",       "riemann_sm.ups",       1, "All", ["exactComparison"])
              ]
#__________________________________


#__________________________________
# The following line is parsed by the local RT script
# and allows the user to select the different subsets
#LIST:  AMRTESTS DEBUGGING LOCALTESTS NIGHTLYTESTS
#__________________________________
# returns the list
def getTestList(me) :
  if me == "AMRTESTS":
    TESTS = AMRTESTS
  elif me == "DEBUGGING":
    TESTS = DEBUGGING
  elif me == "LOCALTESTS":
    TESTS = NIGHTLYTESTS + AMRTESTS
  elif me == "NIGHTLYTESTS":
    TESTS = NIGHTLYTESTS + AMRTESTS
  else:
    print "\nERROR:ICE.py  getTestList:  The test list (%s) does not exist!\n\n" % me
    exit(1)
  return TESTS

#__________________________________
if __name__ == "__main__":

  TESTS = getTestList( environ['WHICH_TESTS'] )

  result = runSusTests(argv, TESTS, "ICE")

  # cleanup modified files
  command = "/bin/rm -rf %s/tmp > /dev/null 2>&1 " % (the_dir)
  system( command )

  exit( result )

