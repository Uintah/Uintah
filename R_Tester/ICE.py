#!/usr/bin/env python

from sys import argv, exit
from os import environ
from helpers.runSusTests import runSusTests, inputs_root, generatingGoldStandards
from helpers.modUPS import modUPS

the_dir = generatingGoldStandards()

if the_dir == "" :
  the_dir = "%s/ICE" % inputs_root()
else :
  the_dir = the_dir + "/ICE"


hotBlob_AMR_3L_ups = modUPS( the_dir,                       \
                             "hotBlob_AMR.ups",             \
                             ["<max_levels>3</max_levels>", \
                              "<lattice_refinement_ratio> [[5,5,1],[2,2,1]]  </lattice_refinement_ratio>", \
                              "<filebase>AMR_HotBlob_3L.uda</filebase>"])

#______________________________________________________________________
#  Test syntax: ( "folder name", "input file", # processors, "OS",["flags1","flag2"])
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

NIGHTLYTESTS = [   ("advect",           "advect.ups",            1, "Linux", ["exactComparison"]),      \
                   ("riemann_sm",       "riemann_sm.ups",        1, "Linux", ["exactComparison"]),      \
                   ("hotBlob2mat",      "hotBlob2mat.ups",       1, "Linux", ["exactComparison"]),      \
                   ("hotBlob2mat_sym",  "hotBlob2mat_sym.ups",   1, "Linux", ["exactComparison"]),      \
                   ("impHotBlob",       "impHotBlob.ups",        1, "Linux", ["exactComparison"]),      \
                   ("hotBlob2mat8patch","hotBlob2mat8patch.ups", 8, "Linux",["exactComparison"]),       \
                   ("advect2matAMR",    "advect2matAMR.ups",     1, "Linux", ["exactComparison"]),      \
                   ("hotBlob_AMR",      "hotBlob_AMR.ups",       4, "Linux", ["exactComparison"]),      \
                   ("hotBlob_AMR_3L",    hotBlob_AMR_3L_ups,     4, "Linux", ["exactComparison"]),      \
                   ("impAdvectAMR",     "impAdvectAMR.ups",    1.1, "Linux", ["exactComparison"])
              ]


# Tests that are run during local regression testing
LOCALTESTS = NIGHTLYTESTS

#LOCALTESTS = [   ("advect",           "advect.ups",           1, "ALL", ["exactComparison"]),    \
#                 ("riemann_sm",       "riemann_sm.ups",       1, "All", ["exactComparison"])       
#              ]
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

  result = runSusTests(argv, TESTS, "ICE")
  exit( result )

