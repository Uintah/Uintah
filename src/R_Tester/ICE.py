#!/usr/bin/env python3

from sys import argv, exit
from os import environ, system
from helpers.runSusTests_git import runSusTests, ignorePerformanceTests, getInputsDir
from helpers.modUPS import modUPS, modUPS2


the_dir = "%s/%s" % ( getInputsDir(),"ICE" )

riemann_1L_ups    = modUPS( the_dir,                       \
                             "riemann_sm.ups" ,            \
                             ["<maxTime>            0.0001      </maxTime>", \
                              "<outputInterval> 0.000025 </outputInterval>"])


advectAMR_perf_ups = modUPS2( the_dir, \
                               "advectAMR.ups", \
                               [("delete", "/Uintah_specification/DataAnalysis")] )

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
#       gpu:                        - run test if machine is gpu enabled
#       no_uda_comparison:          - skip the uda comparisons
#       no_memoryTest:              - skip all memory checks
#       no_restart:                 - skip the restart tests
#       no_dbg:                     - skip all debug compilation tests
#       no_opt:                     - skip all optimized compilation tests
#       no_cuda:                    - skip test if this is a cuda enable build
#       do_performance_test:        - Run the performance test, log and plot simulation runtime.
#                                     (You cannot perform uda comparsions with this flag set)
#       doesTestRun:                - Checks if a test successfully runs
#       abs_tolerance=[double]      - absolute tolerance used in comparisons
#       rel_tolerance=[double]      - relative tolerance used in comparisons
#       exactComparison             - set absolute/relative tolerance = 0  for uda comparisons
#       postProcessRun              - start test from an existing uda in the checkpoints directory.  Compute new quantities and save them in a new uda
#       startFromCheckpoint         - start test from checkpoint. (/home/rt/CheckPoints/..../testname.uda.000)
#       sus_options="string"        - Additional command line options for sus command
#       compareUda_options="string" - Additional command line options for compare_uda
#
#  Notes:
#  1) The "folder name" must be the same as input file without the extension.
#  2) Performance_tests are not run on a debug build.
#______________________________________________________________________

NIGHTLYTESTS = [   ("advect",             "advect.ups",              1, "All", ["exactComparison"]),
                   ("advectPeriodic",     "advect_periodic.ups",     8, "All", ["exactComparison"]),
                   ("riemann_1L",         riemann_1L_ups,            1, "All", ["exactComparison"]),
                   ("hotBlob2mat",        "hotBlob2mat.ups",         1, "All", ["exactComparison"]),
                   ("hotBlob2mat_sym",    "hotBlob2mat_sym.ups",     1, "All", ["exactComparison"]),
                   ("impAdvect",          "impAdvect.ups",           8, "All", ["exactComparison"]),
                   ("impAdvectPeriodic",  "impAdvect_periodic.ups",  8, "All", ["exactComparison"]),
                   ("impHotBlob",         "impHotBlob.ups",          1, "All", ["exactComparison"]),
                   ("hotBlob2mat8patch",  "hotBlob2mat8patch.ups",   8, "All", ["exactComparison"]),
                   ("waterAirOscillator", "waterAirOscillator.ups",  4, "All", ["exactComparison"]),
                   ("constantPress_BC",   "constantPress_BC.ups",    8, "All", ["exactComparison", "no_restart"])  # dat file comparsion not working on restart
              ]

DIFFUSION  = [     ("Poiseuille_XY",      "CouettePoiseuille/XY.ups", 1, "All", ["exactComparison"]),
                   ("Poiseuille_ZX",      "CouettePoiseuille/ZX.ups", 1, "All", ["exactComparison"]),
                   ("Poiseuille_YZ",      "CouettePoiseuille/YZ.ups", 1, "All", ["exactComparison"]),
                   ("rayleigh_dx",        "rayleigh_dx.ups",          1, "All", ["exactComparison"]),
                   ("rayleigh_dy",        "rayleigh_dy.ups",          1, "All", ["exactComparison"]),
                   ("rayleigh_dz",        "rayleigh_dz.ups",          1, "All", ["exactComparison"])
              ]

LODI        = [    ("Lodi_pulse",        "Lodi_pulse.ups",         8, "All", ["exactComparison"])
              ]


AMRTESTS =    [   ("advectAMR",          "advectAMR.ups",           8, "All", ["exactComparison"]),
                  ("advectAMR_perf",     advectAMR_perf_ups,        8, "All", ["do_performance_test"]),
                  ("riemann_AMR_3L",      riemann_AMR_3L_ups,       8, "All", ["exactComparison"]),
                  ("advect2matAMR",      "advect2matAMR.ups",       1, "All", ["exactComparison"]),
                  ("hotBlob_AMR",        "hotBlob_AMR.ups",         4, "All", ["exactComparison"]),
                  ("hotBlob_AMR_3L",      hotBlob_AMR_3L_ups,       4, "All", ["exactComparison"]),
                  ("impAdvect_ML_5L",    "impAdvect_ML_5L.ups",     8, "All", ["exactComparison"])
              ]

DEBUGGING =   [   ("advect",           "advect.ups",           1, "All", ["exactComparison"]),
                  ("riemann_sm",       "riemann_sm.ups",       1, "All", ["exactComparison"])
              ]
#__________________________________


#__________________________________
# The following line is parsed by the local RT script
# and allows the user to select the different subsets
#LIST:  AMRTESTS DIFFUSION DEBUGGING LOCALTESTS LODI NIGHTLYTESTS BUILDBOTTESTS
#__________________________________
# returns the list

NIGHTLYTESTS = NIGHTLYTESTS + AMRTESTS + DIFFUSION + LODI

def getTestList(me) :
  if me == "AMRTESTS":
    TESTS = AMRTESTS
  elif me == "DEBUGGING":
    TESTS = DEBUGGING
  elif me == "DIFFUSION":
    TESTS = DIFFUSION
  elif me == "LOCALTESTS":
    TESTS = NIGHTLYTESTS
  elif me == "LODI":
    TESTS = LODI
  elif me == "NIGHTLYTESTS":
    TESTS = NIGHTLYTESTS
  elif me == "BUILDBOTTESTS":
    TESTS = ignorePerformanceTests( NIGHTLYTESTS )
  else:
    print("\nERROR:ICE.py  getTestList:  The test list (%s) does not exist!\n\n" % me)
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

