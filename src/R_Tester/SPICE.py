#!/usr/bin/env python3

from sys import argv, exit
from os import environ, system
from helpers.runSusTests_git import runSusTests, ignorePerformanceTests, getInputsDir
from helpers.modUPS import modUPS


the_dir = "%s/%s" % ( getInputsDir(),"SPICE" )

#______________________________________________________________________
#  Variants of riemann_gc_x.  Turning off a Kurganov-Tadmor option makes SPICE
#  schedule ICE's version of that task instead of its own copy, so each of the
#  first three tests one of those switches.
riemann_gc_x_noFaceVel_ups   = modUPS( the_dir, "riemann_gc_x.ups",
                                       ["<faceVelocity>   false </faceVelocity>",
                                        "<filebase>riemann_gc_x_noFaceVel.uda</filebase>"] )

riemann_gc_x_noFacePress_ups = modUPS( the_dir, "riemann_gc_x.ups",
                                       ["<facePressure>   false </facePressure>",
                                        "<filebase>riemann_gc_x_noFacePress.uda</filebase>"] )

riemann_gc_x_noKTpress_ups   = modUPS( the_dir, "riemann_gc_x.ups",
                                       ["<ktPressureFlux> false </ktPressureFlux>",
                                        "<filebase>riemann_gc_x_noKTpress.uda</filebase>"] )

riemann_gc_x_Tadmor_ups      = modUPS( the_dir, "riemann_gc_x.ups",
                                       ['<advection type = "CentralUpwind" limiter = "minmod" />',
                                        "<fluxScheme>     Tadmor </fluxScheme>",
                                        "<kappaFunction>  one    </kappaFunction>",
                                        "<filebase>riemann_gc_x_Tadmor.uda</filebase>"] )

#  PIMPLE with 3 passes (the only test that carries one outer iterate into the
#  next) and an outerTolerance that ends about a third of the timesteps after
#  the 2nd pass (the only test of the early exit).  theta = 1 is not used: it
#  diverges on this shock tube in the first timestep, in SPICE and in the
#  original in-ICE PIMPLE code alike.
riemann_gc_x_PIMPLE_ups      = modUPS( the_dir, "riemann_gc_x.ups",
                                       ["<nOuterCorrectors> 3    </nOuterCorrectors>",
                                        "<theta>            0.5  </theta>",
                                        "<outerTolerance>   1e-2 </outerTolerance>",
                                        "<filebase>riemann_gc_x_PIMPLE.uda</filebase>"] )

#______________________________________________________________________
#  Test syntax: ( "folder name", "input file", # processors, "OS", ["flags1","flag2",...])
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
#       mpirun_options="string"     - Additional command line options for mpirun
#       compareUda_options="string" - Additional command line options for compare_uda
#       preProcessCmd="string"      - command run prior to running sus.  The command path must be defined with ADDTL_PATH
#                                     The command's final argument is the ups filename
#
#  Notes:
#  1) The "folder name" must be the same as input file without the extension.
#  2) Performance_tests are not run on a debug build.
#______________________________________________________________________

#  gasCombustion chemistry: 0D constant-volume ignition (stationary, so no KT or PIMPLE)
CHEMISTRY  = [   ("cvHydrogen1",              "RX_MODELS/cvHydrogen1.ups",               1,  "ALL", ["exactComparison"]),
                 ("cvHydrogen2",              "RX_MODELS/cvHydrogen2.ups",               1,  "ALL", ["exactComparison"])
             ]

#  chemistry + species and heat diffusion + LODI: 1D laminar flames
DIFFUSION  = [   ("hydrogenFlame1_KT_PIMPLE", "RX_MODELS/hydrogenFlame1_KT_PIMPLE.ups", 12,  "ALL", ["exactComparison", "no_dbg"]),
                 ("hydrogenFlame2",           "RX_MODELS/hydrogenFlame2.ups",           10,  "ALL", ["exactComparison", "no_dbg"])
             ]

#  Kurganov-Tadmor advection with every option on, along each axis, plus option variants
KT         = [   ("riemann_gc_x",             "riemann_gc_x.ups",              4,  "ALL", ["exactComparison"]),
                 ("riemann_gc_y",             "riemann_gc_y.ups",              4,  "ALL", ["exactComparison"]),
                 ("riemann_gc_z",             "riemann_gc_z.ups",              4,  "ALL", ["exactComparison"]),
                 ("riemann_gc_x_noFaceVel",   riemann_gc_x_noFaceVel_ups,      4,  "ALL", ["exactComparison"]),
                 ("riemann_gc_x_noFacePress", riemann_gc_x_noFacePress_ups,    4,  "ALL", ["exactComparison"]),
                 ("riemann_gc_x_noKTpress",   riemann_gc_x_noKTpress_ups,      4,  "ALL", ["exactComparison"]),
                 ("riemann_gc_x_Tadmor",      riemann_gc_x_Tadmor_ups,         4,  "ALL", ["exactComparison"])
             ]

#  Kurganov-Tadmor advection + the PIMPLE outer loop
KT_PIMPLE  = [   ("argonDetonation",          "RX_MODELS/argonDetonation.ups",           4,  "ALL", ["exactComparison"]),
                 ("riemann_gc_x_PIMPLE",      riemann_gc_x_PIMPLE_ups,         4,  "ALL", ["exactComparison"])
             ]

#  no KT or PIMPLE: ICE's tasks with SPICE's pressure work
ICE_ONLY   = [   ("rayleigh_dx",              "rayleigh_dx.ups",               1,  "ALL", ["exactComparison"]),
                 ("rayleigh_dy",              "rayleigh_dy.ups",               1,  "ALL", ["exactComparison"]),
                 ("rayleigh_dz",              "rayleigh_dz.ups",               1,  "ALL", ["exactComparison"]),
                 ("impHotBlob",               "impHotBlob.ups",                1,  "ALL", ["exactComparison"]),
                 ("mms_1",                    "mms_1.ups",                     1,  "ALL", ["exactComparison", "no_restart"])
             ]

#  multiple dimensions
MULTI_D    = [   ("advect_periodic_KT",       "advect_periodic_KT.ups",        8,  "ALL", ["exactComparison"]),
                 ("riemann3D_KT_PIMPLE",      "riemann3D_KT_PIMPLE.ups",       8,  "ALL", ["exactComparison"])
             ]

DEBUGTESTS   = []

NIGHTLYTESTS = CHEMISTRY + DIFFUSION + KT + KT_PIMPLE + ICE_ONLY + MULTI_D

#__________________________________
ADDTL_PATH=[]           # preprocessing cmd path.  It can be an absolute or relative path from the StandAlone directory
                        # syntax:  (relativePath=<path> or absolutePath=<path>)

#__________________________________
# The following list is parsed by the local RT script
# and allows the user to select the tests to run
#LIST: BUILDBOTTESTS CHEMISTRY DEBUGTESTS DIFFUSION ICE_ONLY KT KT_PIMPLE LOCALTESTS MULTI_D NIGHTLYTESTS
#__________________________________


# returns the list
def getTestList(me) :
  if me == "CHEMISTRY":
    TESTS = CHEMISTRY
  elif me == "DIFFUSION":
    TESTS = DIFFUSION
  elif me == "KT":
    TESTS = KT
  elif me == "KT_PIMPLE":
    TESTS = KT_PIMPLE
  elif me == "ICE_ONLY":
    TESTS = ICE_ONLY
  elif me == "MULTI_D":
    TESTS = MULTI_D
  elif me == "DEBUGTESTS":
    TESTS = DEBUGTESTS
  elif me == "LOCALTESTS":
    TESTS = NIGHTLYTESTS
  elif me == "NIGHTLYTESTS":
    TESTS = NIGHTLYTESTS
  elif me == "BUILDBOTTESTS":
    TESTS = ignorePerformanceTests( NIGHTLYTESTS )
  else:
    print("\nERROR:SPICE.py  getTestList:  The test list (%s) does not exist!\n\n" % me)
    exit(1)
  return TESTS
#__________________________________

if __name__ == "__main__":

  TESTS = getTestList( environ['WHICH_TESTS'] )

  result = runSusTests(argv, TESTS, "SPICE", ADDTL_PATH)

  # cleanup modified input files
  command = "/bin/rm -rf %s/tmp  " % (the_dir)
  system( command )

  exit( result )
