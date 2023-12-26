#!/usr/bin/env python3

from sys import argv, exit
from os import environ, system
from helpers.runSusTests_git import runSusTests, ignorePerformanceTests, getInputsDir
from helpers.modUPS import modUPS, modUPS2
#______________________________________________________________________
#  Modify the input files

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

# use xmlstarlet el -v <ups> for xml paths
chanFlow_powerLaw_ups = modUPS2( the_dir,                       \
                                  "channelFlow_PowerLaw.ups",   \
                               [( "update", "/Uintah_specification/DataArchiver/filebase :powerLaw.uda" ),
                                ( "update", "/Uintah_specification/Grid/BoundaryConditions/include[@href='inputs/ICE/channelFlow.xml' and @section='inletVelocity']/@type :powerLawProfile" ),
                                ( "update", "/Uintah_specification/Grid/BoundaryConditions/Face[@side='x-']/BCType[@id='0' and @label='Velocity']/@var :powerLawProfile" ),
                                ( "update", "/Uintah_specification/CFD/ICE/customInitialization/include[@href='inputs/ICE/channelFlow.xml']/@section :powerLawProfile")
                               ] )

chanFlow_powerLaw2_ups = modUPS2( the_dir,                       \
                                  "channelFlow_PowerLaw.ups",   \
                               [( "update", "/Uintah_specification/DataArchiver/filebase :powerLaw2.uda" ),
                                ( "update", "/Uintah_specification/Grid/BoundaryConditions/include[@href='inputs/ICE/channelFlow.xml' and @section='inletVelocity']/@type :logWindProfile" ),
                                ( "update", "/Uintah_specification/Grid/BoundaryConditions/Face[@side='x-']/BCType[@id='0' and @label='Velocity']/@var :logWindProfile" ),
                                ( "update", "/Uintah_specification/CFD/ICE/customInitialization/include[@href='inputs/ICE/channelFlow.xml']/@section :powerLawProfile2")
                               ] )
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
#       preProcessCmd="string"      - command run prior to running sus.  The command path must be defined with ADDTL_PATH
#                                     The command's final argument is the ups filename
#
#  Notes:
#  1) The "folder name" must be the same as input file without the extension.
#  2) Performance_tests are not run on a debug build.
#______________________________________________________________________

NIGHTLYTESTS = [   ("advect",             "advect.ups",              1, "All", ["exactComparison"]),
                   ("advectPeriodic",     "advect_periodic.ups",     8, "All", ["exactComparison"]),
                   ("advectScalar",       "advectScalar.ups",        4, "All", ["exactComparison"]),
                   ("riemann_1L",         riemann_1L_ups,            1, "All", ["exactComparison"]),
                   ("hotBlob2mat",        "hotBlob2mat.ups",         1, "All", ["exactComparison"]),
                   ("hotBlob2mat_sym",    "hotBlob2mat_sym.ups",     1, "All", ["exactComparison"]),
                   ("impAdvect",          "impAdvect.ups",           8, "All", ["exactComparison"]),
                   ("impAdvectPeriodic",  "impAdvect_periodic.ups",  8, "All", ["exactComparison"]),
                   ("impHotBlob",         "impHotBlob.ups",          1, "All", ["exactComparison"]),
                   ("hotBlob2mat8patch",  "hotBlob2mat8patch.ups",   8, "All", ["exactComparison"]),
                   ("waterAirOscillator", "waterAirOscillator.ups",  4, "All", ["exactComparison"]),
                   ("constantPress_BC",   "constantPress_BC.ups",    8, "All", ["exactComparison", "no_restart"]),  # dat file comparsion not working on restart
                   ("stagnationPoint",    "stagnationPoint.ups",     8, "All", ["exactComparison"]),
                   ("naturalConvection",  "naturalConvectionCavity_dx.ups",
                                                                     9, "All", ["exactComparison"])
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

INITIALIZE =  [   ("chanFlow_powerLaw",   chanFlow_powerLaw_ups,    8, "All", ["exactComparison", "no_restart"]),
                  ("chanFlow_powerLaw2",  chanFlow_powerLaw2_ups,   8, "All", ["exactComparison", "no_restart"])
              ]

DEBUGGING =   [   ("chanFlow_powerLaw",   chanFlow_powerLaw_ups,    8, "All", ["exactComparison", "no_restart", "preProcessCmd=echo2 junk junkw"]),
                  ("chanFlow_powerLaw2",  chanFlow_powerLaw2_ups,   8, "All", ["exactComparison", "no_restart", "preProcessCmd=echo junk junkw"])
              ]
#__________________________________

ADDTL_PATH=["absolutePat=/bin","relativePath tools/pfs"]           # preprocessing cmd path.  It can be an absolute or relative path from the StandAlone directory
                                                                   # syntax:  (relativePath=<path> or absolutePath=<path>)

#__________________________________
# The following line is parsed by the local RT script
# and allows the user to select the different subsets
#LIST:  AMRTESTS BUILDBOTTESTS DIFFUSION DEBUGGING INITIALIZATION LOCALTESTS LODI NIGHTLYTESTS
#__________________________________
# returns the list

NIGHTLYTESTS = NIGHTLYTESTS + AMRTESTS + DIFFUSION + LODI + INITIALIZE

def getTestList(me) :
  if me ==  "AMRTESTS":
    TESTS = AMRTESTS
  elif me == "DEBUGGING":
    TESTS = DEBUGGING
  elif me == "DIFFUSION":
    TESTS = DIFFUSION
  elif me == "INITIALIZATION":
    TESTS = INITIALIZE
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

  result = runSusTests(argv, TESTS, "ICE", ADDTL_PATH)

  # cleanup modified input files
  command = "/bin/rm -rf %s/tmp  " % (the_dir)

  system( command )

  exit( result )

