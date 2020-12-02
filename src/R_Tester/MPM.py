#!/usr/bin/env python3

from sys import argv,exit
from os import environ
from helpers.runSusTests_git import runSusTests, ignorePerformanceTests

#______________________________________________________________________
#  Test syntax: ( "folder name", "input file", # processors, "OS", ["flags1","flag2"])
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

NIGHTLYTESTS = [
                  #----------  All Tests ---------  #
                  ("disks_complex",                       "disks_complex.ups",                       6,  "ALL", ["exactComparison"] ),
                  ("heatcond2mat",                        "heatcond2mat.ups",                        1,  "ALL", ["exactComparison"] ),
                  ("NairnFrictionTest",                   "NairnFrictionTest.ups",               1,  "ALL", ["exactComparison"] ),
                  ("foam_crush",                          "foam_crush.ups",                          4,  "ALL", ["exactComparison"] ),
                  ("ColPlate_AffineTrans",                "ColPlate_AffineTrans.ups",                1,  "ALL", ["exactComparison"] ),
                  ("extrudeRT",                           "extrudeRT.ups",                           5,  "ALL", ["exactComparison"] ),
#                  ("periodic_disks",                      "periodic_disks.ups",                      1,  "ALL", [] ),
#                  ("periodic_disks",                      "periodic_disks.ups",                      1,  "ALL", [] ),
#                  ("periodic_spheres3D",                  "periodic_spheres3D.ups",                  8,  "All", ["no_dbg","exactComparison"] ),
#                  ("const_test_hypo",                     "const_test_hypo.ups",                     1,  "ALL", ["exactComparison"] ),
#                  ("const_test_cmr",                      "const_test_cmr.ups",                      1,  "ALL", ["exactComparison"] ),
                  ("const_test_nhp",                      "const_test_nhp.ups",                      1,  "ALL", ["exactComparison"] ),
                  ("const_test_vs",                       "const_test_vs.ups",                       1,  "ALL", ["exactComparison"] ),
                  ("adiCuJC4000s696K",                    "adiCuJC4000s696K.ups",                    1,  "ALL", ["exactComparison"] ),
#                  ("adiCuMTS4000s696K",                   "adiCuMTS4000s696K.ups",                   1,  "All", ["exactComparison"] ),
#                  ("adiCuPTW4000s696K",                   "adiCuPTW4000s696K.ups",                   1,  "All", ["exactComparison"] ),
#                  ("adiCuSCG4000s696K",                   "adiCuSCG4000s696K.ups",                   1,  "All", ["exactComparison"] ),
#                  ("adiCuZA4000s696K",                    "adiCuZA4000s696K.ups",                    1,  "All", ["exactComparison"] ),
                  ("test_cyl_pene_no_ero_axi_sym",                "test_cyl_pene_no_ero_axi_sym.ups",                4,  "ALL", ["exactComparison"] ),
                  ("test_gurson_beckerdrucker_mts",       "test_gurson_beckerdrucker_mts.ups",       1,  "ALL", ["exactComparison"] ),
                  ("test_hypoviscoelastic_radial_return", "test_hypoviscoelastic_rad_ret.ups",       1,  "ALL", ["exactComparison"] ),
                    ("NanoPillar", "ARL/NanoPillar2D_FBC_Sym.ups",                                   1,  "ALL", ["no_dbg","exactComparison"] ),
                  #("AreniscaTest_01_UniaxialStrainRotate",                  "./Arenisca/AreniscaTest_01_UniaxialStrainRotate.ups",                  1,  "All", ["exactComparison"] ),
                  #("AreniscaTest_02_VertexTreatment",                       "./Arenisca/AreniscaTest_02_VertexTreatment.ups",                       1,  "All", ["exactComparison"] ),
                  #("AreniscaTest_03a_UniaxialStrain_NoHardening",           "./Arenisca/AreniscaTest_03a_UniaxialStrain_NoHardening.ups",           1,  "All", ["exactComparison"] ),
                  #("AreniscaTest_03b_UniaxialStrain_wIsotropicHardening",   "./Arenisca/AreniscaTest_03b_UniaxialStrain_wIsotropicHardening.ups",   1,  "All", ["exactComparison"] ),
                  #("AreniscaTest_03c_UniaxialStrain_wKinematicHardening",   "./Arenisca/AreniscaTest_03c_UniaxialStrain_wKinematicHardening.ups",   1,  "All", ["exactComparison"] ),
                  #("AreniscaTest_04_CurvedYieldSurface",                    "./Arenisca/AreniscaTest_04_CurvedYieldSurface.ups",                    1,  "All", ["exactComparison"] ),
                  #("AreniscaTest_05_HydrostaticCompressionFixedCap",        "./Arenisca/AreniscaTest_05_HydrostaticCompressionFixedCap.ups",        1,  "All", ["exactComparison"] ),
                  #("AreniscaTest_06_UniaxialStrainCapEvolution",            "./Arenisca/AreniscaTest_06_UniaxialStrainCapEvolution.ups",            1,  "All", ["exactComparison"] ),
                  #("AreniscaTest_07_HydrostaticCompressionCapEvolution",    "./Arenisca/AreniscaTest_07_HydrostaticCompressionCapEvolution.ups",    1,  "All", ["exactComparison"] ),
                  #("AreniscaTest_08_LoadingUnloading",                      "./Arenisca/AreniscaTest_08_LoadingUnloading.ups",                      1,  "All", ["exactComparison"] ),
                  #("AreniscaTest_09_FluidFilledPoreSpace",                  "./Arenisca/AreniscaTest_09_FluidFilledPoreSpace.ups",                  1,  "All", ["exactComparison"] ),
                  #("AreniscaTest_10_PureIsochoricStrainRates",              "./Arenisca/AreniscaTest_10_PureIsochoricStrainRates.ups",              1,  "All", ["exactComparison"] ),
                  #("AreniscaTest_11_UniaxialStrainJ2plasticity",            "./Arenisca/AreniscaTest_11_UniaxialStrainJ2plasticity.ups",            1,  "All", ["exactComparison"] ),

            ]

PERFORMANCETESTS = [ ("mpm_perf_test",                       "inclined_plane_sphere.ups",                 1, "All", ["do_performance_test"]),
                ]

AMRTESTS = [
                  ("advect_3L_1D",                        "advect_3L_1D.ups",           1,  "ALL", ["exactComparison"] ),
                  ("advect_3L_3D",                        "advect_3L_3D.ups",           4,  "ALL", ["no_restart"] ),
                  ("advect_2L_3D_slabs",                  "advect_2L_3D_slabs.ups",     3,  "ALL", [ "no_restart","no_dbg"] ),
                  ("advect_2L_3D_edges",                  "advect_2L_3D_edges.ups",     1,  "ALL", ["exactComparison", "no_restart"] ),
                  ("riemannMPM_ML",                       "riemannMPM_ML.ups",          1,  "ALL", ["exactComparison"] ),
#                  ("Collide_AMR_3L",                      "Collide_AMR_3L.ups",         1,  "All", ["exactComparison"] ),
            ]

ARENATESTS = [
                  ("HydrostaticCompressionSaturated",
                   "ArenaSoilBanerjeeBrannon/HydrostaticCompressionSaturated.ups",
                   1,
                   "All",
                   ["exactComparison"] ),
                  ("MultiaxialStrainLoadUnload",
                   "ArenaSoilBanerjeeBrannon/MultiaxialStrainLoadUnload.ups",
                   1,
                   "All",
                   ["exactComparison"] ),
                  ("BoulderClaySHPB072213-014",
                   "ArenaSoilBanerjeeBrannon/BoulderClaySHPB072213-014.ups",
                   1,
                   "All",
                   ["exactComparison"] ),
            ]

              #__________________________________
              # Tests that exercise the damage models
DAMAGETESTS = [   ("const_test_brittle_damage", "const_test_brittle_damage.ups",        1,  "All", ["exactComparison"] ),
                  ("PressSmoothCylCBDI",        "PressSmoothCylCBDI.ups",               16, "All", ["exactComparison"] ),
                  ("disks_complex",             "disks_complex.ups",                    4,  "All", ["exactComparison"] ),
                  ("halfSpaceUCNH_EP_JWLMPM",   "ONR-MURI/halfSpaceUCNH_EP_JWLMPM.ups", 16, "All", ["exactComparison"] ),
              ]

THREADEDTESTS = [ ("Charpy",    "Charpy.ups",    2,  "ALL", ["exactComparison", "sus_options=-nthreads 4"] ),
                ]

# Tests that are run during local regression testing
NIGHTLYTESTS = NIGHTLYTESTS + AMRTESTS + THREADEDTESTS + PERFORMANCETESTS

LOCALTESTS = NIGHTLYTESTS
DEBUGTESTS =[("Charpy",                "Charpy.ups",                  8,  "All", ["exactComparison"] ),
             ("test_cyl_pene_no_ero",  "test_cyl_pene_no_ero.ups",    4,  "All", ["exactComparison"] ),
            ]

#__________________________________
# The following list is parsed by the local RT script
# and allows the user to select the tests to run
#LIST: LOCALTESTS DAMAGETESTS DEBUGTESTS NIGHTLYTESTS AMRTESTS ARENATESTS BUILDBOTTESTS PERFORMANCETESTS
#__________________________________

# returns the list
def getTestList(me) :
  if me == "LOCALTESTS":
    TESTS = LOCALTESTS
  elif me == "DAMAGETESTS":
    TESTS = DAMAGETESTS
  elif me == "DEBUGTESTS":
    TESTS = DEBUGTESTS
  elif me == "NIGHTLYTESTS":
    TESTS = NIGHTLYTESTS
  elif me == "PERFORMANCETESTS":
    TESTS = PERFORMANCETESTS
  elif me == "AMRTESTS":
    TESTS = AMRTESTS
  elif me == "ARENATESTS":
    TESTS = ARENATESTS
  elif me == "BUILDBOTTESTS":
    TESTS = ignorePerformanceTests( NIGHTLYTESTS )
  else:
    print("\nERROR:MPM.py  getTestList:  The test list (%s) does not exist!\n\n" % me)
    exit(1)
  return TESTS
#__________________________________

if __name__ == "__main__":

  TESTS = getTestList( environ['WHICH_TESTS'] )

  result = runSusTests(argv, TESTS, "MPM")
  exit( result )

