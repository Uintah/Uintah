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

NIGHTLYTESTS = [  
                  #----------  Linux Tests ---------  #
                  ("disks_complex",                       "disks_complex.ups",                       4,  "Linux", ["exactComparison"] ), 
                  ("heatcond2mat",                        "heatcond2mat.ups",                        1,  "Linux", ["exactComparison"] ),  
                  ("inclined_plane_sphere",               "inclined_plane_sphere.ups",               1,  "Linux", ["exactComparison"] ),  
                  ("foam_crush",                          "foam_crush.ups",                          4,  "Linux", ["exactComparison"] ),  
                  ("periodic_disks",                      "periodic_disks.ups",                      1,  "Linux", [] ),  
                  ("periodic_spheres3D",                  "periodic_spheres3D.ups",                  8,  "Linux", ["no_dbg","exactComparison"] ),  
                  ("const_test_hypo",                     "const_test_hypo.ups",                     1,  "Linux", ["exactComparison"] ),  
                  ("const_test_cmr",                      "const_test_cmr.ups",                      1,  "Linux", ["exactComparison"] ),  
                  ("const_test_nhp",                      "const_test_nhp.ups",                      1,  "Linux", ["exactComparison"] ),  
                  ("const_test_vs",                       "const_test_vs.ups",                       1,  "Linux", ["exactComparison"] ),  
                  ("adiCuJC4000s696K",                    "adiCuJC4000s696K.ups",                    1,  "Linux", ["exactComparison"] ),  
                  ("adiCuMTS4000s696K",                   "adiCuMTS4000s696K.ups",                   1,  "Linux", ["exactComparison"] ),  
                  ("adiCuPTW4000s696K",                   "adiCuPTW4000s696K.ups",                   1,  "Linux", ["exactComparison"] ),  
                  ("adiCuSCG4000s696K",                   "adiCuSCG4000s696K.ups",                   1,  "Linux", ["exactComparison"] ),  
                  ("adiCuZA4000s696K",                    "adiCuZA4000s696K.ups",                    1,  "Linux", ["exactComparison"] ),  
                  ("test_cyl_pene_no_ero",                "test_cyl_pene_no_ero.ups",                1,  "Linux", ["exactComparison"] ),  
                  ("test_gurson_beckerdrucker_mts",       "test_gurson_beckerdrucker_mts.ups",       1,  "Linux", ["exactComparison"] ),  
                  ("test_hypoviscoelastic_radial_return", "test_hypoviscoelastic_rad_ret.ups", 1,  "Linux", ["exactComparison"] ),  
                  ("advect_3L_3D",                        "advect_3L_3D.ups",                        4,  "Linux", ["exactComparison", "no_restart"] ),  
                  ("advect_2L_3D_slabs",                  "advect_2L_3D_slabs.ups",                  3,  "Linux", [ "no_restart","no_dbg"] ),  
                  ("advect_2L_3D_edges",                  "advect_2L_3D_edges.ups",                  1,  "Linux", ["exactComparison", "no_restart"] ),  
                  ("riemannMPM_ML",                       "riemannMPM_ML.ups",                       1,  "Linux", ["exactComparison"] ),  
                  ("Charpy",                              "Charpy.ups",                              8,  "Linux", ["exactComparison"] ),  
                  
                  #("AreniscaTest_01_UniaxialStrainRotate",                  "./Arenisca/AreniscaTest_01_UniaxialStrainRotate.ups",                  1,  "Linux", ["exactComparison"] ), 
                  #("AreniscaTest_02_VertexTreatment",                       "./Arenisca/AreniscaTest_02_VertexTreatment.ups",                       1,  "Linux", ["exactComparison"] ), 
                  #("AreniscaTest_03a_UniaxialStrain_NoHardening",           "./Arenisca/AreniscaTest_03a_UniaxialStrain_NoHardening.ups",           1,  "Linux", ["exactComparison"] ),
                  #("AreniscaTest_03b_UniaxialStrain_wIsotropicHardening",   "./Arenisca/AreniscaTest_03b_UniaxialStrain_wIsotropicHardening.ups",   1,  "Linux", ["exactComparison"] ),                  
                  #("AreniscaTest_03c_UniaxialStrain_wKinematicHardening",   "./Arenisca/AreniscaTest_03c_UniaxialStrain_wKinematicHardening.ups",   1,  "Linux", ["exactComparison"] ),
                  #("AreniscaTest_04_CurvedYieldSurface",                    "./Arenisca/AreniscaTest_04_CurvedYieldSurface.ups",                    1,  "Linux", ["exactComparison"] ),
                  #("AreniscaTest_05_HydrostaticCompressionFixedCap",        "./Arenisca/AreniscaTest_05_HydrostaticCompressionFixedCap.ups",        1,  "Linux", ["exactComparison"] ),
                  #("AreniscaTest_06_UniaxialStrainCapEvolution",            "./Arenisca/AreniscaTest_06_UniaxialStrainCapEvolution.ups",            1,  "Linux", ["exactComparison"] ),
                  #("AreniscaTest_07_HydrostaticCompressionCapEvolution",    "./Arenisca/AreniscaTest_07_HydrostaticCompressionCapEvolution.ups",    1,  "Linux", ["exactComparison"] ),
                  #("AreniscaTest_08_LoadingUnloading",                      "./Arenisca/AreniscaTest_08_LoadingUnloading.ups",                      1,  "Linux", ["exactComparison"] ),
                  #("AreniscaTest_09_FluidFilledPoreSpace",                  "./Arenisca/AreniscaTest_09_FluidFilledPoreSpace.ups",                  1,  "Linux", ["exactComparison"] ),                              
                  #("AreniscaTest_10_PureIsochoricStrainRates",              "./Arenisca/AreniscaTest_10_PureIsochoricStrainRates.ups",              1,  "Linux", ["exactComparison"] ),                   
                  #("AreniscaTest_11_UniaxialStrainJ2plasticity",            "./Arenisca/AreniscaTest_11_UniaxialStrainJ2plasticity.ups",            1,  "Linux", ["exactComparison"] ),                    

                  #----------  Darwin Tests ---------  #
                  ("disks_complex",                       "disks_complex.ups",                       4,  "Darwin", ["doesTestRun"]    ),     
                  ("heatcond2mat",                        "heatcond2mat.ups",                        1,  "Darwin", ["doesTestRun"]    ),     
                  ("inclined_plane_sphere",               "inclined_plane_sphere.ups",               1,  "Darwin", ["doesTestRun"]    ),     
                  ("const_test_cmr",                      "const_test_cmr.ups",                      1,  "Darwin", ["doesTestRun"]    ),     
                  ("const_test_nhp",                      "const_test_nhp.ups",                      1,  "Darwin", ["doesTestRun"]    ),     
                  ("adiCuJC4000s696K",                    "adiCuJC4000s696K.ups",                    1,  "Darwin", ["doesTestRun"]    ),     
                  ("adiCuMTS4000s696K",                   "adiCuMTS4000s696K.ups",                   1,  "Darwin", ["doesTestRun"]    ),     
                  ("adiCuPTW4000s696K",                   "adiCuPTW4000s696K.ups",                   1,  "Darwin", ["doesTestRun"]    ),     
                  ("adiCuSCG4000s696K",                   "adiCuSCG4000s696K.ups",                   1,  "Darwin", ["doesTestRun"]    ),     
                  ("adiCuZA4000s696K",                    "adiCuZA4000s696K.ups",                    1,  "Darwin", ["doesTestRun"]    ),     
                  ("test_cyl_pene_no_ero",                "test_cyl_pene_no_ero.ups",                1,  "Darwin", ["doesTestRun"]    ),     
                  ("test_gurson_beckerdrucker_mts",       "test_gurson_beckerdrucker_mts.ups",       1,  "Darwin", ["doesTestRun"]    ),
                  
                  #("AreniscaTest_01_UniaxialStrainRotate",                  "./Arenisca/AreniscaTest_01_UniaxialStrainRotate.ups",                  1,  "Darwin", ["exactComparison"] ), 
                  #("AreniscaTest_02_VertexTreatment",                       "./Arenisca/AreniscaTest_02_VertexTreatment.ups",                       1,  "Darwin", ["exactComparison"] ), 
                  #("AreniscaTest_03a_UniaxialStrain_NoHardening",           "./Arenisca/AreniscaTest_03a_UniaxialStrain_NoHardening.ups",           1,  "Darwin", ["exactComparison"] ),
                  #("AreniscaTest_03b_UniaxialStrain_wIsotropicHardening",   "./Arenisca/AreniscaTest_03b_UniaxialStrain_wIsotropicHardening.ups",   1,  "Darwin", ["exactComparison"] ),                  
                  #("AreniscaTest_03c_UniaxialStrain_wKinematicHardening",   "./Arenisca/AreniscaTest_03c_UniaxialStrain_wKinematicHardening.ups",   1,  "Darwin", ["exactComparison"] ),
                  #("AreniscaTest_04_CurvedYieldSurface",                    "./Arenisca/AreniscaTest_04_CurvedYieldSurface.ups",                    1,  "Darwin", ["exactComparison"] ),
                  #("AreniscaTest_05_HydrostaticCompressionFixedCap",        "./Arenisca/AreniscaTest_05_HydrostaticCompressionFixedCap.ups",        1,  "Darwin", ["exactComparison"] ),
                  #("AreniscaTest_06_UniaxialStrainCapEvolution",            "./Arenisca/AreniscaTest_06_UniaxialStrainCapEvolution.ups",            1,  "Darwin", ["exactComparison"] ),
                  #("AreniscaTest_07_HydrostaticCompressionCapEvolution",    "./Arenisca/AreniscaTest_07_HydrostaticCompressionCapEvolution.ups",    1,  "Darwin", ["exactComparison"] ),
                  #("AreniscaTest_08_LoadingUnloading",                      "./Arenisca/AreniscaTest_08_LoadingUnloading.ups",                      1,  "Darwin", ["exactComparison"] ),
                  #("AreniscaTest_09_FluidFilledPoreSpace",                  "./Arenisca/AreniscaTest_09_FluidFilledPoreSpace.ups",                  1,  "Darwin", ["exactComparison"] ),                  
                  #("AreniscaTest_10_PureIsochoricStrainRates",              "./Arenisca/AreniscaTest_10_PureIsochoricStrainRates.ups",              1,  "Darwin", ["exactComparison"] ), 
                  #("AreniscaTest_11_UniaxialStrainJ2plasticity",            "./Arenisca/AreniscaTest_11_UniaxialStrainJ2plasticity.ups",            1,  "Darwin", ["exactComparison"] ),                   
            ]
              
# Tests that are run during local regression testing              
LOCALTESTS = NIGHTLYTESTS
DEBUGTESTS =[]

#__________________________________
# The following list is parsed by the local RT script
# and allows the user to select the tests to run
#LIST: LOCALTESTS DEBUGTESTS NIGHTLYTESTS
#__________________________________

# returns the list  
def getTestList(me) :
  if me == "LOCALTESTS":
    TESTS = LOCALTESTS
  elif me == "DEBUGTESTS":
    TESTS = DEBUGTESTS
  elif me == "NIGHTLYTESTS":
    TESTS = NIGHTLYTESTS
  else:
    print "\nERROR:MPM.py  getTestList:  The test list (%s) does not exist!\n\n" % me
    exit(1)
  return TESTS
#__________________________________

if __name__ == "__main__":

  TESTS = getTestList( environ['WHICH_TESTS'] )

  result = runSusTests(argv, TESTS, "MPM")
  exit( result )

