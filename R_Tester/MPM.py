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

NIGHTLYTESTS = [  
                  ("disks_complex",                       "disks_complex.ups",                       4,  "Linux", ["exactComparison"] ), \
                  ("heatcond2mat",                        "heatcond2mat.ups",                        1,  "Linux", ["exactComparison"] ),  \
                  ("inclined_plane_sphere",               "inclined_plane_sphere.ups",               1,  "Linux", ["exactComparison"] ),  \
                  ("foam_crush",                          "foam_crush.ups",                          4,  "Linux", ["exactComparison"] ),  \
                  ("periodic_disks",                      "periodic_disks.ups",                      1,  "Linux", ["exactComparison"] ),  \
                  ("periodic_spheres3D",                  "periodic_spheres3D.ups",                  8,  "Linux", ["exactComparison"] ),  \
                  ("const_test_hypo",                     "const_test_hypo.ups",                     1,  "Linux", ["exactComparison"] ),  \
                  ("const_test_cmr",                      "const_test_cmr.ups",                      1,  "Linux", ["exactComparison"] ),  \
                  ("const_test_nhp",                      "const_test_nhp.ups",                      1,  "Linux", ["exactComparison"] ),  \
                  ("const_test_vs",                       "const_test_vs.ups",                       1,  "Linux", ["exactComparison"] ),  \
                  ("adiCuJC4000s696K",                    "adiCuJC4000s696K.ups",                    1,  "Linux", ["exactComparison"] ),  \
                  ("adiCuMTS4000s696K",                   "adiCuMTS4000s696K.ups",                   1,  "Linux", ["exactComparison"] ),  \
                  ("adiCuPTW4000s696K",                   "adiCuPTW4000s696K.ups",                   1,  "Linux", ["exactComparison"] ),  \
                  ("adiCuSCG4000s696K",                   "adiCuSCG4000s696K.ups",                   1,  "Linux", ["exactComparison"] ),  \
                  ("adiCuZA4000s696K",                    "adiCuZA4000s696K.ups",                    1,  "Linux", ["exactComparison"] ),  \
                  ("test_corrug_plate",                   "test_corrug_plate.ups",                   1,  "Linux", ["exactComparison"] ),  \
                  ("test_cyl_pene_no_ero",                "test_cyl_pene_no_ero.ups",                1,  "Linux", ["exactComparison"] ),  \
                  ("test_gurson_beckerdrucker_mts",       "test_gurson_beckerdrucker_mts.ups",       1,  "Linux", ["exactComparison"] ),  \
                  ("test_hypoviscoelastic_radial_return", "test_hypoviscoelastic_radial_return.ups", 1,  "Linux", ["exactComparison"] ),  \
                  ("advect_3L_3D",                        "advect_3L_3D.ups",                        4,  "Linux", ["exactComparison", "no_restart"] ),  \
                  ("advect_2L_3D_slabs",                  "advect_2L_3D_slabs.ups",                  3,  "Linux", ["exactComparison", "no_restart"] ),  \
                  ("advect_2L_3D_edges",                  "advect_2L_3D_edges.ups",                  1,  "Linux", ["exactComparison", "no_restart"] ),  \
                  ("Charpy",                              "Charpy.ups",                              8,  "Linux", ["exactComparison"] ),  \
                  ("disks_complex",                       "disks_complex.ups",                       4,  "Darwin", ["doesTestRun"]    ),     \
                  ("heatcond2mat",                        "heatcond2mat.ups",                        1,  "Darwin", ["doesTestRun"]    ),     \
                  ("inclined_plane_sphere",               "inclined_plane_sphere.ups",               1,  "Darwin", ["doesTestRun"]    ),     \
                  ("const_test_cmr",                      "const_test_cmr.ups",                      1,  "Darwin", ["doesTestRun"]    ),     \
                  ("const_test_nhp",                      "const_test_nhp.ups",                      1,  "Darwin", ["doesTestRun"]    ),     \
                  ("adiCuJC4000s696K",                    "adiCuJC4000s696K.ups",                    1,  "Darwin", ["doesTestRun"]    ),     \
                  ("adiCuMTS4000s696K",                   "adiCuMTS4000s696K.ups",                   1,  "Darwin", ["doesTestRun"]    ),     \
                  ("adiCuPTW4000s696K",                   "adiCuPTW4000s696K.ups",                   1,  "Darwin", ["doesTestRun"]    ),     \
                  ("adiCuSCG4000s696K",                   "adiCuSCG4000s696K.ups",                   1,  "Darwin", ["doesTestRun"]    ),     \
                  ("adiCuZA4000s696K",                    "adiCuZA4000s696K.ups",                    1,  "Darwin", ["doesTestRun"]    ),     \
                  ("test_corrug_plate",                   "test_corrug_plate.ups",                   1,  "Darwin", ["doesTestRun"]    ),     \
                  ("test_cyl_pene_no_ero",                "test_cyl_pene_no_ero.ups",                1,  "Darwin", ["doesTestRun"]    ),     \
                  ("test_gurson_beckerdrucker_mts",       "test_gurson_beckerdrucker_mts.ups",       1,  "Darwin", ["doesTestRun"]    ) 
            ]
              
# Tests that are run during local regression testing              
LOCALTESTS = NIGHTLYTESTS

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

  result = runSusTests(argv, TESTS, "MPM")
  exit( result )

