#!/usr/bin/env python

from os import symlink,environ
from sys import argv,exit,platform
from helpers.runSusTests import runSusTests, inputs_root, generatingGoldStandards
from helpers.modUPS import modUPS

the_dir = generatingGoldStandards()

if the_dir == "" :
  the_dir = "%s/ARCHES" % inputs_root()
else :
  the_dir = the_dir + "/ARCHES"

methanePetscRadSolver_ups = modUPS( the_dir,                 \
                            "methane_explicit_table.ups",    \
                            ["<patches>[2,2,2]</patches>",   \
                             "<LinearSolver type=\"petsc\">"] )
                           
#______________________________________________________________________                            
#  Test syntax: ( "folder name", "input file", # processors, "OS", ["flags1","flag2"])
#  flags: 
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
#
#  Notes: 
#  1) The "folder name" must be the same as input file without the extension.
#  2) If the processors is > 1.0 then an mpirun command will be used
#  3) Performance_tests are not run on a debug build.
#______________________________________________________________________

UNUSED_TESTS = []
NIGHTLYTESTS = [
   ("constantMMS"            , "mms/constantMMS.ups"                     , 1.1 , "Linux"  , ["exactComparison"]) , 
   ("almgrenMMS"             , "mms/almgrenMMS.ups"                      , 1.1 , "Linux"  , ["exactComparison"]) , 
   ("periodic"               , "periodicTurb/periodic.ups"               , 1.1 , "Linux"  , ["exactComparison"]) , 
   ("helium_RT"              , "helium_RT.ups"                           , 1.1 , "Linux"  , ["exactComparison"]) , 
   ("periodic"               , "periodicTurb/periodic.ups"               , 1.1 , "Darwin" , ["doesTestRun"       , "no_dbg"]) , 
   ("methane_explicit_table" , "methane_explicit_table.ups"              , 1.1 , "Linux"  , ["exactComparison"]) , 
   ("methane_explicit_table" , "methane_explicit_table.ups"              , 1.1 , "Darwin" , ["doesTestRun"])     , 
   ("methane8patch"          , "methane8patch.ups"                       , 8   , "Linux"  , ["exactComparison"]) , 
   ("methane8patch"          , "methane8patch.ups"                       , 8   , "Darwin" , ["doesTestRun"])     , 
   ("methanePetscRadSolver"  , methanePetscRadSolver_ups                 , 8   , "Linux"  , ["exactComparison"]) , 
   ("dqmom_test_1"           , "DQMOM_regression/dqmom_test_1.ups"       , 1.1 , "Linux"  , ["exactComparison"]) , 
   ("dqmom_test_2"           , "DQMOM_regression/dqmom_test_2.ups"       , 1.1 , "Linux"  , ["exactComparison"]) , 
   ("dqmom_test_3"           , "DQMOM_regression/dqmom_test_3.ups"       , 1.1 , "Linux"  , ["exactComparison"]) , 
   ("dqmom_test_4"           , "DQMOM_regression/dqmom_test_4.ups"       , 1.1 , "Linux"  , ["exactComparison"]) , 
   ("dqmom_test_5"           , "DQMOM_regression/dqmom_test_5.ups"       , 1.1 , "Linux"  , ["exactComparison"]) , 
   ("methane_jet"            , "ClassicMixingTables/ups/methane_jet.ups" , 1.1 , "Linux"  , ["exactComparison", "no_restart"]) , 
   ("acidbase_jet"           , "ClassicMixingTables/ups/acidbase_jet.ups", 1.1 , "Linux"  , ["exactComparison", "no_restart"]) ,
   ("acidbase_jet_2D"        , "ClassicMixingTables/ups/acidbase_jet_2D.ups", 1.1, "Linux", ["exactComparison", "no_restart"]) ,
   ("1Dtabletest"            , "ClassicMixingTables/ups/1DTableTest.ups" , 1.1 , "Linux"  , ["exactComparison", "no_restart"]) , 
   ("xplus_scalar_test"      , "ScalarTests/xplus_scalar_test.ups"       , 6   , "Linux"  , ["exactComparison", "no_restart"]) , 
   ("yplus_scalar_test"      , "ScalarTests/yplus_scalar_test.ups"       , 6   , "Linux"  , ["exactComparison", "no_restart"]) , 
   ("zplus_scalar_test"      , "ScalarTests/zplus_scalar_test.ups"       , 6   , "Linux"  , ["exactComparison", "no_restart"]) , 
   ("xminus_scalar_test"     , "ScalarTests/xminus_scalar_test.ups"      , 6   , "Linux"  , ["exactComparison", "no_restart"]) , 
   ("yminus_scalar_test"     , "ScalarTests/yminus_scalar_test.ups"      , 6   , "Linux"  , ["exactComparison", "no_restart"]) , 
   ("zminus_scalar_test"     , "ScalarTests/zminus_scalar_test.ups"      , 6   , "Linux"  , ["exactComparison", "no_restart"])
]

# Tests that are run during local regression testing
LOCALTESTS = [
#   ("constantMMS"            , "mms/constantMMS.ups"                     , 1.1 , "All"   , ["exactComparison"]) , 
#   ("almgrenMMS"             , "mms/almgrenMMS.ups"                      , 1.1 , "All"   , ["exactComparison"]) , 
#   ("periodic"               , "periodicTurb/periodic.ups"               , 1.1 , "All"   , ["exactComparison"]) , 
#   ("helium_RT"              , "helium_RT.ups"                           , 1.1 , "All"   , ["exactComparison"]) , 
#   ("methane_explicit_table" , "methane_explicit_table.ups"              , 1.1 , "All"   , ["exactComparison"]) , 
#   ("methane8patch"          , "methane8patch.ups"                       , 8   , "All"   , ["exactComparison"]) , 
#   ("methanePetscRadSolver"  , methanePetscRadSolver_ups                 , 8   , "All"   , ["exactComparison"]) , 
#   ("dqmom_test_1"           , "DQMOM_regression/dqmom_test_1.ups"       , 1.1 , "All"   , ["exactComparison"]) , 
#   ("dqmom_test_2"           , "DQMOM_regression/dqmom_test_2.ups"       , 1.1 , "All"   , ["exactComparison"]) , 
#   ("dqmom_test_3"           , "DQMOM_regression/dqmom_test_3.ups"       , 1.1 , "All"   , ["exactComparison"]) , 
#   ("dqmom_test_4"           , "DQMOM_regression/dqmom_test_4.ups"       , 1.1 , "All"   , ["exactComparison"]) , 
#   ("dqmom_test_5"           , "DQMOM_regression/dqmom_test_5.ups"       , 1.1 , "All"   , ["exactComparison"]) , 
#   ("methane_jet"            , "ClassicMixingTables/ups/methane_jet.ups" , 1.1 , "All"   , ["exactComparison", "no_restart"]) ,
#   ("acidbase_jet"           , "ClassicMixingTables/ups/acidbase_jet.ups", 1.1 , "All"   , ["exactComparison", "no_restart"]) ,
#   ("acidbase_jet_2D"        , "ClassicMixingTables/ups/acidbase_jet_2D.ups", 1.1, "All" , ["exactComparison", "no_restart"]) ,
#   ("1DTableTest"            , "ClassicMixingTables/ups/1DTableTest.ups" , 1.1 , "All"   , ["exactComparison", "no_restart"]) ,
   ("xplus_scalar_test"      , "ScalarTests/xplus_scalar_test.ups"       , 6   , "All"  , ["exactComparison", "no_restart"]) , 
   ("yplus_scalar_test"      , "ScalarTests/yplus_scalar_test.ups"       , 6   , "All"  , ["exactComparison", "no_restart"]) , 
   ("zplus_scalar_test"      , "ScalarTests/zplus_scalar_test.ups"       , 6   , "All"  , ["exactComparison", "no_restart"]) , 
   ("xminus_scalar_test"     , "ScalarTests/xminus_scalar_test.ups"      , 6   , "All"  , ["exactComparison", "no_restart"]) , 
   ("yminus_scalar_test"     , "ScalarTests/yminus_scalar_test.ups"      , 6   , "All"  , ["exactComparison", "no_restart"]) , 
   ("zminus_scalar_test"     , "ScalarTests/zminus_scalar_test.ups"      , 6   , "All"  , ["exactComparison", "no_restart"]) , 
]

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

  result = runSusTests(argv, TESTS, "ARCHES")
  exit( result )


