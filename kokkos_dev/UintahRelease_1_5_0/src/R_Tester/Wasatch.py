#!/usr/bin/env python
 
from sys import argv, exit
from os import environ
from helpers.runSusTests import runSusTests, inputs_root, generatingGoldStandards
from helpers.modUPS import modUPS

the_dir = generatingGoldStandards()

if the_dir == "" :
  the_dir = "%s/Wasatch" % inputs_root()
else :
  the_dir = the_dir + "/Wasatch"

liddrivencavity3DRe1000rk3_ups = modUPS( the_dir, \
                                       "lid-driven-cavity-3D-Re1000.ups", \
                                       ["<TimeIntegrator> RK3SSP </TimeIntegrator>", \
                                       "<filebase>liddrivencavity3DRe1000rk3.uda</filebase>"])

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

UNUSED_TESTS = []

NIGHTLYTESTS = [
  ("read-from-file-test",   "read-from-file-test.ups",   8,  "Linux",   ["exactComparison","no_restart"] ),                   \
  ("channel-flow-symmetry-bc",   "channel-flow-symmetry-bc.ups",   6,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("turb-lid-driven-cavity-3D-WALE",   "turb-lid-driven-cavity-3D-WALE.ups",   8,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("turb-lid-driven-cavity-3D-SMAGORINSKY",   "turb-lid-driven-cavity-3D-SMAGORINSKY.ups",   8,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("turb-lid-driven-cavity-3D-scalar",   "turb-lid-driven-cavity-3D-SMAGORINSKY-scalar.ups",   8,  "Linux",  ["exactComparison","no_restart"] ),               \
  ("channel-flow-xy-xminus-pressure-outlet",   "channel-flow-xy-xminus-pressure-outlet.ups",   6,  "Linux",  ["exactComparison","no_restart"] ),               \
  ("channel-flow-xy-xplus-pressure-outlet",    "channel-flow-xy-xplus-pressure-outlet.ups",    6,  "Linux",  ["exactComparison","no_restart"] ),               \
  ("channel-flow-xz-zminus-pressure-outlet",   "channel-flow-xz-zminus-pressure-outlet.ups",   6,  "Linux",  ["exactComparison","no_restart"] ),               \
  ("channel-flow-xz-zplus-pressure-outlet",    "channel-flow-xz-zplus-pressure-outlet.ups",    6,  "Linux",  ["exactComparison","no_restart"] ),               \
  ("channel-flow-yz-yminus-pressure-outlet",   "channel-flow-yz-yminus-pressure-outlet.ups",   6,  "Linux",  ["exactComparison","no_restart"] ),               \
  ("channel-flow-yz-yplus-pressure-outlet",    "channel-flow-yz-yplus-pressure-outlet.ups",    6,  "Linux",  ["exactComparison","no_restart"] ),               \
  ("lid-driven-cavity-3D-Re1000",   "lid-driven-cavity-3D-Re1000.ups",   8,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("liddrivencavity3DRe1000rk3",    liddrivencavity3DRe1000rk3_ups,   8,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("lid-driven-cavity-xy-Re1000",   "lid-driven-cavity-xy-Re1000.ups",   4,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("lid-driven-cavity-xz-Re1000",   "lid-driven-cavity-xz-Re1000.ups",   4,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("lid-driven-cavity-yz-Re1000",   "lid-driven-cavity-yz-Re1000.ups",   4,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("hydrostatic-pressure-test",     "hydrostatic-pressure-test.ups",     8,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("BasicScalarTransportEquation",  "BasicScalarTransportEquation.ups",  1,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("BasicScalarTransportEq_2L",     "BasicScalarTransportEq_2L.ups",     1,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ), \
  ("TabPropsInterface",             "TabPropsInterface.ups",             1,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ), \
  ("bc-test-mixed",                 "bc-test-mixed.ups",                 4,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("ScalarTransportEquation",       "ScalarTransportEquation.ups",       1,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("momentum-test-mms-xy",          "momentum-test-mms-xy.ups",          4,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("momentum-test-mms-xz",          "momentum-test-mms-xz.ups",          4,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("momentum-test-mms-yz",          "momentum-test-mms-yz.ups",          4,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("scalability-test",              "scalability-test.ups",              1,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("momentum-test-mms-3D",          "momentum-test-mms-3D.ups",          8,  "Linux",  ["exactComparison","no_restart"] ),                 \
  ("bc-test-svol-xdir",             "bc-test-svol-xdir.ups",             4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("bc-test-svol-ydir",             "bc-test-svol-ydir.ups",             4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("bc-test-svol-zdir",             "bc-test-svol-zdir.ups",             4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("qmom-realizable-test",          "qmom-realizable-test.ups",          8,  "Linux",  ["exactComparison","no_restart"] ),   \
  ("qmom-test",                     "qmom-test.ups",                     4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),  \
  ("convection-test-svol-xdir",     "convection-test-svol-xdir.ups",     4,  "Linux",  ["exactComparison","no_restart"] ),  \
  ("convection-test-svol-ydir",     "convection-test-svol-ydir.ups",     4,  "Linux",  ["exactComparison","no_restart"] ),  \
  ("convection-test-svol-zdir",     "convection-test-svol-zdir.ups",     4,  "Linux",  ["exactComparison","no_restart"] ),  \
  ("convection-test-svol-xdir-bc",  "convection-test-svol-xdir-bc.ups",  8,  "Linux",  ["exactComparison","no_restart"] ),  \
  ("convection-test-svol-ydir-bc",  "convection-test-svol-ydir-bc.ups",  8,  "Linux",  ["exactComparison","no_restart"] ),  \
  ("convection-test-svol-zdir-bc",  "convection-test-svol-zdir-bc.ups",  8,  "Linux",  ["exactComparison","no_restart"] ),  \
  ("convection-test-svol-mixed-bc", "convection-test-svol-mixed-bc.ups", 8,  "Linux",  ["exactComparison","no_restart"] ),  \
  ("force-on-graph-postprocessing-test",     "force-on-graph-postprocessing-test.ups",   4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] )
]


# Tests that are run during local regression testing
LOCALTESTS = [
  ("read-from-file-test",   "read-from-file-test.ups",   8,  "All",   ["exactComparison","no_restart"] ),                   \
  ("channel-flow-symmetry-bc",   "channel-flow-symmetry-bc.ups",   6,  "All",   ["exactComparison","no_restart"] ),                   \
  ("turb-lid-driven-cavity-3D-WALE",   "turb-lid-driven-cavity-3D-WALE.ups",   8,  "All",  ["exactComparison","no_restart"] ),                 \
  ("turb-lid-driven-cavity-3D-SMAGORINSKY",   "turb-lid-driven-cavity-3D-SMAGORINSKY.ups",   8,  "All",  ["exactComparison","no_restart"] ),                 \
  ("turb-lid-driven-cavity-3D-scalar",   "turb-lid-driven-cavity-3D-SMAGORINSKY-scalar.ups",   8,  "All",  ["exactComparison","no_restart"] ),               \
  ("channel-flow-xy-xminus-pressure-outlet",   "channel-flow-xy-xminus-pressure-outlet.ups",   6,  "All",  ["exactComparison","no_restart"] ),               \
  ("channel-flow-xy-xplus-pressure-outlet",    "channel-flow-xy-xplus-pressure-outlet.ups",    6,  "All",  ["exactComparison","no_restart"] ),               \
  ("channel-flow-xz-zminus-pressure-outlet",   "channel-flow-xz-zminus-pressure-outlet.ups",   6,  "All",  ["exactComparison","no_restart"] ),               \
  ("channel-flow-xz-zplus-pressure-outlet",    "channel-flow-xz-zplus-pressure-outlet.ups",    6,  "All",  ["exactComparison","no_restart"] ),               \
  ("channel-flow-yz-yminus-pressure-outlet",   "channel-flow-yz-yminus-pressure-outlet.ups",   6,  "All",  ["exactComparison","no_restart"] ),               \
  ("channel-flow-yz-yplus-pressure-outlet",    "channel-flow-yz-yplus-pressure-outlet.ups",    6,  "All",  ["exactComparison","no_restart"] ),               \
  ("lid-driven-cavity-3D-Re1000",   "lid-driven-cavity-3D-Re1000.ups",   8,  "All",   ["exactComparison","no_memoryTest"] ),                \
  ("liddrivencavity3DRe1000rk3",   liddrivencavity3DRe1000rk3_ups,   8,  "All",  ["exactComparison","no_restart"] ),                        \
  ("lid-driven-cavity-xy-Re1000",   "lid-driven-cavity-xy-Re1000.ups",   4,  "All",   ["exactComparison","no_restart"] ),                   \
  ("lid-driven-cavity-xz-Re1000",   "lid-driven-cavity-xz-Re1000.ups",   4,  "All",   ["exactComparison","no_restart"] ),                   \
  ("lid-driven-cavity-yz-Re1000",   "lid-driven-cavity-yz-Re1000.ups",   4,  "All",   ["exactComparison","no_restart"] ),                   \
  ("hydrostatic-pressure-test",     "hydrostatic-pressure-test.ups",     8,  "All",   ["exactComparison","no_restart"] ),                   \
  ("BasicScalarTransportEquation", "BasicScalarTransportEquation.ups",   1,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("BasicScalarTransportEq_2L",     "BasicScalarTransportEq_2L.ups",     1,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("TabPropsInterface",             "TabPropsInterface.ups",             1,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("bc-test-mixed",                 "bc-test-mixed.ups",                 4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("ScalarTransportEquation",       "ScalarTransportEquation.ups",       1,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("momentum-test-mms-xy",          "momentum-test-mms-xy.ups",          4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("momentum-test-mms-xz",          "momentum-test-mms-xz.ups",          4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("momentum-test-mms-yz",          "momentum-test-mms-yz.ups",          4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("scalability-test",              "scalability-test.ups",              1,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("momentum-test-mms-3D",          "momentum-test-mms-3D.ups",          8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("bc-test-svol-xdir",             "bc-test-svol-xdir.ups",             4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("bc-test-svol-ydir",             "bc-test-svol-ydir.ups",             4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("bc-test-svol-zdir",             "bc-test-svol-zdir.ups",             4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("qmom-realizable-test",          "qmom-realizable-test.ups",          8,  "All",   ["exactComparison","no_restart"] ),   \
  ("qmom-test",                     "qmom-test.ups",                     4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("convection-test-svol-xdir",     "convection-test-svol-xdir.ups",     4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("convection-test-svol-ydir",     "convection-test-svol-ydir.ups",     4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("convection-test-svol-zdir",     "convection-test-svol-zdir.ups",     4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("convection-test-svol-xdir-bc",  "convection-test-svol-xdir-bc.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("convection-test-svol-ydir-bc",  "convection-test-svol-ydir-bc.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("convection-test-svol-zdir-bc",  "convection-test-svol-zdir-bc.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("convection-test-svol-mixed-bc", "convection-test-svol-mixed-bc.ups", 8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),   \
  ("force-on-graph-postprocessing-test",     "force-on-graph-postprocessing-test.ups",   4,  "All",  ["exactComparison","no_restart","no_memoryTest"] )
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

  result = runSusTests(argv, TESTS, "Wasatch")
  exit( result )

