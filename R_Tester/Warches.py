#!/usr/bin/env python
 
from sys import argv, exit
from os import environ
from helpers.runSusTests import runSusTests, inputs_root, generatingGoldStandards
from helpers.modUPS import modUPS

the_dir = generatingGoldStandards()

if the_dir == "" :
  the_dir = "%s/Warches" % inputs_root()
else :
  the_dir = the_dir + "/Warches"

# liddrivencavity3DRe1000rk3_ups = modUPS( the_dir, \
#                                        "lid-driven-cavity-3D-Re1000.ups", \
#                                        ["<TimeIntegrator> RK3SSP </TimeIntegrator>", \
#                                        "<filebase>liddrivencavity3DRe1000rk3.uda</filebase>"])
# liddrivencavity3Dlaminarperf_ups = modUPS( the_dir, \
#                                        "lid-driven-cavity-3D-Re1000.ups", \
#                                        ["<max_Timesteps> 50 </max_Timesteps>","<resolution>[100,100,100]</resolution>","<patches>[1,1,1]</patches>"])
# liddrivencavity3Dvremanperf_ups = modUPS( the_dir, \
#                                        "turb-lid-driven-cavity-3D-VREMAN.ups", \
#                                        ["<max_Timesteps> 50 </max_Timesteps>","<resolution>[100,100,100]</resolution>","<patches>[1,1,1]</patches>"])
# liddrivencavity3Dsmagorinskyperf_ups = modUPS( the_dir, \
#                                        "turb-lid-driven-cavity-3D-SMAGORINSKY.ups", \
#                                        ["<max_Timesteps> 50 </max_Timesteps>","<resolution>[100,100,100]</resolution>","<patches>[1,1,1]</patches>"])
# liddrivencavity3Dwaleperf_ups = modUPS( the_dir, \
#                                        "turb-lid-driven-cavity-3D-WALE.ups", \
#                                        ["<max_Timesteps> 50 </max_Timesteps>","<resolution>[100,100,100]</resolution>","<patches>[1,1,1]</patches>"])
# scalabilitytestperf_ups = modUPS( the_dir, \
#                                   "scalability-test.ups", \
#                                   ["<max_Timesteps> 1000 </max_Timesteps>"])                                       
# 
# turbulenceDir = the_dir + "/TurbulenceVerification"
# 
# decayIsotropicTurbulenceCSmag32_ups = modUPS( the_dir, \
#                                        "decay-isotropic-turbulence-csmag_32.ups", \
#                                        ["<max_Timesteps> 10 </max_Timesteps>","<outputTimestepInterval>1</outputTimestepInterval>",'<checkpoint cycle = "2" interval = "0.001"/>'])
#                                        
# decayIsotropicTurbulenceCSmag64_ups = modUPS( the_dir, \
#                                        "decay-isotropic-turbulence-csmag_64.ups", \
#                                        ["<max_Timesteps> 10 </max_Timesteps>","<outputTimestepInterval>1</outputTimestepInterval>",'<checkpoint cycle = "2" interval = "0.001"/>'])
#                                        
decayIsotropicTurbulenceVreman32_ups = modUPS( the_dir, \
                                       "warches-decay-isotropic-turbulence-vreman-32.ups", \
                                       ["<max_Timesteps> 10 </max_Timesteps>","<outputTimestepInterval>1</outputTimestepInterval>",'<checkpoint cycle = "4" interval = "0.001"/>'])
                                        
decayIsotropicTurbulenceVreman64_ups = modUPS( the_dir, \
                                       "warches-decay-isotropic-turbulence-vreman-64.ups", \
                                       ["<max_Timesteps> 10 </max_Timesteps>","<outputTimestepInterval>1</outputTimestepInterval>",'<checkpoint cycle = "4" interval = "0.001"/>'])
                                       
decayIsotropicTurbulenceWale32_ups = modUPS( the_dir, \
                                       "warches-decay-isotropic-turbulence-wale-32.ups", \
                                       ["<max_Timesteps> 10 </max_Timesteps>","<outputTimestepInterval>1</outputTimestepInterval>",'<checkpoint cycle = "4" interval = "0.001"/>'])
                                       
decayIsotropicTurbulenceWale64_ups = modUPS( the_dir, \
                                       "warches-decay-isotropic-turbulence-wale-64.ups", \
                                       ["<max_Timesteps> 10 </max_Timesteps>","<outputTimestepInterval>1</outputTimestepInterval>",'<checkpoint cycle = "4" interval = "0.001"/>'])

decayIsotropicTurbulenceDSmag32_ups = modUPS( the_dir, \
                                       "warches-decay-isotropic-turbulence-dsmag-32.ups", \
                                       ["<max_Timesteps> 10 </max_Timesteps>","<outputTimestepInterval>1</outputTimestepInterval>",'<checkpoint cycle = "4" interval = "0.001"/>'])
                                       
decayIsotropicTurbulenceDSmag64_ups = modUPS( the_dir, \
                                       "warches-decay-isotropic-turbulence-dsmag-64.ups", \
                                       ["<max_Timesteps> 10 </max_Timesteps>","<outputTimestepInterval>1</outputTimestepInterval>",'<checkpoint cycle = "4" interval = "0.001"/>'])                                       

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

UNUSED_TESTS = []

NIGHTLYTESTS = []


# Tests that are run during local regression testing
LOCALTESTS = [
  ("decay-isotropic-turbulence-dsmag32" , decayIsotropicTurbulenceDSmag32_ups,  8,  "All",  ["exactComparison"] ),
  ("decay-isotropic-turbulence-dsmag64" , decayIsotropicTurbulenceDSmag64_ups,  8,  "All",  ["exactComparison","no_restart"] ),
  ("decay-isotropic-turbulence-vreman32", decayIsotropicTurbulenceVreman32_ups, 8,  "All",  ["exactComparison"] ),
  ("decay-isotropic-turbulence-vreman64", decayIsotropicTurbulenceVreman64_ups, 8,  "All",  ["exactComparison","no_restart"] ),
  ("decay-isotropic-turbulence-wale32"  , decayIsotropicTurbulenceWale32_ups,   8,  "All",  ["exactComparison"] ),
  ("decay-isotropic-turbulence-wale64"  , decayIsotropicTurbulenceWale64_ups,   8,  "All",  ["exactComparison","no_restart"] )
]

DEBUGTESTS   =[]
#__________________________________
# The following list is parsed by the local RT script
# and allows the user to select the tests to run
#LIST: LOCALTESTS DEUGTESTS NIGHTLYTESTS
#__________________________________

def getNightlyTests() :
  return NIGHTLYTESTS

def getLocalTests() :
  return LOCALTESTS

# returns the list  
def getTestList(me) :
  if me == "LOCALTESTS":
    TESTS = LOCALTESTS
  elif me == "DEBUGTESTS":
    TESTS = DEGUGTESTS
  else:
    TESTS = NIGHTLYTESTS
  return TESTS
#__________________________________

if __name__ == "__main__":

  TESTS = getTestList( environ['WHICH_TESTS'] )

  result = runSusTests(argv, TESTS, "Warches")
  exit( result )

