#!/usr/bin/env python

from sys import argv,exit
from os import environ, system
from helpers.runSusTests import runSusTests, inputs_root, generatingGoldStandards
from helpers.modUPS import modUPS,modUPS2

from os import system

the_dir = generatingGoldStandards()

if the_dir == "" :
  the_dir = "%s/Examples" % inputs_root()
else :
  the_dir = the_dir + "/Examples"

# convert RMCRT:double -> RMCRT:float
system("cd %s ; ./RMCRT_doubleToFloat  RMCRT_bm1_1L.ups  RMCRT_FLT_bm1_1L.ups"   % the_dir )
system("cd %s ; ./RMCRT_doubleToFloat  RMCRT_ML.ups      RMCRT_FLT_ML.ups"      % the_dir )
system("cd %s ; ./RMCRT_doubleToFloat  RMCRT_bm1_DO.ups  RMCRT_FLT_bm1_DO.ups"  % the_dir )

# Modify base files
RMCRT_isoScat_LHC_ups = modUPS( the_dir, \
                               "RMCRT_isoScat.ups", \
                               ["<rayDirSampleAlgo>LatinHyperCube</rayDirSampleAlgo>"])

# Modify base files
RMCRT_1L_perf_GPU_ups = modUPS( the_dir, \
                               "RMCRT_1L_perf.ups", \
                               ["<resolution> [64,64,64]  </resolution>",
                                "<patches>    [2,2,2]     </patches>",
                                "<nDivQRays>  100         </nDivQRays>"
                               ]  )

RMCRT_DO_perf_GPU_ups = modUPS2( the_dir, \
                               "RMCRT_DO_perf.ups", \
                               ["/Uintah_specification/Grid/Level/Box[@label=0]/resolution :[32,32,32]",
                                "/Uintah_specification/Grid/Level/Box[@label=0]/patches    :[2,2,2]",
                                "/Uintah_specification/Grid/Level/Box[@label=1]/resolution :[64,64,64]",
                                "/Uintah_specification/Grid/Level/Box[@label=1]/patches    :[4,4,4]",
                                "Uintah_specification/RMCRT/nDivQRays                      : 100"
                               ] )

#______________________________________________________________________
#  Test syntax: ( "folder name", "input file", # processors, "OS",["flags1","flag2"])
#
#  OS:  Linux, Darwin, or ALL
#
#  flags:
#       gpu:                    - run test if machine is gpu enabled
#       no_uda_comparison:      - skip the uda comparisons
#       no_memoryTest:          - skip all memory checks
#       no_restart:             - skip the restart tests
#       no_dbg:                 - skip all debug compilation tests
#       no_opt:                 - skip all optimized compilation tests
#       no_cuda:                - skip test if this is a cuda enable build
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
#  1) The "folder name" must be the same as uda without the extension.
#  2) Performance_tests are not run on a debug build.
#______________________________________________________________________
NIGHTLYTESTS = [   ("poisson1",         "poisson1.ups",                1, "ALL"),
                   ("RMCRT_test_1L",    "RMCRT_bm1_1L.ups",            1, "ALL", ["exactComparison"]),
                   ("RMCRT_1L_bounded",  "RMCRT_bm1_1L_bounded.ups",   8, "ALL", ["exactComparison"]),
                   ("RMCRT_bm1_DO",     "RMCRT_bm1_DO.ups",            1, "ALL", ["exactComparison"]),
                   ("RMCRT_ML",         "RMCRT_ML.ups",                8, "ALL", ["exactComparison"]),
                   ("RMCRT_VR",         "RMCRT_VR.ups",                1, "ALL", ["abs_tolerance=1e-14","rel_tolerance=1e-11"]),
                   ("RMCRT_radiometer", "RMCRT_radiometer.ups",        8, "ALL", ["exactComparison"]),
                   ("RMCRT_isoScat",    "RMCRT_isoScat.ups",           1, "ALL", ["exactComparison"]),
                   ("RMCRT_isoScat_LHC", RMCRT_isoScat_LHC_ups,        1, "ALL", ["exactComparison"]),
                   ("RMCRT_1L_reflect", "RMCRT_1L_reflect.ups",        1, "ALL", ["exactComparison"]),
                   ("RMCRT_udaInit",    "RMCRT_udaInit.ups",           1, "ALL", ["exactComparison","no_restart"]),
                   ("RMCRT_1L_perf",    "RMCRT_1L_perf.ups",           1, "ALL", ["do_performance_test"]),
                   ("RMCRT_DO_perf",    "RMCRT_DO_perf.ups",           1, "ALL", ["do_performance_test"]),
               ]

# Tests that are run during local regression testing
LOCALTESTS   = [   ("RMCRT_test_1L",    "RMCRT_bm1_1L.ups",            1, "ALL", ["exactComparison"]),
                   ("RMCRT_1L_bounded",  "RMCRT_bm1_1L_bounded.ups",   8, "ALL", ["exactComparison"]),
                   ("RMCRT_bm1_DO",     "RMCRT_bm1_DO.ups",            1, "ALL", ["exactComparison"]),
                   ("RMCRT_ML",         "RMCRT_ML.ups",                8, "ALL", ["exactComparison"]),
                   ("RMCRT_VR",         "RMCRT_VR.ups",                1, "ALL", ["exactComparison"]),
                   ("RMCRT_radiometer", "RMCRT_radiometer.ups",        8, "ALL", ["exactComparison"]),
                   ("RMCRT_1L_reflect", "RMCRT_1L_reflect.ups",        1, "ALL", ["exactComparison"]),
                   ("RMCRT_isoScat",    "RMCRT_isoScat.ups",           1, "ALL", ["exactComparison"]),
                   ("RMCRT_udaInit",    "RMCRT_udaInit.ups",           1, "ALL", ["exactComparison","no_restart"])
               ]

FLOATTESTS    = [  ("RMCRT_FLT_test_1L", "RMCRT_FLT_bm1_1L.ups",     1,   "ALL", ["exactComparison"]),
                   ("RMCRT_FLT_ML",      "RMCRT_FLT_ML.ups",         8,   "ALL", ["exactComparison"]),
                   ("RMCRT_FLT_bm1_DO",  "RMCRT_FLT_bm1_DO.ups",     1,   "ALL", ["exactComparison"])
                 ]

THREADEDTESTS = [  ("RMCRT_test_1L_thread",           "RMCRT_bm1_1L.ups",          1,   "ALL", ["exactComparison", "sus_options=-nthreads 4"]),
                   ("RMCRT_1L_bounded_threaded_2proc", "RMCRT_bm1_1L_bounded.ups", 2,   "ALL", ["exactComparison", "sus_options=-nthreads 4"]),
                   ("RMCRT_bm1_DO_thread",            "RMCRT_bm1_DO.ups",          1,   "ALL", ["exactComparison", "sus_options=-nthreads 8"]),
                   ("RMCRT_bm1_DO_thread_2proc",      "RMCRT_bm1_DO.ups",          2,   "ALL", ["exactComparison", "sus_options=-nthreads 4"]),
                   ("RMCRT_ML_thread",                "RMCRT_ML.ups",              1,   "ALL", ["exactComparison", "sus_options=-nthreads 4"]),
                   ("RMCRT_ML_thread_2proc",          "RMCRT_ML.ups",              2,   "ALL", ["exactComparison", "sus_options=-nthreads 4"]),
                   ("RMCRT_+Domain_thread_2proc",     "RMCRT_+Domain.ups",         2,   "ALL", ["exactComparison", "sus_options=-nthreads 4"]),
                   ("RMCRT_+Domain_ML_thread_2proc",  "RMCRT_+Domain_ML.ups",      2,   "ALL", ["exactComparison", "sus_options=-nthreads 4"]),
                   ("RMCRT_+Domain_DO_thread_2proc",  "RMCRT_+Domain_DO.ups",      2,   "ALL", ["exactComparison", "sus_options=-nthreads 4"])

                 ]

GPUTESTS      = [
                   ("RMCRT_test_1L_GPU",     "RMCRT_bm1_1L.ups",            1, "Linux", ["gpu",  "exactComparison", "sus_options=-nthreads 4 -gpu"]),
                   ("RMCRT_ML_GPU",          "RMCRT_ML.ups",                1, "Linux", ["gpu",  "exactComparison", "sus_options=-nthreads 4 -gpu"]),
                   ("RMCRT_1L_reflect_GPU",  "RMCRT_1L_reflect.ups",        1, "Linux", ["gpu",  "exactComparison", "sus_options=-nthreads 4 -gpu"]),
                   ("RMCRT_bm1_DO_GPU",      "RMCRT_bm1_DO.ups",            1, "Linux", ["gpu",  "exactComparison", "sus_options=-nthreads 4 -gpu"]),
                   ("RMCRT_1L_perf_GPU",      RMCRT_1L_perf_GPU_ups,        1, "Linux", ["gpu",  "do_performance_test", "sus_options=-nthreads 2 -gpu"]),
                   ("RMCRT_DO_perf_GPU",      RMCRT_DO_perf_GPU_ups,        1, "Linux", ["gpu",  "do_performance_test", "sus_options=-nthreads 2 -gpu"])
               ]

DOMAINTESTS   =[   ("RMCRT_+Domain",         "RMCRT_+Domain.ups",        8, "ALL", ["exactComparison"]),
                   ("RMCRT_+Domain_ML",      "RMCRT_+Domain_ML.ups",     8, "ALL", ["exactComparison"]),
                   ("RMCRT_+Domain_DO",      "RMCRT_+Domain_DO.ups",     8, "ALL", ["exactComparison"])
              ]
              
POISSON3TESTS = [ #("poisson3_2L",         "poisson3_2L.ups",             2, "All", ["exactComparison"] ),       
                  #("poisson3_3L",         "poisson3_3L.ups",             2, "All", ["exactComparison"] ),       
                  ("poisson3_+Domain_1L", "poisson3_+Domain_1L.ups",     2, "All", ["exactComparison"] )        
                ]
DEBUGTESTS   =[]

#__________________________________
# The following list is parsed by the local RT script
# and allows the user to select the tests to run
#LIST: LOCALTESTS FLOATTESTS GPUTESTS DEBUGTESTS NIGHTLYTESTS THREADEDTESTS DOMAINTESTS Poisson3_Tests
#__________________________________

# returns the list
def getTestList(me) :
  if me == "LOCALTESTS":
    TESTS = LOCALTESTS + DOMAINTESTS + THREADEDTESTS + FLOATTESTS
  elif me == "FLOATTESTS":
    TESTS = FLOATTESTS
  elif me == "GPUTESTS":
    TESTS = GPUTESTS
  elif me == "DEBUGTESTS":
    TESTS = DEBUGTESTS
  elif me == "DOMAINTESTS":
    TESTS = DOMAINTESTS
  elif me == "NIGHTLYTESTS":
    TESTS = NIGHTLYTESTS + DOMAINTESTS + THREADEDTESTS + FLOATTESTS + GPUTESTS
  elif me == "THREADEDTESTS":
    TESTS = THREADEDTESTS
  elif me == "Poisson3_Tests":
    TESTS = POISSON3TESTS
  else:
    print "\nERROR:Examples.py  getTestList:  The test list (%s) does not exist!\n\n" % me
    exit(1)
  return TESTS

#__________________________________

if __name__ == "__main__":

  TESTS = getTestList( environ['WHICH_TESTS'] )

  result = runSusTests(argv, TESTS, "Examples")

  # cleanup modified files
  command = "/bin/rm -rf %s/tmp > /dev/null 2>&1 " % (the_dir)
  system( command )

  exit( result )
