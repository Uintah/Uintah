#!/usr/bin/env python3

from os         import getenv,environ,unsetenv,rmdir,mkdir,path,system,chdir,stat,getcwd,pathsep,symlink,stat,access,getuid,W_OK
from os         import makedirs
from time       import strftime,time,gmtime,asctime,localtime
from sys        import argv,exit,stdout
from modUPS     import modUPS
from subprocess   import getoutput
from subprocess import PIPE, Popen

import shutil
import socket
import resource
import re         # regular expressions

#______________________________________________________________________
# Assuming that running python with the '-u' arg doesn't fix the i/o buffering problem, this line
# can be added after print statements:
#
# stdout.flush() # Make sure that output (via 'tee' command (from calling script)) is actually printed...
#______________________________________________________________________

#print("000 %s %s" % (argv, len(argv) ))

# default value for the inputs path
if ( len(argv) > 1 ):
  inputpath = path.abspath( argv[2] )
else:
  inputpath = ""


def getTestName(test):
    return test[0]
    
def getUpsFile(test):
    return test[1]

def getMPISize(test):
    return test[2]

def getTestOS(test):
    return test[3].upper()

def setInputsDir( here ):
    global inputpath 
    inputpath = here

def getInputsDir():
    global inputpath
    return inputpath
    
def date ():
    return asctime(localtime(time()))

def getTestFlags (test):
    return test[-1]

def nullCallback (test, susdir, inputsdir, compare_root, dbg_opt, max_parallelism):
    pass

#______________________________________________________________________
#  returns a list of tests, with performance tests filtered out
def ignorePerformanceTests( TESTS ):
  
  myTests=[]
  for test in TESTS:
    
    if len(test) == 5:
      flags = getTestFlags( test )
      
      if not "do_performance_test" in str( flags ):
        myTests.append( test )
  return myTests
        
    
#______________________________________________________________________    
# Function used for checking the input files
# skip tests that contain 
#    <outputInitTimestep/> AND  outputTimestepInterval > 1
def isValid_inputFile( inputxml, startFrom, do_restart ):
  from xml.etree.ElementTree import ElementTree
  
  # these options are OK
  if startFrom == "checkpoint" or startFrom == "postProcessUda" or do_restart == 0:
    return True

  # load index.xml into tree
  ET     = ElementTree()
  uintah = ET.parse(inputxml)
  
  #  Note <outputInitTimestep/> + outputTimestepInterval > 1 the uda/index.xml != restartUda/index.xml
  #             ( initTS )              ( intrvl )
  da     = uintah.find( 'DataArchiver' ) 
  
  # find the timestepInterval
  intrvl = da.find( 'outputTimestepInterval' )
  
  if intrvl is None:
    intrvl = int(-9)
  else:
    intrvl = int( intrvl.text )
  
  # was outputInitTimestep set?
  initTS = da.find( 'outputInitTimestep' )
   
  if ( initTS != None and intrvl > 1 ):
    print('     isValid_inputFile %s'% inputxml)
    print( "    *** ERROR: The xml file is not valid, (DataArchiver:outputInitTimestep) is not allowed in regression testing.")
    return False
  else:
    return True
    
#__________________________________   
# Use this if you need to capture stdout and the command's return code.  It also prevents the output from becoming scrambled.
def cmdline(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        stderr=PIPE,
        shell=True,
        universal_newlines=True     # needed to 
    )
    out, err = process.communicate()
    return (out, err, process.returncode)
    
#__________________________________
# returns the path of either opt/dbg
def build_root():
    opt_dbg = path.normpath(path.join(getcwd(), "../"))
    return opt_dbg

#______________________________________________________________________
# if a callback is given, it is executed before running each test and given
# all of the paramaters given to runSusTest
def runSusTests(argv, TESTS, application, callback = nullCallback):

  print("runSusTests: %s " % application)

  if len(argv) < 6 or len(argv) > 7 or not argv[4] in ["dbg", "opt", "unknown"] :
    print( "usage: %s <susdir> <inputsdir> <testdata_goldstandard> <dbg_opt> <max_parallelsim> <test>" % argv[0] )
    print( "    where <test> is optional" )
    exit(1)
  #__________________________________
  # setup variables and paths
  global helperspath
  inputpath     = getInputsDir()
  susdir        = path.normpath(path.join(getcwd(), argv[1]))
  gold_standard = path.normpath(path.join(getcwd(), argv[3]))
  helperspath   = "%s/%s" % (path.normpath(path.join(getcwd(), path.dirname(argv[0]))), "helpers")
  toolspath     = path.normpath(path.join(getcwd(), "tools"))

  global startpath
  startpath       = getcwd()
  dbg_opt         = argv[4]
  max_parallelism = float(argv[5])

  global svn_revision
  svn_revision = getoutput("svn info ../src |grep Revision")
  svn_revision = svn_revision.split(" ")[1]

  #check sus for CUDA capabilities
  has_gpu  = 0
  print( "Running command to test if sus was compiled with CUDA and there is a GPU is active: " + susdir + "/sus -gpucheck" )
  sus = susdir + "/sus"

  (stdout,stderr,rc) = cmdline( "%s -gpucheck" % sus)

  if rc == 1:
    print( "sus was compiled with CUDA and a GPU is available" )
    has_gpu = 1

  #__________________________________
  # set environmental variables
  environ['PATH']              = "%s%s%s%s%s" % (helperspath, pathsep, toolspath, pathsep, environ['PATH'])
  environ['SCI_EXCEPTIONMODE'] = 'abort'
  environ['MPI_TYPE_MAX']      = '10000'

  solotest = ""
  if len(argv) == 7:
    solotest = argv[6]

  outputpath    = startpath
  
  # If running Nightly RT, output logs in web dir
  # otherwise, save it in the build.  Also turn on plotting
  do_plots = 0
  if environ['LOCAL_OR_NIGHTLY_TEST'] == "nightly" :
    do_plots = 1


  #__________________________________
  # bulletproofing
  if max_parallelism < 1:
    max_parallelism = 1;

  try:
    stat( inputpath )
  except Exception:
    print(" ERROR: runSusTests: the path to the inputs directory (%s) is not valid" % inputpath)
    exit(1)  
  try:
    chdir( helperspath )
  except Exception:
    print( " ERROR runSusTests: the path to the 'helpers' directory (%s) is not valid" % (helperspath) )
    exit(1)

  try:
    chdir( susdir )
    stat( "sus" )
  except Exception:
    print( " ERROR runSusTests: path to sus (%s/sus) is not valid" % (susdir) )
    exit(1)

  try:
    chdir( gold_standard )
  except Exception:
    print( " ERROR runSusTests: path to gold standards (%s) is not valid" % (gold_standard) )
    exit(1)
  compare_root = "%s/%s" % (gold_standard, application)

  try:
    chdir(compare_root)
  except Exception:
    # create the gold_standard component sub-directory
    chdir(gold_standard)
    
    statinfo = stat(gold_standard)
    file_uid = statinfo.st_uid
    my_uid   = getuid()

    # only create component directory if the user is the owner
    if access(gold_standard, W_OK) and my_uid == file_uid:
      print( " The directory %s does not exist in the gold standards %s" % ( application, gold_standard) )
      print( " Now creating it...." )    
      mkdir(application)
      system("chmod -R 775 %s" % application)

  results_dir = "%s/%s-results" % (startpath, application)
  chdir(startpath)

  try:
    mkdir(results_dir)
  except Exception:
    if solotest == "":
      print( "Remove %s before running this test\n" % results_dir )
      exit(1)

  chdir(results_dir)

  print( "" )
  if solotest == "":
    print( "Performing %s-%s tests." % (application, dbg_opt) )
  else:
    print( "Performing %s-%s test %s." % (application, dbg_opt, solotest) )
  print( "====================================" )
  print( "" )

  #______________________________________________________________________
  # Loop over tests
  ran_any_tests  = 0
  failcode       = 0
  solotest_found = 0
  nTestsFinished = 0
  comp_time0 = time()

  # clean up any old log files
  system("rm -rf %s/%s-short.log" % (startpath,application))
  system("rm -rf %s/%s.log" % (startpath,application))

  for test in TESTS:

    testname = getTestName(test)
    inputxml = path.basename( getUpsFile(test) )

    if solotest != "" and testname != solotest:
      continue

    if getTestOS(test) != environ['OS'].upper() and getTestOS(test) != "ALL":
      continue

    print( "__________________" )
    test_time0 = time()
    solotest_found = 1
    #__________________________________
    # defaults
    do_uda_comparisons = 1
    do_memory       = 1
    do_restart      = 1
    do_performance  = 0
    do_debug        = 1
    do_opt          = 1
    do_gpu          = 0           # run test if gpu is supported
    do_cuda         = 1           # test will run if this is a cuda build
    abs_tolerance   = 1e-9        # defaults used in compare_uda
    rel_tolerance   = 1e-6
    sus_options     = ""
    compareUda_options = ""
    startFrom       = "inputFile"
    create_gs0      = "no"           #create the gold standard
    

    environ['SCI_DEBUG'] = ''   # reset it for each test

    #__________________________________
    # override defaults if the flags has been specified
    if len(test) == 5:
      flags = getTestFlags(test)
      print( "User Flags:" )
      
      #  parse the user flags
      for i in range(len(flags)):
        print( i,flags[i] )

        if flags[i] == "no_uda_comparison":
          do_uda_comparisons = 0
        if flags[i] == "no_memoryTest":
          do_memory = 0
        if flags[i] == "no_restart":
          do_restart = 0
        if flags[i] == "gpu":
          do_gpu = 1
        if flags[i] == "no_dbg":
          do_debug = 0
        if flags[i] == "no_opt":
          do_opt = 0
        if flags[i] == "no_cuda":
          do_cuda = 0
        if flags[i] == "do_performance_test":
          do_restart         = 0
          do_debug           = 0
          do_uda_comparisons = 0
          do_memory          = 0
          do_performance     = 1
        if flags[i] == "doesTestRun":
          do_restart         = 1
          do_uda_comparisons = 0
          do_memory          = 0
          do_performance     = 0
        if flags[i] == "postProcessUda":
          startFrom          = "postProcessUda"
          do_restart         = 0
          do_memory          = 0
          do_performance     = 0
        if flags[i] == "startFromCheckpoint":
          startFrom          = "checkpoint"
        # parse the flags for
        #    abs_tolerance=<number>
        #    rel_tolerance=<number>
        #    sus_option=" "
        tmp = flags[i].rsplit('=')
        if tmp[0] == "sus_options":
           sus_options      = tmp[1]
        if tmp[0] == "compareUda_options":
           compareUda_options = tmp[1]
        if tmp[0] == "abs_tolerance":
          abs_tolerance     = tmp[1]
        if tmp[0] == "rel_tolerance":
          rel_tolerance     = tmp[1]
        if flags[i] == "exactComparison":
          abs_tolerance     = 0.0
          rel_tolerance     = 0.0


    #Warnings
    if dbg_opt == "dbg" and do_performance == 1:
      print( "\nERROR: performance tests cannot be run with a debug build, skipping this test\n" )
      continue

    if do_debug == 0 and dbg_opt == "dbg":
      print( "\nWARNING: skipping this test (do_debug: %s, dbg_opt: %s)\n" % (do_debug, dbg_opt) )
      continue

    if do_opt == 0 and dbg_opt == "opt":
      print( "\nWARNING: skipping this test (do_opt: %s, dbg_opt: %s)\n" % (do_opt, dbg_opt) )
      continue

    if do_gpu == 1 and has_gpu == 0:
      print( "\nWARNING: skipping this test.  This machine is not configured to run GPU tests\n" )
      continue

    if has_gpu == 1 and do_cuda == 0:
      print( "\nWARNING: skipping this test.\n" )
      continue

    if dbg_opt == "opt" :
      do_memory = 0

    if environ['SCI_MALLOC_ENABLED'] != "yes" :
      do_memory = 0

    if has_gpu == 1:                                 #HACK:  TURN OFF ALL MEMORY CHECKS IS THE CODE WAS COMPILED WITH CUDA.  MALLOC STATS CAN'T BE TRUSTED
      print( "\nWARNING: skipping memory tests.  This build was compiled with CUDA.  MallocStats and CUDA has not been verified.\n" )
      do_memory = 0

    if do_gpu == 1 and has_gpu == 1:
      environ['CUDA_VISIBLE_DEVICES'] = "0"            # This will have to change for multiple GPU runs.  May need to make it a machine dependent environmenal variable

    tests_to_do = [do_uda_comparisons, do_memory, do_performance]
    tolerances  = [abs_tolerance, rel_tolerance]
    varBucket   = [sus_options, do_plots, compareUda_options]

    ran_any_tests = 1

    #__________________________________
    # bulletproofing
    # Does gold standard exists?
    # If it doesn't then either throw an error (local RT) or generate it (Nightly RT).
    
    try:
      chdir(compare_root)
      chdir(testname)
    except Exception:
      if environ['LOCAL_OR_NIGHTLY_TEST'] == "local" :
        print( "ERROR: The gold standard for the (%s) test does not exist." % testname )
        print( "To generate it run: \n   make gold_standards" )
        exit(1)

      if environ['LOCAL_OR_NIGHTLY_TEST'] == "nightly" :
        print( "gold Standard being created for  (%s)" % testname )
        chdir(compare_root)
        mkdir(testname)
        create_gs0 = "yes"

    if startFrom == "checkpoint" or startFrom == "postProcessUda":
      try:
        here = "%s/CheckPoints/%s/%s/%s.uda.000/" %(startpath,application,testname,testname)
        chdir(here)
      except Exception:
        print( " ERROR: runSusTests: checkpoint uda (%s) does not exist" % here )
        print( "This file must exist when using 'startFromCheckpoint' or 'PostProcessUda' option" )
        exit(1)


    # need to set the inputs dir here, since it could be different per test
    inputsdir = "%s/%s" % (inputpath, application)

    try:
      chdir(inputsdir)
    except Exception:
      print( " ERROR: runSusTests: the path to the inputs directory (%s) is not valid" % (inputsdir) ) 
      exit(1)

    chdir(results_dir)

    try:
      mkdir(testname)
    except Exception:
      print( "Remove %s/%s before running this test" % (results_dir, testname) )
      exit(1)

    system("echo '%s/replace_gold_standard %s %s/%s-results %s' > %s/replace_gold_standard" % (helperspath, compare_root, startpath, application, testname, testname))
    system("chmod gu+rwx %s/replace_gold_standard" % testname)

    chdir(testname)


    # call the callback function before running each test
    list = callback(test, susdir, inputsdir, compare_root, dbg_opt, max_parallelism)

    system("cp %s/%s %s > /dev/null 2>&1" % (inputsdir, getUpsFile( test ), inputxml))
    symlink(inputpath, "inputs")
    
    #________________________________
    # is the input file valid
    if isValid_inputFile( inputxml, startFrom, do_restart ) == False:
      print ("    Now skipping test %s " % testname)
      continue
        
    #__________________________________
    # Run test and perform comparisons on the uda

    create_gs = "%s-%s" % (create_gs0, startFrom)
    
    rc = runSusTest(test, susdir, inputxml, compare_root, application, dbg_opt, max_parallelism, tests_to_do, tolerances, startFrom, varBucket, create_gs)
    system("rm inputs")
      
    # Return Code (rc) of 2 means it failed comparison or memory test, so try to run restart
    if rc == 0 or rc == 2:
      # Prepare for restart test
      if rc == 2:
        failcode = 1

      mkdir("restart")
      chdir("restart")

      # call the callback function before running each test
      callback(test, susdir, inputsdir, compare_root, dbg_opt, max_parallelism);

      #__________________________________
      # Run restart test
      if do_restart == 1:
        symlink(inputpath, "inputs")

        startFrom = "restart"
        create_gs = "%s-%s" % (create_gs0, startFrom)
        
        rc = runSusTest(test, susdir, inputxml, compare_root, application, dbg_opt, max_parallelism, tests_to_do, tolerances, startFrom, varBucket, create_gs)

        if rc > 0:
          failcode = 1
        system("rm inputs")

      chdir("..")
    elif rc == 1: # negative one means skipping -- not a failure
      failcode = 1
    chdir("..")

    # timer
    test_timer = time() - test_time0
    print( "Test Timer: %s" % strftime("%H:%M:%S",gmtime(test_timer)) )

    #__________________________________
    # If the test passed put an svn revision stamp in the goldstandard
    # Only do this if the nightly RT cronjob is running
    if failcode == 0 and getenv('AUTO_UPDATE_SVN_STAMP') == "yes":
      print( "Updating the svn revision file %s" %svn_revision )
      svn_file = "%s/%s/%s/svn_revision" % (gold_standard,application,testname)
      system( "echo 'This test last passed with Revision: %s'> %s" %(svn_revision, svn_file))
    #__________________________________
    # end of test loop

  #______________________________________________________________________
  # change the group and permissons on the results directory to a common group name
  chdir("..")
  try:
    common_group = "%s" % (environ['COMMON_GROUP'])
    system("chgrp -R %s %s > /dev/null 2>&1" % (common_group, results_dir) )
  except:
    pass

  system("chmod -R g+rwX %s > /dev/null 2>&1" % results_dir)


  #______________________________________________________________________
  #  copy component tests results to web page
  if  getenv( 'OUTPUT_HTML' ) == "yes":
    web_result_dir = "%s/%s/%s-results" % (environ['PUBLIC_HTML'], dbg_opt, application)
    
    if path.exists( web_result_dir ) == False:
      makedirs( web_result_dir )
    
    print( "__________________________________" )
    print( "\nNow copying %s results to: %s" % (application, web_result_dir))

    (stdout,stderr,rc) = cmdline("rsync -avp --delete --exclude  '*uda*' --exclude 'replace*' %s/ %s/ " % (results_dir, web_result_dir))

    if rc != 0:
      print( "There was a problem copying the component results.  Error:" );
      print(stderr)
    
  #__________________________________
  if solotest != "" and solotest_found == 0:
    print( "unknown test: %s" % solotest )
    system("rm -rf %s" % (results_dir))
    exit(1)

  #__________________________________
  # no tests ran
  if ran_any_tests == 0:
    print( "\nERROR: Zero regression tests ran to completion.  Hint: is the OS you're using appropriate for the component's tests?\n" )
    exit(3)

  #__________________________________
  # If the tests successfully ran and passed all tests

  if failcode == 0:
    if solotest != "":
      print( "" )
      print( "%s-%s test %s passed successfully!" % (application, dbg_opt, solotest) )
    else:
      print( "" )
      print( "All %s-%s tests passed successfully!" % (application, dbg_opt) )
  else:
    print( "" )
    print( "Some tests failed" )

  comp_timer = time() - comp_time0
  print( "Component Timer: %s" % strftime("%H:%M:%S",gmtime(comp_timer)) )
  return failcode

#______________________________________________________________________
# runSusTest()
# parameters are basically strings, except for tests_to_do which is a list of
# 3 ints stating whether to do comparison, memory, and performance tests
# in that order

def runSusTest(test, susdir, inputxml, compare_root, application, dbg_opt, max_parallelism, tests_to_do, tolerances, startFrom, varBucket, create_gs):
  global startpath
  global helperspath
  global svn_revision

  testname = getTestName(test)

  np = float(getMPISize(test))
  if (np > max_parallelism):
    if np == 1.1:
      print( "Skipping test %s because it requires mpi and max_parallism < 1.1" % testname )
      return -1;

  sus_options             = varBucket[0]
  do_plots                = varBucket[1]
  compareUda_options      = varBucket[2]
  do_uda_comparison_test  = tests_to_do[0]
  do_memory_test          = tests_to_do[1]
  do_performance_test     = tests_to_do[2]
  compUda_RC      = 0   # compare_uda return code
  performance_RC  = 0   # performance return code
  memory_RC       = 0   # memory return code

  # turn off plotting option on restarts
  if startFrom == "restart":
    do_plots = 0

  #__________________________________
  # define the maximum run time
  Giga = 2**30
  Kilo = 2**10
  Mega = 2**20
  #resource.setrlimit(resource.RLIMIT_AS, (90 * Mega,100*Mega) )  If we ever want to limit the memory

  if dbg_opt == "dbg":
    maxAllowRunTime = 30*60   # 30 minutes
  else:
    maxAllowRunTime = 15*60   # 15 minutes

  resource.setrlimit(resource.RLIMIT_CPU, (maxAllowRunTime,maxAllowRunTime) )

  #__________________________________
  #  turn on malloc_stats
  if not do_memory_test :
      unsetenv('MALLOC_STATS')

  if getenv('MALLOC_STATS') == None:
    MALLOCSTATS = ""
  else:
    MALLOCSTATS = "-x MALLOC_STATS"

  #__________________________________
  # Does mpirun command exist or has the environmental variable been set?
  try :
    MPIRUN = environ['MPIRUN']    # first try the environmental variable
  except :
    try:
      MPIRUN = shutil.which("mpirun")
    except:
      print( "ERROR:runSusTests.py ")
      print( "      mpirun command was not found and the environmental variable MPIRUN was not set." )
      print( "      You must either add mpirun to your path, or set the 'MPIRUN' environment variable." )
      exit (1)

  MPIHEAD="%s -n" % MPIRUN       #default
  
  # pass in environmental variables to mpirun
  if environ['OS'] == "Linux":
    MPIHEAD="%s %s -n" % (MPIRUN, MALLOCSTATS)

                                   # openmpi
  rc = system("%s -x TERM echo 'hello' > /dev/null 2>&1" % MPIRUN)
  if rc == 0:
    MPIHEAD="%s %s -n" % (MPIRUN, MALLOCSTATS)

                                   #  mvapich
  rc = system("%s -genvlist TERM echo 'hello' > /dev/null 2>&1" % MPIRUN)
  if rc == 0:
    MPIHEAD="%s -genvlist MALLOC_STATS -n" % MPIRUN


  # if running performance tests, strip the output and checkpoints portions
  # use the the input file local
  if do_performance_test == 1:
    localPath = "./"
    inputxml = modUPS(localPath, inputxml,["<outputInterval>0</outputInterval>",
                                    "<outputTimestepInterval>0</outputTimestepInterval>",
                                    '<checkpoint cycle="0" interval="0"/>'])

    # create a file in tmp/filename, copy it back
    system("cp %s ." % inputxml)
    inputxml = path.basename(inputxml)


  SVN_OPTIONS = "-svnStat -svnDiff"
  #SVN_OPTIONS = "" # When debugging, if you don't want to spend time waiting for SVN, uncomment this line.

  command = "/usr/bin/time -p %s %s %s/sus %s %s " % (MPIHEAD, int(np), susdir, sus_options, SVN_OPTIONS)
  mpimsg = " (mpi %s proc)" % (int(np))

  time0 =time()  #timer

  #__________________________________
  # setup input for sus
  if startFrom == "restart":
    print( "Running restart test  ---%s--- %s at %s" % (testname, mpimsg, strftime( "%I:%M:%S")) )
    susinput     = "-restart ../*.uda.000 -t 0 -copy"
    restart_text = " (restart)"

  if startFrom == "inputFile":
    print( "Running test  ---%s--- %s at %s" % (testname, mpimsg, strftime( "%I:%M:%S")) )
    susinput     = "%s" % (inputxml)
    restart_text = " "

  if startFrom == "checkpoint":
    print( "Running test from checkpoint ---%s--- %s at %s" % (testname, mpimsg, strftime( "%I:%M:%S")) )
    susinput     = "-restart %s/CheckPoints/%s/%s/*.uda.000" %  (startpath,application,testname)
    restart_text = " "

  if startFrom == "postProcessUda":
    print( "Running test from checkpoint ---%s--- %s at %s" % (testname, mpimsg, strftime( "%I:%M:%S")) )
    susinput     = "-postProcessUda %s/CheckPoints/%s/%s/*.uda.000" %  (startpath,application,testname)
    restart_text = " "

  if do_memory_test == 1:
    environ['MALLOC_STRICT'] = "set"
    env = "%s,%s" % (environ['SCI_DEBUG'], "VarLabel:+") # append to the existing SCI_DEBUG
    environ['SCI_DEBUG'] = env

    if startFrom == "restart":
      malloc_stats_file = "restart_malloc_stats"
    else:
      malloc_stats_file = "malloc_stats"
    environ['MALLOC_STATS'] = malloc_stats_file

  #__________________________________
  #  define failure messages
  if getenv('OUTPUT_HTML') == "yes":
  
    logpath     =  "%s/%s/%s-results/%s" % (environ['RT_URL'],  dbg_opt, application, testname )
    if startFrom == "restart":
      logpath   =  logpath + "/restart"
    
    sus_log_msg = '\t<A href=\"%s/sus.log.txt\">See sus.log</a> for details' % (logpath)
    compare_msg = '\t<A href=\"%s/compare_sus_runs.log.txt\">See compare_sus_runs.log</A> for more comparison information.' % (logpath)
    memory_msg  = '\t<A href=\"%s/mem_leak_check.log.txt\">See mem_leak_check.log</a> for more comparison information.' % (logpath)
    perf_msg    = '\t<A href=\"%s/performance_check.log.txt\">See performance_check.log</a> for more comparison information.' % (logpath)
  else:
    logpath     = "%s/%s-results/%s"  %  (startpath,application,testname)
    sus_log_msg = '\tSee %s/sus.log.txt for details' % (logpath)
    compare_msg = '\tSee %s/compare_sus_runs.log.txt for more comparison information.' % (logpath)
    memory_msg  = '\tSee %s/mem_leak_check.log.txt for more comparison information.' % (logpath)
    perf_msg    = '\tSee %s/performance_check.log.txt for more performance information.' % (logpath)
  
  #__________________________________
  # actually run the test!
  short_cmd = command.replace(susdir+'/','')

  print( "Command Line: %s %s" % (short_cmd, susinput) )
  rc = system("env > sus.log.txt; %s %s >> sus.log.txt 2>&1" % (command, susinput))

  # Check to see if an exception was thrown.  (Use "grep -v 'cout'" to avoid false positive
  # when source code line was that prints the exception is changed.)
  # Did sus run to completion.
  exception = system("grep -q 'Caught exception' sus.log.txt | grep -v cout");

  try:
    file = open('sus.log.txt', 'r')
    lines = file.read()
    susSuccess = re.findall("Sus: going down successfully", lines)
  except Exception:
    pass

  if exception == 0:
    print( "\t*** An exception was thrown ***" )
    rc = -9

  (nTimeSteps,err,rc) = cmdline("grep -c 'Timestep [0-9]' sus.log.txt")
  
  # determine path of replace_msg in 2 places to not have 2 different msgs.
  replace_msg = "\tTo replace this test's goldStandards run:\n\t    "

  if startFrom == "restart":
    chdir("..")
    replace_msg = "%s%s/replace_gold_standard" % (replace_msg, getcwd())
    chdir("restart")
  else:
    replace_msg = "%s%s/replace_gold_standard" % (replace_msg, getcwd())

  replace_msg = "%s\n\t\t\tor\n\t    %s/replace_all_GS\n" % (replace_msg,startpath)

  #__________________________________
  #  Error checking
  return_code = 0
  if rc == 35072 or rc == 36608 :
    print( "\t*** Test %s exceeded maximum allowable run time\n" % (testname) )
    system("echo '  :%s: %s test exceeded maximum allowable run time' >> %s/%s-short.log" % (testname,restart_text,startpath,application))
    return_code = 1
    return return_code

  elif rc != 0 or len(susSuccess) == 0  :
    print( "\t*** Test %s failed to run to completion, (code %d)" % (testname, rc) )

    if startFrom == "restart":
      print( "\t\tMake sure the problem makes checkpoints before finishing" )

    print( sus_log_msg )
    
    system("echo '  :%s: %s test did not run to completion' >> %s/%s-short.log" % (testname,restart_text,startpath,application))
    
    return_code = 1
    return return_code
    
  elif int( nTimeSteps ) <= 1 :         
    print( "\t*** ERROR Test %s did not run a sufficient number of timeteps.\n" % (testname) )
    system("echo '  :%s: %s test did not run a sufficient number of timeteps.' >> %s/%s-short.log" % (testname,restart_text,startpath,application))
    return_code = 1
    return return_code
  
  else:
    # Sus completed successfully - now run memory, compare_uda and performance tests
    # get the time from sus.log
    # /usr/bin/time outputs 3 lines, the one called 'real' is what we want
    # it is the third line from the bottom

    # save this file independent of performance tests being done
    print( "\tSuccessfully ran to completion" )

    if startFrom == "restart":
      ts_file = "restart_timestamp"
    else:
      ts_file = "timestamp"
    system("tail -n3 sus.log.txt > %s" % ts_file)

    #__________________________________
    # performance test
    if do_performance_test == 1:
      print( "\tPerforming performance test on %s" % (date()) )

      performance_RC = system("performance_check %s %s %s %s %s %s > performance_check.log.txt 2>&1" %
                             (testname, do_plots, ts_file, compare_root, helperspath, "sus.log.txt"))
      try:
        short_message_file = open("performance_shortmessage.txt", 'r+', 500)
        short_message = rstrip(short_message_file.readline(500))
      except Exception:
        short_message = ""

      if performance_RC == 0:
        print( "\tPerformance tests passed." )
        if short_message != "":
          print( "\t%s" % (short_message)     )
      
      elif performance_RC == 5 * 256:
        print( "\t* Warning, no timestamp file created.  No performance test performed." )
      
      elif performance_RC == 2*256:
        print( "\t*** Warning, test %s failed performance test." % (testname) )
        if short_message != "":
          print( "\t%s" % (short_message) )

        print( perf_msg )
        print( "%s" % replace_msg )
      
      else:
        print( "\tPerformance tests passed. (Note: no previous performace stats)." )

    #__________________________________
    # Memory leak test
    if do_memory_test == 1:

      memory_RC = system("mem_leak_check %s %d %s %s %s %s> mem_leak_check.log.txt 2>&1" %
                        (testname, do_plots, malloc_stats_file, compare_root, ".", helperspath))

      try:
        short_message_file = open("highwater_shortmessage.txt", 'r+', 500)
        short_message = rstrip(short_message_file.readline(500))
      except Exception:
        short_message = ""

      if memory_RC == 0:
          print( "\tMemory leak tests passed." )
          if short_message != "":
            print( "\t%s" % (short_message) )
            
      elif memory_RC == 5 * 256:
          print( "\t*** ERROR, missing malloc_stats files.  No memory tests performed." )
      
      elif memory_RC == 256:
          print( "\t*** ERROR, test %s failed memory leak test." % (testname) )
          print( memory_msg )
          # check that all VarLabels were deleted
          rc = system("mem_leak_checkVarLabels sus.log.txt >> mem_leak_check.log.txt 2>&1")
      
      elif memory_RC == 2*256:
          print( "\t*** ERROR, test %s failed memory highwater test." % (testname) )
          if short_message != "":
            print( "\t%s" % (short_message) )
          print( memory_msg )
          print( "%s" % replace_msg )
      
      else:
          print( "\tMemory leak tests passed. (Note: no previous memory usage stats)." )
          
    #__________________________________
    # uda comparison
    if do_uda_comparison_test == 1:
      print( "\tComparing udas" )

      if dbg_opt == "dbg":
        environ['MALLOC_STATS'] = "compare_uda_malloc_stats"

      abs_tol= tolerances[0]
      rel_tol= tolerances[1]

      compUda_RC = system("compare_sus_runs %s %s %s %s %s %s %s \"%s\"> compare_sus_runs.log.txt 2>&1" % 
                          (testname, getcwd(), compare_root, susdir,abs_tol, rel_tol, create_gs, compareUda_options))
      
      if compUda_RC != 0:
        if compUda_RC == 10 * 256:
          print( "\t*** Input file(s) differs from the goldstandard" )

        elif compUda_RC == 1 * 256:
          print( "\t*** ERROR, test (%s) failed uda comparison, tolerances exceeded (%s)" % (testname, compUda_RC) ) 
          print(compare_msg)
           
        elif compUda_RC == 5*256:
          print( "\t*** ERROR: test (%s) uda comparison aborted (%s)" % (testname, compUda_RC) )
          print(compare_msg)
        
          if startFrom != "restart":
          
            (out,err,rc) = cmdline("tail -40 compare_sus_runs.log.txt | \
                                    sed --silent /ERROR/,/ERROR/p |     \
                                    sed /'^$'/d | sed /'may not be compared'/,+1d")   # clean out blank lines and cruft from the eror section
            print( "\n\n%s\n" % out )
            #print( "%s" % replace_msg )

        elif compUda_RC == 65280: # (-1 return code)
          print( "\tComparison tests passed.  (Note: No dat files to compare.)" )

        else:
          print( "\tComparison tests passed.  (Note: No previous gold standard.)" )

      else:
        print( "\tComparison tests passed." )


    #__________________________________
    # print error codes
    # if comparison, memory, performance tests fail, return here, so mem_leak tests can run
    if compUda_RC == 5*256 or compUda_RC == 1*256:
      system("echo '  :%s: \t%s test failed comparison tests' >> %s/%s-short.log" % (testname,restart_text,startpath,application))
      return_code = 2;

    if performance_RC == 2*256:
      system("echo '  :%s: \t%s test failed performance tests' >> %s/%s-short.log" % (testname,restart_text,startpath,application))
      return_code = 2;

    if memory_RC == 1*256 or memory_RC == 2*256 or memory_RC == 5*256:
      system("echo '  :%s: \t%s test failed memory tests' >> %s/%s-short.log" % (testname,restart_text,startpath,application))
      return_code = 2;

    if return_code != 0:
      # as the file is only created if a certain test fails, change the permissions here as we are certain the file exists
      system("chmod gu+rw,a+r %s/%s-short.log > /dev/null 2>&1" % (startpath, application))

  return return_code
