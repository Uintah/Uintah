#!/usr/bin/env python3

from os                 import getenv,environ,unsetenv,rmdir,mkdir,path,system,chdir,stat,getcwd,pathsep,symlink,stat,access,getuid,W_OK
from os                 import makedirs
from time               import strftime,time,gmtime,asctime,localtime
from sys                import argv,exit,stdout
from modUPS             import modUPS
from subprocess         import getoutput
from subprocess         import PIPE, Popen
from helpers.utilities  import writeDividerLine, appendFile
import shutil
import socket
import resource
import platform
import re         # regular expressions

#______________________________________________________________________
# os.system()/system() encodes a shell exit code as (exit_code << 8).
# Each helper script below has its own exit-code convention, so the same
# raw number means different things depending on which one produced it.

# sus itself, killed for exceeding resource.setrlimit(RLIMIT_CPU, ...)
SUS_KILLED_SIGKILL = 137 * 256   # exit 137 = 128 + SIGKILL
SUS_KILLED_SIGTERM = 143 * 256   # exit 143 = 128 + SIGTERM

# performance_check
PERF_NO_TIMESTAMP_FILE = 5 * 256
PERF_FAILED            = 2 * 256

# mem_leak_check
MEM_MISSING_MALLOC_STATS = 5 * 256
MEM_FAILED_LEAK_TEST     = 1 * 256
MEM_FAILED_HIGHWATER     = 2 * 256

# compare_sus_runs
COMPUDA_INPUT_DIFFERS    = 10 * 256
COMPUDA_TOLERANCE_FAILED = 1 * 256
COMPUDA_ABORTED          = 5 * 256
COMPUDA_NO_DAT_FILES     = 65280   # -1, wrapped to 255

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

def parse_addtlpath( addtlpath, susdir ):
    addtlpath_clean = ""

    if addtlpath != None:

      for elem in addtlpath:
        tmp=re.split( r'=|\s', elem)                   # deliminter = " " or =

        if tmp[0].upper() =='ABSOLUTEPATH':
          addtlpath_clean = tmp[1] + pathsep  + addtlpath_clean

        elif tmp[0].upper() == 'RELATIVEPATH':
          p = path.normpath( path.join(susdir, tmp[1]) )
          addtlpath_clean = p + pathsep  + addtlpath_clean

        else:
          print( "\t*** WARNING syntax error specifiying ADDTL_PATH (%s). Valid options" % elem)
          print( "\t*** absolutePath=<path>  or  relativePath=<path>" )


    #print ("new path: %s  oldpath: %s susdir: %s" % (addtlpath_clean, addtlpath, susdir))
    return addtlpath_clean


#______________________________________________________________________

def runPreProcessCmd( preProcess_cmd, inputxml, logfile, log_msg):
    if preProcess_cmd != "":

      tmp  = preProcess_cmd.rsplit(' ')
      cmd0 = tmp[0]
      cmd  = shutil.which(cmd0)
      options = tmp[1:]
      options = ' '.join([str(elem) for elem in options])      # convert list to a string

      if cmd == None:

        msg = "\t*** ERROR: runSusTests: the preprocess cmd (%s %s %s) was not found\n" % (cmd0, options, inputxml)
        msg += "\t*** Path: %s\n" % environ['PATH']
        msg += "\t*** cwd: %s" % getcwd()

        appendFile( logfile, msg)

        print( msg )
        print( log_msg )
        return 1
      else:
        writeDividerLine( logfile )
        msg = "Preprocess Command: %s %s %s\n" %( cmd, options, inputxml)
        appendFile( logfile, msg )

        rc = system( "%s %s %s >> %s 2>&1" % (cmd, options, inputxml, logfile) )
        writeDividerLine( logfile )
        return 0

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
  from xml.etree.ElementTree import ParseError

  # these options are OK
  if startFrom == "checkpoint" or startFrom == "postProcessUda" or do_restart == 0:
    return True

  # load index.xml into tree
  ET = ElementTree()

  try:
    uintah = ET.parse(inputxml)
  except ParseError:
    print('    *** ERROR: The xml file {} is corrupt and cannot be parsed'.format(inputxml))
    return False


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
    print( "    *** ERROR: The xml file is not valid, (DataArchiver:outputInitTimestep) is not allowed for regression testing.")
    return False
  else:
    return True

#______________________________________________________________________
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

#______________________________________________________________________

# returns the path of either opt/dbg
def build_root():
    opt_dbg = path.normpath(path.join(getcwd(), "../"))
    return opt_dbg

#______________________________________________________________________
#  Helpers for runSusTests(), restructured to thread two small context
#  dicts instead of long parameter lists:
#
#    run_ctx  -- stable for the whole component run: application,
#                dbg_opt, max_parallelism, susdir, compare_root,
#                results_dir, solotest, has_gpu, do_plots, inputpath,
#                gold_standard, toolspath, sus.
#
#    test_ctx -- rebuilt each loop iteration (starts as parse_test_flags'
#                dict), accumulating testname/inputxml/inputsdir/is_valid
#                as the iteration proceeds, and -- inside runSusTest --
#                the per-run scratch values (susinput, messages, rc, ...).
#
#  Every function that adds fields to one of these dicts returns it
#  explicitly (test_ctx = some_helper(run_ctx, test_ctx)), even though
#  the dict is mutated in place, so the call site shows what changed.
#  "test" (the raw test tuple) and "callback" stay as their own
#  parameters -- bundling a function reference into a data context
#  would be confusing.
#______________________________________________________________________

def setup_paths_and_env( argv, application, addtlpath ):

  """Resolve every path runSusTests needs, probe for CUDA/GPU support,
  and set the environment variables sus and its helper scripts expect.
  Returns run_ctx (see module-level comment above)."""

  global helperspath
  inputpath     = getInputsDir()
  susdir        = path.normpath(path.join(getcwd(), argv[1]))
  gold_standard = path.normpath(path.join(getcwd(), argv[3]))
  helperspath   = "%s/%s" % (path.normpath(path.join(getcwd(), path.dirname(argv[0]))), "helpers")
  toolspath     = path.normpath(path.join(getcwd(), "tools"))

  global startpath
  startpath       = getcwd()

  addtlpath       = parse_addtlpath( addtlpath, susdir )

  dbg_opt         = argv[4]
  max_parallelism = float(argv[5])

  global git_revision
  myHash = getoutput("cd ../src; git log -1 --pretty=format:%H:" )
  myDate = getoutput("cd ../src; git log -1 --pretty=format:%cd:" )
  git_revision = "%s %s" % (myHash, myDate)

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
  environ['PATH']              =  helperspath + pathsep + toolspath + pathsep + addtlpath + pathsep + environ['PATH']
  environ['SCI_EXCEPTIONMODE'] = 'abort'
  environ['MPI_TYPE_MAX']      = '10000'

  solotest = ""
  if len(argv) == 7:
    solotest = argv[6]

  outputpath = startpath

  # If running Nightly RT, output logs in web dir
  # otherwise, save it in the build.  Also turn on plotting
  do_plots = 0
  if environ['LOCAL_OR_NIGHTLY_TEST'] == "nightly" :
    do_plots = 1

  run_ctx = {
      "application":     application,
      "inputpath":       inputpath,
      "susdir":          susdir,
      "gold_standard":   gold_standard,
      "toolspath":       toolspath,
      "dbg_opt":         dbg_opt,
      "max_parallelism": max_parallelism,
      "has_gpu":         has_gpu,
      "sus":             sus,
      "solotest":        solotest,
      "do_plots":        do_plots,
  }
  return run_ctx

#______________________________________________________________________

def bulletproof_and_prepare_dirs( run_ctx ):

  """Validate every required path, create (or reuse) the gold-standard
  component directory and the results directory, chdir into the
  latter, and print the startup banner.  Returns run_ctx with
  max_parallelism (possibly adjusted), compare_root, and results_dir
  added."""

  inputpath       = run_ctx["inputpath"]
  susdir          = run_ctx["susdir"]
  gold_standard   = run_ctx["gold_standard"]
  application     = run_ctx["application"]
  max_parallelism = run_ctx["max_parallelism"]
  solotest        = run_ctx["solotest"]
  dbg_opt         = run_ctx["dbg_opt"]

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

  run_ctx["max_parallelism"] = max_parallelism
  run_ctx["compare_root"]    = compare_root
  run_ctx["results_dir"]     = results_dir
  return run_ctx

#______________________________________________________________________

def parse_test_flags( test ):

  """Return this test's run configuration as test_ctx: the hard-coded
  defaults, overridden by any flags the test tuple specifies."""

  cfg = {
      "do_uda_comparisons": 1,
      "do_memory":          1,
      "do_restart":         1,
      "do_performance":     0,
      "do_debug":           1,
      "do_opt":             1,
      "do_gpu":             0,           # run test if gpu is supported
      "do_cuda":            1,           # test will run if this is a cuda build
      "abs_tolerance":      1e-9,        # defaults used in compare_uda
      "rel_tolerance":      1e-6,
      "mpirun_options":     "",
      "sus_options":        "",
      "compareUda_options": "",
      "startFrom":          "inputFile",
      "preProcess_cmd":     "",
      "create_gs0":         "no",           #create the gold standard
  }

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
        cfg["do_uda_comparisons"] = 0
      if flags[i] == "no_memoryTest":
        cfg["do_memory"] = 0
      if flags[i] == "no_restart":
        cfg["do_restart"] = 0
      if flags[i] == "gpu":
        cfg["do_gpu"] = 1
      if flags[i] == "no_dbg":
        cfg["do_debug"] = 0
      if flags[i] == "no_opt":
        cfg["do_opt"] = 0
      if flags[i] == "no_cuda":
        cfg["do_cuda"] = 0
      if flags[i] == "do_performance_test":
        cfg["do_restart"]         = 0
        cfg["do_debug"]           = 0
        cfg["do_uda_comparisons"] = 0
        cfg["do_memory"]          = 0
        cfg["do_performance"]     = 1
      if flags[i] == "doesTestRun":
        cfg["do_restart"]         = 1
        cfg["do_uda_comparisons"] = 0
        cfg["do_memory"]          = 0
        cfg["do_performance"]     = 0
      if flags[i] == "postProcessUda":
        cfg["startFrom"]          = "postProcessUda"
        cfg["do_restart"]         = 0
        cfg["do_memory"]          = 0
        cfg["do_performance"]     = 0
      if flags[i] == "startFromCheckpoint":
        cfg["startFrom"]          = "checkpoint"
      # parse the flags for
      #    abs_tolerance=<number>
      #    rel_tolerance=<number>
      #    sus_option=" "
      tmp = flags[i].split('=', 1)
      if tmp[0] == "preProcessCmd":
        cfg["preProcess_cmd"]    = tmp[1]
      if tmp[0] == "sus_options":
         cfg["sus_options"]      = tmp[1]
      if tmp[0] == "mpirun_options":
         cfg["mpirun_options"]   = tmp[1]
      if tmp[0] == "compareUda_options":
         cfg["compareUda_options"] = tmp[1]
      if tmp[0] == "abs_tolerance":
        cfg["abs_tolerance"]     = tmp[1]
      if tmp[0] == "rel_tolerance":
        cfg["rel_tolerance"]     = tmp[1]
      if flags[i] == "exactComparison":
        cfg["abs_tolerance"]     = 0.0
        cfg["rel_tolerance"]     = 0.0

  return cfg

#______________________________________________________________________

def should_skip_test( run_ctx, test_ctx ):

  """Return True (after printing the matching warning) if this test
  should be skipped for the current build configuration."""

  dbg_opt        = run_ctx["dbg_opt"]
  has_gpu        = run_ctx["has_gpu"]
  do_performance = test_ctx["do_performance"]
  do_debug       = test_ctx["do_debug"]
  do_opt         = test_ctx["do_opt"]
  do_gpu         = test_ctx["do_gpu"]
  do_cuda        = test_ctx["do_cuda"]

  if dbg_opt == "dbg" and do_performance == 1:
    print( "\nERROR: performance tests cannot be run with a debug build, skipping this test\n" )
    return True

  if do_debug == 0 and dbg_opt == "dbg":
    print( "\nWARNING: skipping this test (do_debug: %s, dbg_opt: %s)\n" % (do_debug, dbg_opt) )
    return True

  if do_opt == 0 and dbg_opt == "opt":
    print( "\nWARNING: skipping this test (do_opt: %s, dbg_opt: %s)\n" % (do_opt, dbg_opt) )
    return True

  if do_gpu == 1 and has_gpu == 0:
    print( "\nWARNING: skipping this test.  This machine is not configured to run GPU tests\n" )
    return True

  if has_gpu == 1 and do_cuda == 0:
    print( "\nWARNING: skipping this test.\n" )
    return True

  return False

#______________________________________________________________________

def apply_memory_and_gpu_adjustments( run_ctx, test_ctx ):

  """Apply the remaining per-test adjustments to do_memory, and set
  CUDA_VISIBLE_DEVICES if this test will use the GPU.  Returns test_ctx
  with do_memory (possibly adjusted)."""

  dbg_opt   = run_ctx["dbg_opt"]
  has_gpu   = run_ctx["has_gpu"]
  do_memory = test_ctx["do_memory"]
  do_gpu    = test_ctx["do_gpu"]

  if dbg_opt == "opt" :
    do_memory = 0

  if environ.get('SCI_MALLOC_ENABLED', 'no') != "yes" :
    do_memory = 0

  if has_gpu == 1:                                 #HACK:  TURN OFF ALL MEMORY CHECKS IS THE CODE WAS COMPILED WITH CUDA.  MALLOC STATS CAN'T BE TRUSTED
    print( "\nWARNING: skipping memory tests.  This build was compiled with CUDA.  MallocStats and CUDA has not been verified.\n" )
    do_memory = 0

  if do_gpu == 1 and has_gpu == 1:
    environ['CUDA_VISIBLE_DEVICES'] = "0"            # This will have to change for multiple GPU runs.  May need to make it a machine dependent environmenal variable

  test_ctx["do_memory"] = do_memory
  return test_ctx

#______________________________________________________________________

def prepare_test_run_dir( run_ctx, test_ctx, test, callback ):

  """Bulletproof and set up this test's results directory: gold-standard
  existence/creation, checkpoint validation, inputs-dir validation,
  the replace_gold_standard helper script, the pre-test callback, and
  the ups-file validity check.

  Returns test_ctx with create_gs0, inputsdir, and is_valid added/
  updated.  If is_valid is False, the caller should set failcode and
  skip to the next test -- the skip message has already been printed
  and logged."""

  testname        = test_ctx["testname"]
  compare_root    = run_ctx["compare_root"]
  application     = run_ctx["application"]
  startFrom       = test_ctx["startFrom"]
  results_dir     = run_ctx["results_dir"]
  inputpath       = run_ctx["inputpath"]
  inputxml        = test_ctx["inputxml"]
  susdir          = run_ctx["susdir"]
  dbg_opt         = run_ctx["dbg_opt"]
  max_parallelism = run_ctx["max_parallelism"]
  do_restart      = test_ctx["do_restart"]
  create_gs0      = test_ctx["create_gs0"]

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
      print( "ERROR: runSusTests: checkpoint uda (%s) does not exist" % here )
      print( "This file must exist when using 'startFromCheckpoint' or 'PostProcessUda' option" )
      print( "You can override the default path to the checkpoints directory with the environmental variable RT_CHECKPOINTS" )
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
  is_valid = True
  if isValid_inputFile( inputxml, startFrom, do_restart ) == False:
    print ("    Now skipping test %s " % testname)
    system("echo '  :%s:  The ups file is not valid, (DataArchiver:outputInitTimestep) is not allowed for regression testing.' >> %s/%s-short.log" % (testname,startpath,application))
    is_valid = False

  test_ctx["create_gs0"] = create_gs0
  test_ctx["inputsdir"]  = inputsdir
  test_ctx["is_valid"]   = is_valid
  return test_ctx

#______________________________________________________________________

def run_test_with_restart( run_ctx, test_ctx, test, callback ):

  """Run the test, and -- if it passed or merely failed comparison/memory
  checks (rc 0 or 2) -- run its restart variant too.  Returns True if
  this test (or its restart) failed."""

  susdir          = run_ctx["susdir"]
  compare_root    = run_ctx["compare_root"]
  application     = run_ctx["application"]
  dbg_opt         = run_ctx["dbg_opt"]
  max_parallelism = run_ctx["max_parallelism"]
  inputsdir       = test_ctx["inputsdir"]
  inputpath       = run_ctx["inputpath"]
  do_restart      = test_ctx["do_restart"]

  test_ctx["create_gs"] = "%s-%s" % (test_ctx["create_gs0"], test_ctx["startFrom"])

  rc = runSusTest(run_ctx, test_ctx, test)

  failed = False

  # Return Code (rc) of 2 means it failed comparison or memory test, so try to run restart
  if rc == 0 or rc == 2:
    # Prepare for restart test
    if rc == 2:
      failed = True

    mkdir("restart")
    chdir("restart")

    # call the callback function before running each test
    callback(test, susdir, inputsdir, compare_root, dbg_opt, max_parallelism);

    #__________________________________
    # Run restart test
    if do_restart == 1:
      symlink(inputpath, "inputs")

      test_ctx["startFrom"] = "restart"
      test_ctx["create_gs"] = "%s-%s" % (test_ctx["create_gs0"], test_ctx["startFrom"])

      rc = runSusTest(run_ctx, test_ctx, test)

      if rc > 0:
        failed = True

    chdir("..")
  elif rc == 1: # negative one means skipping -- not a failure
    failed = True
  chdir("..")

  return failed

#______________________________________________________________________

def finalize_component_run( run_ctx, solotest_found, ran_any_tests, failcode, comp_time0 ):

  """Wrap up after the test loop: fix permissions on the results dir,
  optionally mirror results to the web dir, and print the final
  pass/fail report.  Returns failcode (runSusTests' return value)."""

  results_dir = run_ctx["results_dir"]
  application = run_ctx["application"]
  dbg_opt     = run_ctx["dbg_opt"]
  solotest    = run_ctx["solotest"]

  #______________________________________________________________________
  # change the group and permissons on the results directory to a common group name
  chdir("..")
  common_group = environ.get('COMMON_GROUP')
  if common_group is not None:
    system("chgrp -R %s %s > /dev/null 2>&1" % (common_group, results_dir) )

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
# if a callback is given, it is executed before running each test and given
# all of the paramaters given to runSusTest
def runSusTests(argv, TESTS, application, addtlpath = None, callback = nullCallback):

  print("runSusTests: %s " % application)

  if len(argv) < 6 or len(argv) > 8 or not argv[4] in ["dbg", "opt", "unknown"] :
    print( "usage: %s <susdir> <inputsdir> <testdata_goldstandard> <dbg_opt> <max_parallelsim> < addtl_path> <test>" % argv[0] )
    print( "    where <test> is optional" )
    exit(1)

  run_ctx = setup_paths_and_env(argv, application, addtlpath)

  run_ctx = bulletproof_and_prepare_dirs(run_ctx)

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

    if run_ctx["solotest"] != "" and testname != run_ctx["solotest"]:
      continue

    if getTestOS(test) != platform.system().upper() and getTestOS(test) != "ALL":
      continue

    print( "__________________" )
    test_time0 = time()
    solotest_found = 1

    #__________________________________
    # defaults, overridden by this test's flags
    test_ctx = parse_test_flags(test)
    test_ctx["testname"] = testname
    test_ctx["inputxml"] = inputxml

    #Warnings
    if should_skip_test(run_ctx, test_ctx):
      continue

    test_ctx = apply_memory_and_gpu_adjustments(run_ctx, test_ctx)

    ran_any_tests = 1

    test_ctx = prepare_test_run_dir(run_ctx, test_ctx, test, callback)

    if not test_ctx["is_valid"]:
      failcode = 1
      continue

    #__________________________________
    # Run test and perform comparisons on the uda
    if run_test_with_restart(run_ctx, test_ctx, test, callback):
      failcode = 1

    # timer
    test_timer = time() - test_time0
    print( "Test Timer: %s" % strftime("%H:%M:%S",gmtime(test_timer)) )

    #__________________________________
    # If the test passed put an git "revision" stamp in the goldstandard
    # Only do this if the nightly RT cronjob is running
    if failcode == 0 and getenv('AUTO_UPDATE_GIT_STAMP') == "yes":
      print( "Updating the git revision file %s" %git_revision )
      git_file = "%s/%s/%s/git_revision" % (run_ctx["gold_standard"],application,testname)
      system( "echo 'This test last passed with Revision: %s'> %s" %(git_revision, git_file))

    #__________________________________
    # end of test loop

  failcode = finalize_component_run(run_ctx, solotest_found, ran_any_tests, failcode, comp_time0)
  return failcode

#______________________________________________________________________
#  Helpers for runSusTest() -- same (run_ctx, test_ctx) approach.  Every
#  value runSusTest computes (MPIHEAD, command, susinput, the messages,
#  rc/susSuccess/nTimeSteps, ...) is stored into test_ctx as it's
#  produced, so each sub-helper below takes just (run_ctx, test_ctx).
#______________________________________________________________________

def resolve_mpi_command( run_ctx, test_ctx ):

  """Set the CPU-time rlimit for this test, decide whether malloc_stats
  should be passed through to mpirun, and resolve the mpirun command
  line.  Returns test_ctx with MPIHEAD added."""

  dbg_opt        = run_ctx["dbg_opt"]
  do_memory_test = test_ctx["do_memory"]

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
    MALLOCSTATS = None
  else:
    MALLOCSTATS = "-x MALLOC_STATS"

  #__________________________________
  # Does mpirun command exist or has the environmental variable been set?
  try :
    MPIRUN = environ['MPIRUN']    # first try the environmental variable
  except :
    MPIRUN = shutil.which("mpirun")

  if MPIRUN is None:
    print( "ERROR:runSusTests.py ")
    print( "      mpirun command was not found and the environmental variable MPIRUN was not set." )
    print( "      You must either add mpirun to your path, or set the 'MPIRUN' environment variable." )
    exit (1)

  MPIHEAD="%s -n" % MPIRUN       #default

  # pass in environmental variables to mpirun
  if MALLOCSTATS is not None:
                                   # openmpi
    rc = system("%s -x TERM echo 'hello' > /dev/null 2>&1" % MPIRUN)
    if rc == 0:
      MPIHEAD="%s %s -n" % (MPIRUN, MALLOCSTATS)

                                   #  mvapich and mpich
    rc = system("%s -genvlist TERM echo 'hello' > /dev/null 2>&1" % MPIRUN)
    if rc == 0:
      MPIHEAD="%s -genvlist MALLOC_STATS -n" % MPIRUN

  test_ctx["MPIHEAD"] = MPIHEAD
  return test_ctx

#______________________________________________________________________

def maybe_strip_for_performance( test_ctx ):

  """If this is a performance test, rewrite the ups file with output
  and checkpointing disabled, and copy it back locally.  Returns
  test_ctx with inputxml (possibly updated)."""

  do_performance_test = test_ctx["do_performance"]
  inputxml            = test_ctx["inputxml"]

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

  test_ctx["inputxml"] = inputxml
  return test_ctx

#______________________________________________________________________

def build_susinput( run_ctx, test_ctx ):

  """Print the "Running test..." banner and build sus's command-line
  input argument for the given startFrom mode.  Returns test_ctx with
  susinput and restart_text added."""

  startFrom   = test_ctx["startFrom"]
  testname    = test_ctx["testname"]
  mpimsg      = test_ctx["mpimsg"]
  inputxml    = test_ctx["inputxml"]
  application = run_ctx["application"]

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

  test_ctx["susinput"]     = susinput
  test_ctx["restart_text"] = restart_text
  return test_ctx

#______________________________________________________________________

def setup_memory_test_env( test_ctx ):

  """If this test runs memory checks, turn on strict malloc tracking
  and VarLabel debugging.  Returns test_ctx with malloc_stats_file
  added (None if memory checks are off)."""

  do_memory_test = test_ctx["do_memory"]
  startFrom      = test_ctx["startFrom"]

  malloc_stats_file = None

  if do_memory_test == 1:
    environ['MALLOC_STRICT'] = "set"
    env = "%s,%s" % (environ['SCI_DEBUG'], "VarLabel:+") # append to the existing SCI_DEBUG
    environ['SCI_DEBUG'] = env

    if startFrom == "restart":
      malloc_stats_file = "restart_malloc_stats"
    else:
      malloc_stats_file = "malloc_stats"
    environ['MALLOC_STATS'] = malloc_stats_file

  test_ctx["malloc_stats_file"] = malloc_stats_file
  return test_ctx

#______________________________________________________________________

def build_messages( run_ctx, test_ctx ):
  """Build the "see this log for details" messages used when a test
  fails, as either web links (OUTPUT_HTML) or plain paths.  Returns
  test_ctx with sus_log_msg, compare_msg, memory_msg, perf_msg added."""

  testname    = test_ctx["testname"]
  dbg_opt     = run_ctx["dbg_opt"]
  application = run_ctx["application"]
  startFrom   = test_ctx["startFrom"]

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
    if startFrom == "restart":
      logpath   = logpath + "/restart"

    sus_log_msg = '\tSee %s/sus.log.txt for details' % (logpath)
    compare_msg = '\tSee %s/compare_sus_runs.log.txt for more comparison information.' % (logpath)
    memory_msg  = '\tSee %s/mem_leak_check.log.txt for more comparison information.' % (logpath)
    perf_msg    = '\tSee %s/performance_check.log.txt for more performance information.' % (logpath)

  test_ctx["sus_log_msg"] = sus_log_msg
  test_ctx["compare_msg"] = compare_msg
  test_ctx["memory_msg"]  = memory_msg
  test_ctx["perf_msg"]    = perf_msg
  return test_ctx

#______________________________________________________________________

def run_sus( run_ctx, test_ctx ):
  """Actually run sus, logging everything to sus.log.txt.  Detects a
  thrown C++ exception and whether sus reported a clean shutdown.
  Returns test_ctx with rc, susSuccess, nTimeSteps added."""

  command  = test_ctx["command"]
  susinput = test_ctx["susinput"]
  susdir   = run_ctx["susdir"]

  #__________________________________
  # actually run the test!
  short_cmd = command.replace(susdir+'/','')

  print( "Command Line: %s %s" % (short_cmd, susinput) )

  writeDividerLine( 'sus.log.txt' )
  system("env >> sus.log.txt")
  writeDividerLine( 'sus.log.txt' )

  rc = system(" %s %s >> sus.log.txt 2>&1" % (command, susinput))

  # Check to see if an exception was thrown.  (Use "grep -v 'cout'" to avoid false positive
  # when source code line was that prints the exception is changed.)
  # Did sus run to completion.
  exception = system("grep 'Caught exception' sus.log.txt | grep -qv cout");

  try:
    with open('sus.log.txt', 'r') as file:
      lines = file.read()
    susSuccess = re.findall("Sus: going down successfully", lines)
  except Exception:
    susSuccess = []

  if exception == 0:
    print( "\t*** An exception was thrown ***" )
    rc = -9

  # NOTE: must not reuse "rc" here -- it still holds sus's own exit code
  # (or the -9 exception sentinel above), which the checks below need.
  (nTimeSteps,err,grep_rc) = cmdline("grep -c 'Timestep [0-9]' sus.log.txt")

  test_ctx["rc"]         = rc
  test_ctx["susSuccess"] = susSuccess
  test_ctx["nTimeSteps"] = nTimeSteps
  return test_ctx

#______________________________________________________________________

def build_replace_msg( test_ctx ):
  """Build the "replace this test's gold standard" message, cd'ing up
  and back down temporarily to compute the right path when run from a
  restart subdirectory.  Returns test_ctx with replace_msg added."""

  startFrom = test_ctx["startFrom"]

  # determine path of replace_msg in 2 places to not have 2 different msgs.
  replace_msg = "\tTo replace this test's goldStandards run:\n\t    "

  if startFrom == "restart":
    chdir("..")
    replace_msg = "%s%s/replace_gold_standard" % (replace_msg, getcwd())
    chdir("restart")
  else:
    replace_msg = "%s%s/replace_gold_standard" % (replace_msg, getcwd())

  replace_msg = "%s\n\t\t\tor\n\t    %s/replace_all_GS\n" % (replace_msg,startpath)

  test_ctx["replace_msg"] = replace_msg
  return test_ctx

#______________________________________________________________________

def check_sus_completion( run_ctx, test_ctx ):
  """Check whether sus ran to completion.  Returns None if everything
  looks fine (go on to performance/memory/uda-comparison checks), or
  the return_code runSusTest should return immediately."""

  rc           = test_ctx["rc"]
  susSuccess   = test_ctx["susSuccess"]
  nTimeSteps   = test_ctx["nTimeSteps"]
  testname     = test_ctx["testname"]
  startFrom    = test_ctx["startFrom"]
  sus_log_msg  = test_ctx["sus_log_msg"]
  application  = run_ctx["application"]
  restart_text = test_ctx["restart_text"]

  if rc == SUS_KILLED_SIGKILL or rc == SUS_KILLED_SIGTERM :
    print( "\t*** Test %s exceeded maximum allowable run time\n" % (testname) )
    system("echo '  :%s: %s test exceeded maximum allowable run time' >> %s/%s-short.log" % (testname,restart_text,startpath,application))
    return 1

  elif rc != 0 or len(susSuccess) == 0  :
    print( "\t*** Test %s failed to run to completion, (code %d)" % (testname, rc) )

    if startFrom == "restart":
      print( "\t\tMake sure the problem makes checkpoints before finishing" )

    print( sus_log_msg )

    system("echo '  :%s: %s test did not run to completion' >> %s/%s-short.log" % (testname,restart_text,startpath,application))

    return 1

  elif int( nTimeSteps ) <= 1 :
    print( "\t*** ERROR Test %s did not run a sufficient number of timeteps.\n" % (testname) )
    system("echo '  :%s: %s test did not run a sufficient number of timeteps.' >> %s/%s-short.log" % (testname,restart_text,startpath,application))
    return 1

  return None

#______________________________________________________________________

def run_performance_test( run_ctx, test_ctx ):
  """Run performance_check and print the result.  Returns
  performance_RC."""

  testname     = test_ctx["testname"]
  do_plots     = test_ctx["do_plots"]
  ts_file      = test_ctx["ts_file"]
  compare_root = run_ctx["compare_root"]
  perf_msg     = test_ctx["perf_msg"]
  replace_msg  = test_ctx["replace_msg"]

  print( "\tPerforming performance test on %s" % (date()) )

  performance_RC = system("performance_check %s %s %s %s %s %s > performance_check.log.txt 2>&1" %
                         (testname, do_plots, ts_file, compare_root, helperspath, "sus.log.txt"))
  try:
    short_message_file = open("performance_shortmessage.txt", 'r+', 500)
    short_message = short_message_file.readline(500).rstrip()
  except Exception:
    short_message = ""

  if performance_RC == 0:
    print( "\tPerformance tests passed." )
    if short_message != "":
      print( "\t%s" % (short_message)     )

  elif performance_RC == PERF_NO_TIMESTAMP_FILE:
    print( "\t* Warning, no timestamp file created.  No performance test performed." )

  elif performance_RC == PERF_FAILED:
    print( "\t*** Warning, test %s failed performance test." % (testname) )
    if short_message != "":
      print( "\t%s" % (short_message) )

    print( perf_msg )
    print( "%s" % replace_msg )

  else:
    print( "\tPerformance tests passed. (Note: no previous performace stats)." )

  return performance_RC

#______________________________________________________________________

def run_memory_test( run_ctx, test_ctx ):
  """Run mem_leak_check and print the result.  Returns memory_RC."""

  testname          = test_ctx["testname"]
  do_plots          = test_ctx["do_plots"]
  malloc_stats_file = test_ctx["malloc_stats_file"]
  compare_root      = run_ctx["compare_root"]
  memory_msg        = test_ctx["memory_msg"]
  replace_msg       = test_ctx["replace_msg"]

  memory_RC = system("mem_leak_check %s %d %s %s %s %s> mem_leak_check.log.txt 2>&1" %
                    (testname, do_plots, malloc_stats_file, compare_root, ".", helperspath))

  try:
    short_message_file = open("highwater_shortmessage.txt", 'r+', 500)
    short_message = short_message_file.readline(500).rstrip()
  except Exception:
    short_message = ""

  if memory_RC == 0:
      print( "\tMemory leak tests passed." )
      if short_message != "":
        print( "\t%s" % (short_message) )

  elif memory_RC == MEM_MISSING_MALLOC_STATS:
      print( "\t*** ERROR, missing malloc_stats files.  No memory tests performed." )

  elif memory_RC == MEM_FAILED_LEAK_TEST:
      print( "\t*** ERROR, test %s failed memory leak test." % (testname) )
      print( memory_msg )
      # check that all VarLabels were deleted -- mem_leak_checkVarLabels always
      # exits 0
      system("mem_leak_checkVarLabels sus.log.txt >> mem_leak_check.log.txt 2>&1")

  elif memory_RC == MEM_FAILED_HIGHWATER:
      print( "\t*** ERROR, test %s failed memory highwater test." % (testname) )
      if short_message != "":
        print( "\t%s" % (short_message) )
      print( memory_msg )
      print( "%s" % replace_msg )

  else:
      print( "\tMemory leak tests passed. (Note: no previous memory usage stats)." )

  return memory_RC

#______________________________________________________________________

def run_uda_comparison( run_ctx, test_ctx ):
  """Run compare_sus_runs and print the result.  Returns compUda_RC."""

  testname           = test_ctx["testname"]
  dbg_opt            = run_ctx["dbg_opt"]
  compare_root       = run_ctx["compare_root"]
  susdir             = run_ctx["susdir"]
  abs_tol            = test_ctx["abs_tolerance"]
  rel_tol            = test_ctx["rel_tolerance"]
  create_gs          = test_ctx["create_gs"]
  compareUda_options = test_ctx["compareUda_options"]
  compare_msg        = test_ctx["compare_msg"]
  startFrom          = test_ctx["startFrom"]

  print( "\tComparing udas" )

  if dbg_opt == "dbg":
    environ['MALLOC_STATS'] = "compare_uda_malloc_stats"

  compUda_RC = system("compare_sus_runs %s %s %s %s %s %s %s \"%s\"> compare_sus_runs.log.txt 2>&1" %
                      (testname, getcwd(), compare_root, susdir,abs_tol, rel_tol, create_gs, compareUda_options))

  if compUda_RC != 0:
    if compUda_RC == COMPUDA_INPUT_DIFFERS:
      print( "\t*** Input file(s) differs from the goldstandard" )

    elif compUda_RC == COMPUDA_TOLERANCE_FAILED:
      print( "\t*** ERROR, test (%s) failed uda comparison, tolerances exceeded (%s)" % (testname, compUda_RC) )
      print(compare_msg)

    elif compUda_RC == COMPUDA_ABORTED:
      print( "\t*** ERROR: test (%s) uda comparison aborted (%s)" % (testname, compUda_RC) )
      print(compare_msg)

      if startFrom != "restart":

        (out,err,rc) = cmdline("tail -40 compare_sus_runs.log.txt | \
                                sed --silent /ERROR/,/ERROR/p |     \
                                sed /'^$'/d | sed /'may not be compared'/,+1d")   # clean out blank lines and cruft from the eror section
        print( "\n\n%s\n" % out )
        #print( "%s" % replace_msg )

    elif compUda_RC == COMPUDA_NO_DAT_FILES:
      print( "\tComparison tests passed.  (Note: No dat files to compare.)" )

    else:
      print( "\tComparison tests passed.  (Note: No previous gold standard.)" )

  else:
    print( "\tComparison tests passed." )

  return compUda_RC

#______________________________________________________________________

def aggregate_result( run_ctx, test_ctx, compUda_RC, performance_RC, memory_RC ):
  """Combine the three sub-test return codes into runSusTest's overall
  return_code, logging to the short.log and fixing its permissions if
  anything failed.  Returns return_code."""

  testname     = test_ctx["testname"]
  restart_text = test_ctx["restart_text"]
  application  = run_ctx["application"]

  #__________________________________
  # print error codes
  # if comparison, memory, performance tests fail, return here, so mem_leak tests can run
  return_code = 0

  if compUda_RC == COMPUDA_ABORTED or compUda_RC == COMPUDA_TOLERANCE_FAILED:
    system("echo '  :%s: \t%s test failed comparison tests' >> %s/%s-short.log" % (testname,restart_text,startpath,application))
    return_code = 2;

  if performance_RC == PERF_FAILED:
    system("echo '  :%s: \t%s test failed performance tests' >> %s/%s-short.log" % (testname,restart_text,startpath,application))
    return_code = 2;

  if memory_RC == MEM_FAILED_LEAK_TEST or memory_RC == MEM_FAILED_HIGHWATER or memory_RC == MEM_MISSING_MALLOC_STATS:
    system("echo '  :%s: \t%s test failed memory tests' >> %s/%s-short.log" % (testname,restart_text,startpath,application))
    return_code = 2;

  if return_code != 0:
    # as the file is only created if a certain test fails, change the permissions here as we are certain the file exists
    system("chmod gu+rw,a+r %s/%s-short.log > /dev/null 2>&1" % (startpath, application))

  return return_code

#______________________________________________________________________
# runSusTest()
# run_ctx/test_ctx hold everything this needs -- see the module-level
# comment above the runSusTests() helpers for what each contains.

def runSusTest(run_ctx, test_ctx, test):

  global startpath
  global helperspath
  global git_revision

  testname = test_ctx["testname"]

  np = float(getMPISize(test))
  if (np > run_ctx["max_parallelism"]):
    if np == 1.1:
      print( "Skipping test %s because it requires mpi and max_parallism < 1.1" % testname )
      return -1;

  do_plots = run_ctx["do_plots"]

  # turn off plotting option on restarts
  if test_ctx["startFrom"] == "restart":
    do_plots = 0
  test_ctx["do_plots"] = do_plots

  compUda_RC      = 0   # compare_uda return code
  performance_RC  = 0   # performance return code
  memory_RC       = 0   # memory return code

  test_ctx = resolve_mpi_command(run_ctx, test_ctx)

  test_ctx = maybe_strip_for_performance(test_ctx)

  #SVN_OPTIONS = "-svnStat -svnDiff"
  GIT_OPTIONS = "" # When debugging, if you don't want to spend time waiting for git, uncomment this line.

  test_ctx["command"] = "/usr/bin/time -p %s %s %s %s/sus %s %s " % (test_ctx["MPIHEAD"], int(np), test_ctx["mpirun_options"], run_ctx["susdir"], test_ctx["sus_options"], GIT_OPTIONS)
  test_ctx["mpimsg"]  = " (mpi %s proc)" % (int(np))

  time0 =time()  #timer

  test_ctx = build_susinput(run_ctx, test_ctx)

  test_ctx = setup_memory_test_env(test_ctx)

  test_ctx = build_messages(run_ctx, test_ctx)

  #__________________________________
  #  run preprocess command on ups file -- not needed (or possible: the
  #  ups file was never copied into restart/) when restarting from a
  #  checkpoint uda instead of a fresh ups file.
  if test_ctx["startFrom"] != "restart":
    rc = runPreProcessCmd( test_ctx["preProcess_cmd"], test_ctx["inputxml"], 'sus.log.txt', test_ctx["sus_log_msg"] )
    if rc==1:
      return 1;

  test_ctx = run_sus(run_ctx, test_ctx)

  test_ctx = build_replace_msg(test_ctx)

  #__________________________________
  #  Error checking
  early_rc = check_sus_completion(run_ctx, test_ctx)
  if early_rc is not None:
    return early_rc

  # Sus completed successfully - now run memory, compare_uda and performance tests
  # get the time from sus.log
  # /usr/bin/time outputs 3 lines, the one called 'real' is what we want
  # it is the third line from the bottom

  # save this file independent of performance tests being done
  print( "\tSuccessfully ran to completion" )

  if test_ctx["startFrom"] == "restart":
    test_ctx["ts_file"] = "restart_timestamp"
  else:
    test_ctx["ts_file"] = "timestamp"
  system("tail -n3 sus.log.txt > %s" % test_ctx["ts_file"])

  #__________________________________
  # performance test
  if test_ctx["do_performance"] == 1:
    performance_RC = run_performance_test(run_ctx, test_ctx)

  #__________________________________
  # Memory leak test
  if test_ctx["do_memory"] == 1:
    memory_RC = run_memory_test(run_ctx, test_ctx)

  #__________________________________
  # uda comparison
  if test_ctx["do_uda_comparisons"] == 1:
    compUda_RC = run_uda_comparison(run_ctx, test_ctx)

  return_code = aggregate_result(run_ctx, test_ctx, compUda_RC, performance_RC, memory_RC)
  return return_code
