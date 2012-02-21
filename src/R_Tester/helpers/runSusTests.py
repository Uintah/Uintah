#!/usr/bin/python

from os import environ,unsetenv,rmdir,mkdir,path,system,chdir,stat,getcwd,pathsep,symlink
from time import strftime,time,gmtime,asctime,localtime
from sys import argv,exit,stdout
from string import upper,rstrip,rsplit
from modUPS import modUPS
from commands import getoutput

#______________________________________________________________________
# Assuming that running python with the '-u' arg doesn't fix the i/o buffering problem, this line
# can be added after print statements:
#
# stdout.flush() # Make sure that output (via 'tee' command (from calling script)) is actually printed...
#______________________________________________________________________

def nameoftest (test):
    return test[0]
def input (test):
    return test[1]
def num_processes (test):
    return test[2]
def testOS(test):
    return upper(test[3])
def inputs_root ():
    return argv[2]
def date ():
    return asctime(localtime(time()))
def userFlags (test):
    return test[-1]
def nullCallback (test, susdir, inputsdir, compare_root, dbg_opt, max_parallelism):
    pass

#______________________________________________________________________
# Used by toplevel/generateGoldStandards.py

inputs_dir = ""

def setGeneratingGoldStandards( inputs ) :
    global inputs_dir
    inputs_dir = inputs

def generatingGoldStandards() :
    global inputs_dir
    return inputs_dir
    
#______________________________________________________________________
# if a callback is given, it is executed before running each test and given
# all of the paramaters given to runSusTest
def runSusTests(argv, TESTS, ALGO, callback = nullCallback):
 
  if len(argv) < 6 or len(argv) > 7 or not argv[4] in ["dbg", "opt", "unknown"] :
    print "usage: %s <susdir> <inputsdir> <testdata_goldstandard> <dbg_opt> " \
             "<max_parallelsim> <test>" % argv[0]
    print "    where <test> is optional"
    exit(1)
  #__________________________________
  # setup variables and paths
  susdir        = path.normpath(path.join(getcwd(), argv[1]))
  gold_standard = path.normpath(path.join(getcwd(), argv[3]))
  helperspath   = "%s/%s" % (path.normpath(path.join(getcwd(), path.dirname(argv[0]))), "helpers")
  toolspath     = path.normpath(path.join(getcwd(), "tools"))
  inputpath     = path.normpath(path.join(getcwd(), inputs_root()))
  
  global startpath
  startpath     = getcwd()
  
  dbg_opt         = argv[4]
  max_parallelism = float(argv[5])
  svn_revision    = getoutput("svn info ../src |grep Revision")
  
  #__________________________________
  # set environmental variables
  environ['PATH']              = "%s%s%s%s%s" % (helperspath, pathsep, toolspath, pathsep, environ['PATH'])
  environ['SCI_SIGNALMODE']    = 'exit'
  environ['SCI_EXCEPTIONMODE'] = 'abort'
  environ['MPI_TYPE_MAX']      = '10000'
  environ['outputlinks']       = "0"  #display html links on output

  solotest = ""
  if len(argv) == 7:
    solotest = argv[6]
  
  
  outputpath = startpath
  weboutputpath = startpath
  # If running Nightly RT, output logs in web dir
  # otherwise, save it in the build
  if environ['LOCAL_OR_NIGHTLY_TEST'] == "nightly" :
    try:
      outputpath    = "%s-%s" % (environ['HTMLLOG'], dbg_opt)
      weboutputpath = "%s-%s" % (environ['WEBLOG'],  dbg_opt)
    except Exception:
      pass
      
    try:
      # make outputpath/dbg or opt dirs
      environ['outputlinks'] ="1"
      mkdir(outputpath)
      system("chmod -R 775 %s" % outputpath)
    except Exception:
      pass

  #__________________________________
  # bulletproofing
  if max_parallelism < 1:
    max_parallelism = 1;
  
  try:
    chdir(helperspath)
  except Exception:
    print "%s does not exist" % (helperspath)
    print "'helpers' directory could not be found"
    exit(1)

  try:
    chdir(susdir)
    stat("sus")
  except Exception:
    print "%s/sus does not exist" % (susdir)
    print "Please give a valid <susdir> argument"
    exit(1)

  try:
    chdir(gold_standard)
  except Exception:
    print "%s does not exist" % (gold_standard)
    print "Please give a valid <testdata_goldstandard> argument"
    exit(1)
  compare_root = "%s/%s" % (gold_standard, ALGO)
  
  try:
    chdir(compare_root)
  except Exception:
    # create the gold_standard algo subdir
    chdir(gold_standard)
    mkdir(ALGO)
    system("chmod -R 775 %s" % ALGO)
  
  resultsdir = "%s/%s-results" % (startpath, ALGO)
  chdir(startpath)
  
  try:
    mkdir(resultsdir)
  except Exception:
    if solotest == "":
      print "Remove %s before running this test\n" % resultsdir
      exit(1)

  chdir(resultsdir)

  if outputpath != startpath and path.exists("%s/%s-results" % (outputpath, ALGO)) != 1:
    mkdir("%s/%s-results" % (outputpath, ALGO))

  print ""
  if solotest == "":
    print "Performing %s-%s tests." % (ALGO, dbg_opt)
  else:
    print "Performing %s-%s test %s." % (ALGO, dbg_opt, solotest)
  print "===================================="
  print ""

  #______________________________________________________________________
  # Loop over tests 
  ran_any_tests  = 0
  failcode       = 0
  solotest_found = 0
  comp_time0 = time()
  
  # clean up any old log files
  system("rm -rf %s/%s-short.log" % (startpath,ALGO))
  system("rm -rf %s/%s.log" % (startpath,ALGO))
  
  for test in TESTS:

    testname = nameoftest(test)
    
    if solotest != "" and testname != solotest:
      continue
    
    if testOS(test) != upper(environ['OS']) and testOS(test) != "ALL":
      continue
      
    print "__________________"
    test_time0 = time() 
    solotest_found = 1
    #__________________________________
    # defaults
    do_uda_comparisons = 1
    do_memory      = 1
    do_restart     = 1
    do_performance = 0
    do_debug = 1
    do_opt   = 1
    abs_tolerance = 1e-9   # defaults used in compare_uda
    rel_tolerance = 1e-6    
    startFrom = "inputFile"
    
    #__________________________________
    # override defaults if the flags has been specified
    if len(test) == 5:
      flags = userFlags(test)
      print "User Flags:"
      
      #  parse the user flags
      for i in range(len(flags)):
        print i,flags[i]
        
        if flags[i] == "no_uda_comparison":
          do_uda_comparisons = 0
        if flags[i] == "no_memoryTest":
          do_memory = 0
        if flags[i] == "no_restart":
          do_restart = 0
        if flags[i] == "no_dbg":
          do_debug = 0
        if flags[i] == "no_opt":
          do_opt = 0
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
        if flags[i] == "startFromCheckpoint":
          startFrom          = "checkpoint"
        # parse the flags for 
        #    abs_tolerance=<number>
        #    rel_tolerance=<number>  
        tmp = flags[i].rsplit('=')
        if tmp[0] == "abs_tolerance":
          abs_tolerance = tmp[1]
        if tmp[0] == "rel_tolerance":
          rel_tolerance = tmp[1]
        if flags[i] == "exactComparison":
          abs_tolerance = 0.0
          rel_tolerance = 0.0
    
    #Warnings
    if dbg_opt == "dbg" and do_performance == 1:
      print "\nERROR: performance tests cannot be run with a debug build, skipping this test\n"
      continue
           
    if do_debug == 0 and dbg_opt == "dbg":
      print "\nWARNING: skipping this test (do_debug: %s, dbg_opt: %s)\n" % (do_debug, dbg_opt)
      continue
    
    if do_opt == 0 and dbg_opt == "opt":
      print "\nWARNING: skipping this test (do_opt: %s, dbg_opt: %s)\n" % (do_opt, dbg_opt)
      continue
    
    if dbg_opt == "opt" : # qwerty: Think this is right now...
      do_memory = 0

    if environ['SCI_MALLOC_ENABLED'] != "yes" :
      do_memory = 0
      
    tests_to_do = [do_uda_comparisons, do_memory, do_performance]
    tolerances = [abs_tolerance, rel_tolerance]
    
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
        print "ERROR: The gold standard for the (%s) test does not exist." % testname
        print "To generate it run: \n   make gold_standards"
        exit(1) 
        
      if environ['LOCAL_OR_NIGHTLY_TEST'] == "nightly" :     
        print "gold Standard being created for  (%s)" % testname
        chdir(compare_root)
        mkdir(testname)
    
    if startFrom == "checkpoint":
      try:
        here = "%s/CheckPoints/%s/%s/%s.uda.000/" %(startpath,ALGO,testname,testname)
        chdir(here)
      except Exception:
        print "checkpoint uda %s does not exist" % here
        print "This file must exist when using 'startFromCheckpoint' option"
        exit(1)
      

    # need to set the inputs dir here, since it could be different per test
    inputsdir = "%s/%s" % (inputpath, ALGO)

    try:
      chdir(inputsdir)
    except Exception:
      print "%s does not exist" % (inputsdir)
      print "Please give a valid <inputsdir> argument"
      exit(1)

    chdir(resultsdir)

    try:
      mkdir(testname)
    except Exception:
      print "Remove %s/%s before running this test" % (resultsdir, testname)
      exit(1)

    system("echo '%s/replace_gold_standard %s %s/%s-results %s' > %s/replace_gold_standard" % (helperspath, compare_root, startpath, ALGO, testname, testname))
    system("chmod gu+rwx %s/replace_gold_standard" % testname)

    chdir(testname)


    # call the callback function before running each test
    list = callback(test, susdir, inputsdir, compare_root, dbg_opt, max_parallelism)

    inputxml = path.basename(input(test))
    system("cp %s/%s %s" % (inputsdir, input(test), inputxml))
    symlink(inputpath, "inputs")
    
    #__________________________________
    # Run test and perform comparisons on the uda
    environ['WEBLOG'] = "%s/%s-results/%s" % (weboutputpath, ALGO, testname)
    rc = runSusTest(test, susdir, inputxml, compare_root, ALGO, dbg_opt, max_parallelism, tests_to_do, tolerances, startFrom)
    system("rm inputs")

    # copy results to web server
    if outputpath != startpath:
      if path.exists("%s/%s-results/%s" % (outputpath, ALGO, testname)) != 1:
        mkdir("%s/%s-results/%s" % (outputpath, ALGO, testname))
      system("cp `ls -1 | grep -v uda` %s/%s-results/%s/" % (outputpath, ALGO, testname))
    
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
        environ['WEBLOG'] = "%s/%s-results/%s/restart" % (weboutputpath, ALGO, testname)
        startFrom = "restart"
        rc = runSusTest(test, susdir, inputxml, compare_root, ALGO, dbg_opt, max_parallelism, tests_to_do, tolerances, startFrom)

        if rc > 0:
          failcode = 1
        system("rm inputs")

        # copy results to web server
        if outputpath != startpath:
          if path.exists("%s/%s-results/%s/restart" % (outputpath, ALGO, testname)) != 1:
            mkdir("%s/%s-results/%s/restart" % (outputpath, ALGO, testname))
          system("cp `ls -1 | grep -v uda` %s/%s-results/%s/restart/" % (outputpath, ALGO, testname))

      chdir("..")
    elif rc == 1: # negative one means skipping -- not a failure
      failcode = 1
    chdir("..")
    
    # timer
    test_timer = time() - test_time0
    print "Test Timer:",strftime("%H:%M:%S",gmtime(test_timer))
    
    # If the test passed put an svn revision stamp in the goldstandard
    # user root is running the cronjob
    user = getoutput("whoami");
    
    if rc > 0:
      print "Failed %i user %s" %(failcode,user)
    
    if failcode == 0 and (user == "csafe-tester" or user == "root"):
      print "Updating the svn revision file %s" %svn_revision
      svn_file = "%s/%s/%s/svn_revision" % (gold_standard,ALGO,testname)
      system( "echo 'This test last passed with %s'> %s" %(svn_revision, svn_file))  
    #__________________________________
    # end of test loop

  # change the group on the results directory to a common group name 
  chdir("..")
  try:
    common_group = "%s" % (environ['COMMON_GROUP'])
    system("chgrp -R %s %s > /dev/null 2>&1" % (common_group, resultsdir) )  
  except:
    pass

  system("chmod -R g+rwX %s > /dev/null 2>&1" % resultsdir)

  # copied results - permissions
  if outputpath != startpath:
    system("chmod -R gu+rwX,a+rX %s/%s-results > /dev/null 2>&1" % (outputpath, ALGO))

  if solotest != "" and solotest_found == 0:
    print "unknown test: %s" % solotest
    system("rm -rf %s" % (resultsdir))
    exit(1)

  # no tests ran
  if ran_any_tests == 0:
    print "Didn't run any tests!"
    exit(3)
  
  #__________________________________  
  # If the tests successfully ran and passed all tests 
  
  if failcode == 0:
    if solotest != "":
      print ""
      print "%s-%s test %s passed successfully!" % (ALGO, dbg_opt, solotest)
    else:
      print ""
      print "All %s-%s tests passed successfully!" % (ALGO, dbg_opt)
  else:
    print ""
    print "Some tests failed"
    
  comp_timer = time() - comp_time0
  print "Component Timer:",strftime("%H:%M:%S",gmtime(comp_timer))
  return failcode

#______________________________________________________________________
# runSusTest() 
# parameters are basically strings, except for tests_to_do which is a list of
# 3 ints stating whether to do comparison, memory, and performance tests
# in that order

def runSusTest(test, susdir, inputxml, compare_root, ALGO, dbg_opt, max_parallelism, tests_to_do, tolerances, startFrom):
  global startpath
  
  testname = nameoftest(test)

  np = float(num_processes(test))
  if (np > max_parallelism):
    if np == 1.1:
      print "Skipping test %s because it requires mpi and max_parallism < 1.1" % testname;
      return -1; 

  do_uda_comparison_test  = tests_to_do[0]
  do_memory_test          = tests_to_do[1]
  do_performance_test     = tests_to_do[2]
  compUda_RC      = 0   # compare_uda return code
  performance_RC  = 0   # performance return code
  memory_RC       = 0   # memory return code


  output_to_browser=1
  try:
    blah = environ['HTMLLOG']
  except Exception:
    output_to_browser=0
  
  #__________________________________
  # Does mpirun command exist or has the environmental variable been set? 
  try :
    MPIRUN = environ['MPIRUN']    # first try the environmental variable
  except :
    MPIRUN = "mpirun"
    rc = system("which mpirun > /dev/null 2>&1")

    if rc == 256:
      print "ERROR:runSusTests.py "
      print "      mpirun command was not found and the environmental variable MPIRUN was not set."
      print "      You must either put mpirun in your path or set the environmental variable"
      exit (1)

  if not do_memory_test :
      unsetenv('MALLOC_STATS')
      unsetenv('SCI_DEBUG')
      
  MPIHEAD="%s -np" % MPIRUN       #default 
  
  # pass in environmental variables to mpirun
  if environ['OS'] == "Linux":
    MPIHEAD="%s -x MALLOC_STATS -x SCI_SIGNALMODE -np" % MPIRUN 
  
                                   # openmpi
  rc = system("mpirun -x TERM echo 'hello' > /dev/null 2>&1")
  if rc == 0:
    MPIHEAD="%s -x MALLOC_STATS -x SCI_SIGNALMODE -np" % MPIRUN
  
                                   #  mvapich
  rc = system("mpirun -genvlist TERM echo 'hello' > /dev/null 2>&1")
  if rc == 0:
    MPIHEAD="%s -genvlist MALLOC_STATS,SCI_SIGNALMODE -np" % MPIRUN
    
  
  # set where to view the log files
  logpath = environ['WEBLOG']

  # if doing performance tests, strip the output and checkpoints portions
  if do_performance_test == 1:
      
    inputxml = modUPS("", inputxml,["<outputInterval>0</outputInterval>",
                                    "<outputTimestepInterval>0</outputTimestepInterval>",
                                    '<checkpoint cycle="0" interval="0"/>'])

    # create a file in tmp/filename, copy it back
    system("cp %s ." % inputxml)
    inputxml = path.basename(inputxml)


  SVN_OPTIONS = "-svnStat -svnDiff"
  #SVN_OPTIONS = "" # When debugging, if you don't want to spend time waiting for SVN, uncomment this line.

  # set the command for sus, based on # of processors
  # the /usr/bin/time is to tell how long it took
  if np == 1:
    command = "/usr/bin/time -p %s/sus %s" % (susdir, SVN_OPTIONS)
    mpimsg = ""
  else:
    command = "/usr/bin/time -p %s %s %s/sus %s -mpi" % (MPIHEAD, int(np), susdir, SVN_OPTIONS)
    mpimsg = " (mpi %s proc)" % (int(np))

  time0 =time()  #timer
  
  #__________________________________ 
  # setup input for sus
  if startFrom == "restart":
    print "Running restart test  ---%s--- %s at %s" % (testname, mpimsg, strftime( "%I:%M:%S"))
    susinput     = "-restart ../*.uda.000 -t 0 -copy"
    restart_text = " (restart)"
    
  if startFrom == "inputFile":
    print "Running test  ---%s--- %s at %s" % (testname, mpimsg, strftime( "%I:%M:%S"))
    susinput     = "%s" % (inputxml)
    restart_text = " "
 
  if startFrom == "checkpoint":
    print "Running test from checkpoint ---%s--- %s at %s" % (testname, mpimsg, strftime( "%I:%M:%S"))
    susinput     = "-restart %s/CheckPoints/%s/%s/*.uda.000" %  (startpath,ALGO,testname)
    restart_text = " "
  #________________________________
  

  # set sus to exit upon crashing (and not wait for a prompt)
  environ['SCI_SIGNALMODE'] = "exit"

  if do_memory_test == 1:
  
    environ['MALLOC_STRICT'] = "set"
    environ['SCI_DEBUG']     ="VarLabel:+"
    
    if startFrom == "restart":
      malloc_stats_file = "restart_malloc_stats"        
    else:
      malloc_stats_file = "malloc_stats"
    environ['MALLOC_STATS'] = malloc_stats_file

  # messages to print
  if environ['outputlinks'] == "1":
    sus_log_msg = '\t<A href=\"%s/sus.log.txt\">See sus.log</a> for details' % (logpath)
    compare_msg = '\t<A href=\"%s/compare_sus_runs.log.txt\">See compare_sus_runs.log</A> for more comparison information.' % (logpath)
    memory_msg  = '\t<A href=\"%s/mem_leak_check.log.txt\">See mem_leak_check.log</a> for more comparison information.' % (logpath)
    perf_msg    = '\t<A href=\"%s/performance_check.log.txt\">See performance_check.log</a> for more comparison information.' % (logpath)
  else:
    sus_log_msg = '\tSee %s/sus.log.txt for details' % (logpath)
    compare_msg = '\tSee %s/compare_sus_runs.log.txt for more comparison information.' % (logpath)
    memory_msg  = '\tSee %s/mem_leak_check.log.txt for more comparison information.' % (logpath)
    perf_msg    = '\tSee %s/performance_check.log.txt for more performance information.' % (logpath)

  # actually run the test!
  short_cmd = command.replace(susdir+'/','')

  print "Command Line: %s %s" % (short_cmd, susinput)
  rc = system("%s %s > sus.log.txt 2>&1" % (command, susinput))
  
  # was an exception thrown
  exception = system("grep -q 'Caught exception' sus.log.txt");
  if exception == 0:
    print "\t*** An exception was thrown ***";
    rc = -9
    
  # determine path of replace_msg in 2 places to not have 2 different msgs.
  replace_msg = "\tTo replace this test's goldStandards run:\n\t    "
  
  if startFrom == "restart":
    chdir("..")
    replace_msg = "%s%s/replace_gold_standard" % (replace_msg, getcwd())
    chdir("restart")
  else:
    replace_msg = "%s%s/replace_gold_standard" % (replace_msg, getcwd())
  
  replace_msg = "%s\n\tTo replace multiple tests that have failed run:\n\t    %s/replace_all_GS\n" % (replace_msg,startpath)
  
  return_code = 0
  if rc != 0:
    print "\t*** Test %s failed with code %d" % (testname, rc)
    
    if startFrom == "restart":
      print "\t\tMake sure the problem makes checkpoints before finishing"
    
    print sus_log_msg
    print 
    system("echo '  :%s: %s test did not run to completion' >> %s/%s-short.log" % (testname,restart_text,startpath,ALGO))
    return_code = 1
  else:
    # Sus completed successfully - now run memory,compar_uda and performance tests

    # get the time from sus.log
    # /usr/bin/time outputs 3 lines, the one called 'real' is what we want
    # it is the third line from the bottom

    # save this file independent of performance tests being done
    print "\tSuccessfully ran to completion"

    if startFrom == "restart":
      ts_file = "restart_timestamp"
    else:
      ts_file = "timestamp"
    system("tail -n3 sus.log.txt > %s" % ts_file)
    
    #__________________________________
    # performance test
    if do_performance_test == 1:
      print "\tPerforming performance test on %s" % (date())
      performance_RC = system("performance_check %s %s %s/%s/%s > performance_check.log.txt 2>&1" % (testname, ts_file, compare_root, testname, ts_file))
      try:
        short_message_file = open("performance_shortmessage.txt", 'r+', 500)
        short_message = rstrip(short_message_file.readline(500))
      except Exception:
        short_message = ""
        
      if performance_RC == 0:
        print "\tPerformance tests passed."
        if short_message != "":
          print "\t%s" % (short_message)    
      elif performance_RC == 5 * 256:
        print "\t* Warning, no timestamp file created.  No performance test performed."
      elif performance_RC == 2*256:
        print "\t*** Warning, test %s failed performance test." % (testname)
        if short_message != "":
          print "\t%s" % (short_message)
          
        print perf_msg
        print "%s" % replace_msg
      else:
        print "\tPerformance tests passed. (Note: no previous performace stats)."

    #__________________________________
    # uda comparison
    if do_uda_comparison_test == 1:
      print "\tComparing udas"

      if dbg_opt == "dbg":
        environ['MALLOC_STATS'] = "compare_uda_malloc_stats"

      abs_tol= tolerances[0]
      rel_tol= tolerances[1]
      compUda_RC = system("compare_sus_runs %s %s %s %s %s %s> compare_sus_runs.log.txt 2>&1" % (testname, getcwd(), compare_root, susdir,abs_tol, rel_tol))
      if compUda_RC != 0:
        if compUda_RC == 10 * 256:
          print "\t*** Input file(s) differs from the goldstandard"
          print "%s" % replace_msg
          
        elif compUda_RC == 1 * 256 or compUda_RC == 5*256:
          print "\t*** Warning, test %s failed uda comparison with error code %s" % (testname, compUda_RC)
          print compare_msg
          
          if startFrom != "restart":
           print "%s" % replace_msg
        
        elif compUda_RC == 65280: # (-1 return code)
          print "\tComparison tests passed.  (Note: No dat files to compare.)"
        
        else:
          print "\tComparison tests passed.  (Note: No previous gold standard.)"
      
      else:
        print "\tComparison tests passed."
    #__________________________________
    # Memory leak test
    if do_memory_test == 1:
      memory_RC = system("mem_leak_check %s %s %s/%s/%s %s > mem_leak_check.log.txt 2>&1" % (testname, malloc_stats_file, compare_root, testname, malloc_stats_file, "."))
      try:
        short_message_file = open("highwater_shortmessage.txt", 'r+', 500)
        short_message = rstrip(short_message_file.readline(500))
      except Exception:
        short_message = ""

      if memory_RC == 0:
          print "\tMemory leak tests passed."
          if short_message != "":
            print "\t%s" % (short_message)    
      elif memory_RC == 5 * 256:
          print "\t* Warning, no malloc_stats file created.  No memory leak test performed."
      elif memory_RC == 256:
          print "\t*** Warning, test %s failed memory leak test." % (testname)
          print memory_msg
          # check that all VarLabels were deleted
          rc = system("mem_leak_checkVarLabels sus.log.txt")  
      elif memory_RC == 2*256:
          print "\t*** Warning, test %s failed memory highwater test." % (testname)
          if short_message != "":
            print "\t%s" % (short_message)
          print memory_msg
          print "%s" % replace_msg
      else:
          print "\tMemory leak tests passed. (Note: no previous memory usage stats)."
    #__________________________________
    # print error codes
    # if comparison, memory, performance tests fail, return here, so mem_leak tests can run
    if compUda_RC == 5*256 or compUda_RC == 1*256:
      system("echo '  :%s: \t%s test failed comparison tests' >> %s/%s-short.log" % (testname,restart_text,startpath,ALGO))
      return_code = 2;
        
    if performance_RC == 2*256:
      system("echo '  :%s: \t%s test failed performance tests' >> %s/%s-short.log" % (testname,restart_text,startpath,ALGO))
      return_code = 2;
    
    if memory_RC == 1*256 or memory_RC == 2*256:
      system("echo '  :%s: \t%s test failed memory tests' >> %s/%s-short.log" % (testname,restart_text,startpath,ALGO))
      return_code = 2;
    
    if return_code != 0:
      # as the file is only created if a certain test fails, change the permissions here as we are certain the file exists
      system("chmod gu+rw,a+r %s/%s-short.log > /dev/null 2>&1" % (startpath, ALGO))
          
  return return_code
