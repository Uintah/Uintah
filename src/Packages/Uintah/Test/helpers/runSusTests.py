#!/usr/local/bin/python

from os import environ,rmdir,mkdir,path,system,chdir,stat,getcwd,pathsep,symlink
from time import asctime,localtime,time
from sys import argv,exit
from string import upper,rstrip
from modUPS import modUPS

def nameoftest (test):
    return test[0]
def input (test):
    return test[1]
def extra_sus_flags (test):
    return test[2]
def num_processes (test):
    return test[3]
def testOS(test):
    return test[4]
def inputs_root ():
    return argv[2]
def date ():
    return asctime(localtime(time()))
def perf_algo (test):
    return test[-1]
def nullCallback (test, susdir, inputsdir, compare_root, algo, mode, max_parallelism):
    pass

# if a callback is given, it is executed before running each test and given
# all of the paramaters given to runSusTest
def runSusTests(argv, TESTS, algo, callback = nullCallback):
  if len(argv) < 6 or len(argv) > 7 or not argv[4] in ["dbg", "opt"] :
    print "usage: %s <susdir> <inputsdir> <testdata_goldstandard> <mode> " \
	     "<max_parallelsim> <test>" % argv[0]
    print "    where <mode> is 'dbg' or 'opt' and <test> is optional"
    exit(1)

  # setup up parameter variables
  ALGO = upper(algo)
  susdir =  path.normpath(path.join(getcwd(), argv[1]))

  if ALGO == "EXAMPLES":
    ALGO = "Examples"
  gold_standard = path.normpath(path.join(getcwd(), argv[3]))
  mode = argv[4]
  max_parallelism = float(argv[5])

  if max_parallelism < 1:
    max_parallelism = 1;

  solotest = ""
  if len(argv) == 7:
    solotest = argv[6]
  
  # performance tests are run only by passing "performance" as the algo
  #   if algo is "performance", then there can be any algo associated with that
  # determine which tests to do
  if algo == "performance":
    do_restart = 0
    do_dbg = 0
    do_comparisons = 0
    do_memory = 0
    do_performance = 1
  else:
    do_restart = 1  
    do_dbg = 1
    do_comparisons = 1
    do_memory = 1
    do_performance = 0

  if mode == "dbg" and do_dbg == 0:
    print "Skipping %s tests because we're in debug mode" % algo
    return 3

  if mode == "opt":
    do_memory = 0
    
  tests_to_do = [do_comparisons, do_memory, do_performance]


  startpath = getcwd()

  # whether or not to display html links on output
  environ['outputlinks']="0"
  
  # If run from startTester, tell it to output logs in web dir
  # otherwise, save it in the build, and display links
  try:

    # if webpath exists, use that, otherwise, use BUILDROOT/mode
    outputpath = "%s-%s" % (environ['HTMLLOG'], mode)
    weboutputpath = "%s-%s" % (environ['WEBLOG'], mode)
    try:
      # make outputpath/dbg or opt dirs
      environ['outputlinks'] ="1"
      mkdir(outputpath)
    except Exception:
      pass
  except Exception:
    outputpath = startpath
    weboutputpath = startpath
  
  helperspath = "%s/%s" % (path.normpath(path.join(getcwd(), path.dirname(argv[0]))), "helpers")

  inputpath = path.normpath(path.join(getcwd(), inputs_root()))

  
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

  environ['PATH'] = "%s%s%s" % (helperspath, pathsep, environ['PATH'])
  environ['SCI_SIGNALMODE'] = 'exit'
  environ['SCI_EXCEPTIONMODE'] = 'abort'

  resultsdir = "%s/%s-results" % (startpath, ALGO)

  chdir(startpath)
  try:
    mkdir(resultsdir)
  except Exception:
    if solotest == "":
      print "Remove %s before running this test" % resultsdir
      exit(1)

  chdir(resultsdir)



  environ['MPI_TYPE_MAX'] = '10000'

  print ""
  if solotest == "":
    print "Performing %s-%s tests." % (ALGO, mode)
  else:
    print "Performing %s-%s test %s." % (ALGO, mode, solotest)
  print "===================================="
  print ""

  failcode = 0

  # this is to see if any tests actually ran (so we can say tests skipped)
  ran_any_tests = 0

  solotest_found = 0
  for test in TESTS:
    if solotest != "" and nameoftest(test) != solotest:
      continue

    # if test can be run on this OS
    if testOS(test) != environ['OS'] and testOS(test) != "ALL":
      continue
    solotest_found = 1 # if there is a solotest, that is
    testname = nameoftest(test)
    ran_any_tests = 1

    # make sure that this test exists in the gold standard
    try:
      chdir(compare_root)
      chdir(testname)
    except Exception:
      chdir(compare_root)
      mkdir(testname)
    
    
    # in certain cases (like when algo was performance), we need to make it
    # something usable by sus (MPM, ARCHES, etc.), but we will also need to
    # have the original ALGO name, i.e., to save in PERFORMANCE-results

    # set inputsdir in the loop since certain (performance) algos can have
    # different inputs dirs or have different sus flags
    if ALGO == "Examples":
      newalgo = testname
      NEWALGO = ALGO
    elif do_performance == 1:
      newalgo = perf_algo(test)
      NEWALGO = upper(newalgo)
    else:
      newalgo = ""
      NEWALGO = ALGO


    inputsdir = "%s/%s" % (inputpath, NEWALGO)

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
    callback(test, susdir, inputsdir, compare_root, algo, mode, max_parallelism)

    inputxml = path.basename(input(test))
    system("cp %s/%s %s" % (inputsdir, input(test), inputxml))

    # Run normal test
    environ['WEBLOG'] = "%s/%s-results/%s" % (weboutputpath, ALGO, testname)
    rc = runSusTest(test, susdir, inputxml, compare_root, algo, mode, max_parallelism, tests_to_do, "no", newalgo)

    # rc of 2 means it failed comparison or memory test, so try to run restart
    # anyway
    if rc == 0 or rc == 2:
      # Prepare for restart test
      if rc == 2:
          failcode = 1
      mkdir("restart")
      chdir("restart")
      # call the callback function before running each test
      callback(test, susdir, inputsdir, compare_root, algo, mode, max_parallelism);

      # Run restart test
      if do_restart == 1:
        environ['WEBLOG'] = "%s/%s-results/%s/restart" % (weboutputpath, ALGO, testname)
        rc = runSusTest(test, susdir, inputxml, compare_root, algo, mode, max_parallelism, tests_to_do, "yes", newalgo)
        if rc > 0:
          failcode = 1
      chdir("..")
    elif rc == 1: # negative one means skipping -- not a failure
      failcode = 1
    chdir("..")
  
  chdir("..")

  system("chgrp -R csafe %s" % resultsdir)
  system("chmod -R g+rwX %s" % resultsdir)

  # copy results to web server.
  if outputpath != startpath:
    #system("cp -f %s-short* %s/ > /dev/null 2>&1" % (upper(algo),outputpath))
    system("cp -r %s %s/" % (resultsdir, outputpath))
    # remove uda dirs from web server
    system("find %s -name '*.uda*' | xargs rm -rf " % outputpath)

  if solotest != "" and solotest_found == 0:
    print "unknown test: %s" % solotest
    system("rm -rf %s" % (resultsdir))
    exit(1)

  # no tests ran
  if ran_any_tests == 0:
    exit(3)
    
  if failcode == 0:
    if solotest != "":
      print ""
      print "%s-%s test %s passed successfully!" % (ALGO, mode, solotest)
    else:
      print ""
      print "All %s-%s tests passed successfully!" % (ALGO, mode)
  else:
    print ""
    print "Some tests failed"
  exit(failcode)


# parameters are basically strings, except for tests_to_do which is a list of
# 3 ints stating whether to do comparison, memory, and performance tests
# in that order

def runSusTest(test, susdir, inputxml, compare_root, algo, mode, max_parallelism, tests_to_do, restart = "no", newalgo = ""):
  if newalgo != "":
    sim = newalgo
  else:
    sim = algo
  testname = nameoftest(test)

  np = float(num_processes(test))
  if (np > max_parallelism):
    if np == 1.1:
      print "Skipping test %s because it requires mpi and max_parallism < 1.1" % testname;
    else:
      print "Skipping test %s because %s processors exceeds maximum of %s" % (testname, np, max_parallelism);
    return -1; 

  do_comparison_test = tests_to_do[0]
  do_memory_test = tests_to_do[1]
  do_performance_test = tests_to_do[2]
  cu_rc = 0
  pf_rc = 0
  mem_rc = 0


  extra_flags = extra_sus_flags(test)
  output_to_browser=1
  try:
    blah = environ['HTMLLOG']
  except Exception:
    output_to_browser=0

  # set the command name for mpirun - differs on different platforms
  MPIHEAD="mpirun -np"
  if environ['OS'] == "Linux":
    MPIHEAD="mpirun -x MALLOC_STATS,SCI_SIGNALMODE -np" 

  # set where to view the log files
  logpath = environ['WEBLOG']

  startpath = "../.."

  # if doing performance tests, strip the output and checkpoints portions
  if do_performance_test == 1:
    inputxml = modUPS("", inputxml,["<outputInterval>0</outputInterval>",
                      "<outputTimestepInterval>0</outputTimestepInterval>",
                      '<checkpoint interval="0"/>'])

    # will create a file in tmp/filename, copy it back
    system("cp %s ." % inputxml)
    inputxml = path.basename(inputxml)


  # set the command for sus, based on # of processors
  # the /usr/bin/time is to tell how long it took
  if np == 1:
    command = "/usr/bin/time -p %s/sus -%s %s" % (susdir, sim, extra_flags)
    mpimsg = ""
  else:
    command = "/usr/bin/time -p %s %s %s/sus -mpi -%s %s" % (MPIHEAD, int(np), susdir, sim, extra_flags)
    mpimsg = " (mpi %s proc)" % (int(np))

  if restart == "yes":
    print "Running restart test for %s%s on %s" % (testname, mpimsg, date())
    susinput = "-restart ../*.uda.000 -t 0 -move"
    startpath = "../../.."
    restart_text = " (restart)"
  else:
    print "Running test %s%s on %s" % (testname, mpimsg, date())
    susinput = "%s" % (inputxml)
    restart_text = " "

  # set sus to exit upon crashing (and not wait for a prompt)
  environ['SCI_SIGNALMODE'] = "exit"

  if do_memory_test == 1:
    if restart == "yes":
      malloc_stats_file = "restart_malloc_stats"        
    else:
      malloc_stats_file = "malloc_stats"
    environ['MALLOC_STATS'] = malloc_stats_file

    # if regression tester was called with -malloc_strict
    try:
      if environ['mallocstrict'] == "yes":
        environ['MALLOC_STRICT'] = "blah"
    except Exception:
      pass


  # messages to print
  if environ['outputlinks'] == "1":
    sus_log_msg = '\t<A href=\"%s/sus.log.txt\">See sus.log</a> for details' % (logpath)
    compare_msg = '\t<A href=\"%s/compare_sus_runs.log.txt\">See compare_sus_runs.log</A> for more comparison information.' % (logpath)
    memory_msg  = '\t<A href=\"%s/mem_leak_check.log.txt\">See mem_leak_check.log</a> for more comparison information.' % (logpath)
    perf_msg = '\t<A href=\"%s/performance_check.log.txt\">See performance_check.log</a> for more comparison information.' % (logpath)
  else:
    sus_log_msg = '\tSee %s/sus.log.txt for details' % (logpath)
    compare_msg = '\tSee %s/compare_sus_runs.log.txt for more comparison information.' % (logpath)
    memory_msg  = '\tSee %s/mem_leak_check.log.txt for more comparison information.' % (logpath)
    perf_msg  = '\tSee %s/performance_check.log.txt for more performance information.' % (logpath)

  # actually run the test!
  print "Command Line: %s %s" % (command, susinput)
  rc = system("%s %s > sus.log.txt 2>&1" % (command, susinput))


  # determine path of replace_msg in 2 places to not have 2 different msgs.
  replace_msg = "\tTo replace the gold standard uda and memory usage with these results,\n\trun: "

  if restart == "yes":
    chdir("..")
    replace_msg = "%s%s/replace_gold_standard" % (replace_msg, getcwd())
    system("rm *.uda")
    system("ln -s restart/*.uda .")
    chdir("restart")
  else:
    replace_msg = "%s%s/replace_gold_standard" % (replace_msg, getcwd())
    restart_text = ""

  if rc != 0:
    print "\t*** Test %s failed with code %d" % (testname, rc)
    if restart == "yes":
	print "\t\tMake sure the problem makes checkpoints before finishing"
    print sus_log_msg
    system("echo '  -- %s%s test failed to complete' >> %s/%s-short.log" % (testname,restart_text,startpath,upper(algo)))
    return 1
  else:
    # Sus completed successfully - now do comp, mem, and perf tests

    # get the time from sus.log
    # /usr/bin/time outputs 3 lines, the one called 'real' is what we want
    # it is the third line from the bottom

    # save this file independent of performance tests being done
    if restart == "yes":
      ts_file = "restart_timestamp"
    else:
      ts_file = "timestamp"
    system("tail -n3 sus.log.txt > %s" % ts_file)
 
    if do_performance_test == 1:
      print "\tPerforming performance test on %s" % (date())
      pf_rc = system("performance_check %s %s %s/%s/%s > performance_check.log.txt 2>&1" % (testname, ts_file, compare_root, testname, ts_file))
      try:
        short_message_file = open("performance_shortmessage.txt", 'r+', 500)
        short_message = rstrip(short_message_file.readline(500))
      except Exception:
        short_message = ""
      if pf_rc == 0:
        print "\tPerformance tests passed."
        if short_message != "":
	  print "\t%s" % (short_message)    
      elif pf_rc == 5 * 256:
        print "\t* Warning, no timestamp file created.  No performance test performed."
      elif pf_rc == 2*256:
        print "\t*** Warning, test %s failed performance test." % (testname)
        if short_message != "":
	  print "\t%s" % (short_message)
        print perf_msg
        print "%s" % replace_msg
      else:
	print "\tPerformance tests passed. (Note: no previous performace stats)."


    if do_comparison_test == 1:
      print "\tComparing udas on %s" % (date())

      if mode == "dbg":
        environ['MALLOC_STATS'] = "compare_uda_malloc_stats"

      cu_rc = system("compare_sus_runs %s %s %s %s > compare_sus_runs.log.txt 2>&1" % (testname, getcwd(), compare_root, susdir))
      if cu_rc != 0:
	  if cu_rc == 5 * 256:
     	      print "\t*** Warning, %s has changed or has different defaults.\n\tYou must update the gold standard." % (input(test))
    	      print compare_msg
 	      print "%s" % replace_msg
	  elif cu_rc == 10 * 256:
     	    print "\t*** Warning, %s has changed or has different defaults.\n\tYou must update the gold standard." % (input(test))
    	    print "\tAll other comparison tests passed so the change was likely trivial."
 	    print "%s" % replace_msg
	  elif cu_rc == 1 * 256:
    	    print "\t*** Warning, test %s failed uda comparison with error code %s" % (testname, cu_rc)
            print compare_msg
	    if restart != "yes":
 	    	print "%s" % replace_msg
	  elif cu_rc == 65280: # (-1 return code)
	    print "\tComparison tests passed.  (Note: No dat files to compare.)"
          else:
	    print "\tComparison tests passed.  (Note: No previous gold standard.)"
      else:
        print "\tComparison tests passed."

    if do_memory_test == 1:
	mem_rc = system("mem_leak_check %s %s %s/%s/%s %s > mem_leak_check.log.txt 2>&1" % (testname, malloc_stats_file, compare_root, testname, malloc_stats_file, "."))
	try:
	  short_message_file = open("highwater_shortmessage.txt", 'r+', 500)
	  short_message = rstrip(short_message_file.readline(500))
	except Exception:
	  short_message = ""

        if mem_rc == 0:
	    print "\tMemory leak tests passed."
	    if short_message != "":
		print "\t%s" % (short_message)    
	elif mem_rc == 5 * 256:
	    print "\t* Warning, no malloc_stats file created.  No memory leak test performed."
	elif mem_rc == 256:
	    print "\t*** Warning, test %s failed memory leak test." % (testname)
            print memory_msg
	elif mem_rc == 2*256:
	    print "\t*** Warning, test %s failed memory highwater test." % (testname)
	    if short_message != "":
		print "\t%s" % (short_message)
            print memory_msg
 	    print "%s" % replace_msg
	else:
	    print "\tMemory leak tests passed. (Note: no previous memory usage stats)."

    # if comparison tests fail, return here, so mem_leak tests can run
    if cu_rc == 5*256 or cu_rc == 1*256:
        system("echo '  -- %s%s test failed comparison tests' >> %s/%s-short.log" % (testname,restart_text,startpath,upper(algo)))
        # debug
        return 2;
    if pf_rc == 2*256:
        # For the present, hard code the PERFORMANCE as the algo, as
        # performance is a test set and not an algorithm.  This follows the
        # current model of performance tests (all in one test file), but will
        # need to change if the model changes
        system("echo '  -- %s%s test failed performance tests' >> %s/PERFORMANCE-short.log" % (testname,restart_text,startpath))
        return 2;
    if mem_rc == 1*256 or mem_rc == 2*256:
        # debug
        system("echo '  -- %s%s test failed memory tests' >> %s/%s-short.log" % (testname,restart_text,startpath,upper(algo)))
        return 2;
  return 0
