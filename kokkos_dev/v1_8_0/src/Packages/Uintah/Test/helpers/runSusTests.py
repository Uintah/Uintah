#!/usr/local/bin/python

from os import environ,rmdir,mkdir,path,system,chdir,stat,getcwd,pathsep,symlink
from time import asctime,localtime,time
from sys import argv,exit
from string import upper,rstrip

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

  ALGO = upper(algo)
  susdir =  path.normpath(path.join(getcwd(), argv[1]))

  if ALGO == "EXAMPLES":
    ALGO = "Examples"
  inputsdir = "%s/%s" % (path.normpath(path.join(getcwd(), inputs_root())), \
                         ALGO)
  gold_standard = path.normpath(path.join(getcwd(), argv[3]))
  mode = argv[4]
  max_parallelism = float(argv[5])

  if max_parallelism < 1:
    max_parallelism = 1;

  solotest = ""
  if len(argv) == 7:
    solotest = argv[6]
  
  startpath = getcwd()

# If run from startTester, tell it to output logs in web dir
# otherwise, save it in the build
  try:

# if webpath exists, use that, otherwise, use BUILDROOT/mode
    outputpath = "%s-%s" % (environ['HTMLLOG'], mode)
    weboutputpath = "%s-%s" % (environ['WEBLOG'], mode)
    try:
      # make outputpath/dbg or opt dirs
      mkdir(outputpath)
    except Exception:
      pass
  except Exception:
    outputpath = startpath
    weboutputpath = startpath
  
  helperspath = "%s/%s" % (path.normpath(path.join(getcwd(), path.dirname(argv[0]))), "helpers")

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
    chdir(inputsdir)
  except Exception:
    print "%s does not exist" % (inputsdir)
    print "Please give a valid <inputsdir> argument"
    exit(1)

  try:
    chdir(gold_standard)
  except Exception:
    print "%s does not exist" % (gold_standard)
    print "Please give a valid <testdata_goldstandard> argument"
    exit(1)
  compare_root = "%s/%s" % (gold_standard, ALGO)

  environ['PATH'] = "%s%s%s" % (helperspath, pathsep, environ['PATH'])
  environ['SCI_SIGNALMODE'] = 'exit'
  environ['SCI_EXCEPTIONMODE'] = 'abort'

  resultsdir = "%s/%s-results" % (outputpath, ALGO)

  chdir(startpath)
  try:
    mkdir(resultsdir)
  except Exception:
    if solotest == "":
      print "Remove %s before running this test" % resultsdir
      exit(1)

  chdir(resultsdir)

  failcode = 0

  DO_RESTART = "yes"

  print ""
  if solotest == "":
    print "Performing %s-%s tests." % (ALGO, mode)
  else:
    print "Performing %s-%s test %s." % (ALGO, mode, solotest)
  print "===================================="
  print ""


  solotest_found = 0
  for test in TESTS:
    if solotest != "" and nameoftest(test) != solotest:
      continue

    # if test can be run on this OS
    if testOS(test) != environ['OS'] and testOS(test) != "ALL":
      continue
    solotest_found = 1 # if there is a solotest, that is
    testname = nameoftest(test)

    try:
      mkdir(testname)
    except Exception:
      print "Remove %s/%s before running this test" % (resultsdir, testname)
      exit(1)

    system("echo '%s/replace_gold_standard %s %s %s' > %s/replace_gold_standard" % (helperspath, compare_root, getcwd(), testname, testname))
    system("chmod gu+rwx %s/replace_gold_standard" % testname)

    chdir(testname)

    # call the callback function before running each test
    callback(test, susdir, inputsdir, compare_root, algo, mode, max_parallelism)
    if ALGO == "Examples":
      newalgo = testname
    else:
      newalgo = ""

    inputxml = path.basename(input(test))
    system("cp %s/%s %s" % (inputsdir, input(test), inputxml))

    # Run normal test
    environ['WEBLOG'] = "%s/%s-results/%s" % (weboutputpath, ALGO, testname)
    rc = runSusTest(test, susdir, inputxml, compare_root, algo, mode, max_parallelism, "no", newalgo)

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
      environ['WEBLOG'] = "%s/%s-results/%s/restart" % (weboutputpath, ALGO, testname)
      rc = runSusTest(test, susdir, inputxml, compare_root, algo, mode, max_parallelism, DO_RESTART, newalgo)
      if rc > 0:
        failcode = 1
      chdir("..")
    elif rc == 1: # negative one means skipping -- not a failure
      failcode = 1

    chdir("..")
  
  chdir("..")

  system("chgrp -R csafe %s" % resultsdir)
  system("chmod -R g+rwX %s" % resultsdir)

  # if results saved on the web server, copy back to build root
  if outputpath != startpath:
    system("cp -r %s %s/%s-results" % (resultsdir, startpath, ALGO))

  if solotest != 0 and solotest_found == 0:
    print "unknown test: %s" % solotest
    system("rm -rf %s" % (resultsdir))
    exit(1)

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


def runSusTest(test, susdir, inputxml, compare_root, algo, mode, max_parallelism, do_restart = "no", newalgo = ""):
  if newalgo != "":
    algo = newalgo
  testname = nameoftest(test)
  np = float(num_processes(test))
  if (np > max_parallelism):
    if np == 1.1:
      print "Skipping test %s because it requires mpi and max_parallism < 1.1" % testname;
    else:
      print "Skipping test %s because %s processors exceeds maximum of %s" % (testname, np, max_parallelism);
    return -1; 

  extra_flags = extra_sus_flags(test)

  # set where to view the log files
  logpath = environ['WEBLOG']
# set the command for sus, based on # of processors
  if np == 1:
    command = "%s/sus -%s %s" % (susdir, algo, extra_flags)
    mpimsg = ""
  else:
    command = "mpirun -np %s %s/sus -%s %s" % (int(np), susdir, algo, extra_flags)
    mpimsg = " (mpi %s proc)" % (int(np))

  if do_restart == "yes":
    print "Running restart test for %s%s on %s" % (testname, mpimsg, date())
  else:
    print "Running test %s%s on %s" % (testname, mpimsg, date())


  if do_restart == "yes":
    susinput = "-restart ../*.uda.000 -t 0 -move"
  else:
    susinput = "%s" % (inputxml)

  if mode == "dbg":
    if do_restart == "yes":
      malloc_stats_file = "restart_malloc_stats"
    else:
      malloc_stats_file = "malloc_stats"
    environ['MALLOC_STATS'] = malloc_stats_file

  rc = system("nice %s %s > sus.log.txt 2>&1" % (command, susinput))
  print "Command Line: %s %s" % (command, susinput)

  if mode == "dbg":
    environ['MALLOC_STATS'] = "compare_uda_malloc_stats"

  if rc != 0:
    print "\t*** Test %s failed with code %d" % (testname, rc)
    if do_restart == "yes":
	print "\t\tMake sure the problem makes checkpoints before finishing"
    print '\t<A href=\"%s/sus.log.txt\">See sus.log</a> for details' % (logpath)
    return 1
  else:
    # determine path of replace_msg in 2 places to not have 2 different msgs.
    replace_msg = "\tTo replace the gold standard uda and memory usage with these results,\n\trun: "

    if do_restart == "yes":
      chdir("..")
      replace_msg = "%s%s/replace_gold_standard" % (replace_msg, getcwd())
      system("rm *.uda")
      system("ln -s restart/*.uda .")
      chdir("restart")
    else:
      replace_msg = "%s%s/replace_gold_standard" % (replace_msg, getcwd())
    print "\tComparing udas on %s" % (date())

    cu_rc = system("compare_sus_runs %s %s %s %s > compare_sus_runs.log.txt 2>&1" % (testname, getcwd(), compare_root, susdir))
    if cu_rc != 0:
	if cu_rc == 5 * 256:
     	    print "\t*** Warning, %s has changed.  You must update the gold standard." % (input(test))
    	    print '\t<A href=\"%s/compare_sus_runs.log.txt\">See compare_sus_runs.log</A> for more comparison information.' % (logpath)
 	    print "%s" % replace_msg
	elif cu_rc == 10 * 256:
     	    print "\t*** Warning, %s has changed.  You must update the gold standard." % (input(test))
    	    print "\tAll other comparison tests passed so the change was likely trivial."
 	    print "%s" % replace_msg
	elif cu_rc == 1 * 256:
    	    print "\t*** Warning, test %s failed uda comparison with error code %s" % (testname, cu_rc)
            print '\t<A href=\"%s/compare_sus_runs.log.txt\">See compare_sus_runs.log</a> for more comparison information.' % (logpath)
	    if do_restart != "yes":
 	    	print "%s" % replace_msg
	elif cu_rc == 65280: # (-1 return code)
	    print "\tComparison tests passed.  (Note: No dat files to compare.)"
        else:
	    print "\tComparison tests passed.  (Note: No previous gold standard.)"
    else:
        print "\tComparison tests passed."

    if mode == "dbg":
	rc = system("mem_leak_check %s %s %s/%s/%s %s > mem_leak_check.log.txt 2>&1" % (testname, malloc_stats_file, compare_root, testname, malloc_stats_file, "."))
	try:
	  short_message_file = open("highwater_shortmessage.txt", 'r+', 500)
	  short_message = rstrip(short_message_file.readline(500))
	except Exception:
	  short_message = ""

        if rc == 0:
	    print "\tMemory leak tests passed."
	    if short_message != "":
		print "\t%s" % (short_message)    
	elif rc == 5 * 256:
	    print "\t* Warning, no malloc_stats file created.  No memory leak test performed."
	elif rc == 256:
	    print "\t*** Warning, test %s failed memory leak test." % (testname)
            print '\t<A href=\"%s/mem_leak_check.log.txt\">See mem_leak_check.log</a> for more comparison information.' % (logpath)
	    return 2
	elif rc == 2*256:
	    print "\t*** Warning, test %s failed memory highwater test." % (testname)
	    if short_message != "":
		print "\t%s" % (short_message)
            print '\t<A href=\"%s/mem_leak_check.log.txt\">See mem_leak_check.log</a> for more comparison information.' % (logpath)
 	    print "%s" % replace_msg
	    return 2
	else:
	    print "\tMemory leak tests passed. (Note: no previous memory usage stats)."

    # if comparison tests fail, return here, so mem_leak tests can run
    if cu_rc != 0:
        return 2;
  return 0



