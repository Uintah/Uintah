#!/usr/local/bin/python

from os import environ,rmdir,mkdir,path,system,chdir,stat,getcwd,pathsep,symlink
from time import asctime,localtime,time
from sys import argv,exit
from string import upper

def nameoftest (test):
    return test[0]
def input (test):
    return test[1]
def num_processes (test):
    return test[2]
def date ():
    return asctime(localtime(time()))

def nullCallback (test, susdir, inputsdir, compare_root, algo, mode, max_parallelism):
    pass

# if a callback is given, it is executed before running each test and given
# all of the paramaters given to runSusTest
def runSusTests(argv, TESTS, algo, callback = nullCallback):
  if len(argv) < 6 or len(argv) > 7 or not argv[4] in ["dbg", "opt"] :
    print "usage: %s <susdir> <inputsdir> <testdata_goldstandard> <mode> <max_parallelsim> <test>" % argv[0]
    print "    where <mode> is 'dbg' or 'opt' and <test> is optional"
    exit(1)

  ALGO = upper(algo)
  susdir =  path.normpath(path.join(getcwd(), argv[1]))
  inputsdir = "%s/%s" % (path.normpath(path.join(getcwd(), argv[2])), ALGO)
  gold_standard = path.normpath(path.join(getcwd(), argv[3]))
  mode = argv[4]
  max_parallelism = float(argv[5])

  if max_parallelism < 1:
    max_parallelism = 1;

  solotest = ""
  if len(argv) == 7:
    solotest = argv[6]
  
  startpath = getcwd()

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

  resultsdir = "%s-results" % ALGO

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
    callback(test, susdir, inputsdir, compare_root, algo, mode, max_parallelism);

    # Run normal test
    rc = runSusTest(test, susdir, inputsdir, compare_root, algo, mode, max_parallelism)
    if rc == 0:
      # Prepare for restart test
      mkdir("restart")
      chdir("restart")
    
      # call the callback function before running each test
      callback(test, susdir, inputsdir, compare_root, algo, mode, max_parallelism);

      # Run restart test
      rc = runSusTest(test, susdir, inputsdir, compare_root, algo, mode, max_parallelism, DO_RESTART)
      if rc == 1:
        failcode = 1
      chdir("..")
    elif rc == 1: # negative one means skipping -- not a failure
      failcode = 1

    chdir("..")
  
  chdir("..")

  system("chmod g+w %s" % resultsdir)


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


def runSusTest(test, susdir, inputsdir, compare_root, algo, mode, max_parallelism, do_restart = "no"):
  ALGO = upper(algo)
  testname = nameoftest(test)
  np = float(num_processes(test))
  if (np > max_parallelism):
    if np == 1.1:
      print "Skipping test %s because it requires mpi and max_parallism < 1.1";
    else:
      print "Skipping test %s because %s processors exceeds maximum of %s" % (testname, np, max_parallelism);
    return -1; 

  if np == 1:
    command = "%s/sus -%s" % (susdir, algo)
    mpimsg = ""
  else:
    command = "mpirun -np %s %s/sus -%s" % (int(np), susdir, algo)
    mpimsg = " (mpi %s proc)" % (int(np))

  if do_restart == "yes":
    print "Running restart test for %s%s on %s" % (testname, mpimsg, date())
  else:
    print "Running test %s%s on %s" % (testname, mpimsg, date())


  if do_restart == "yes":
    susinput = "-restart ../*.uda.000 -t 0 -move"
  else:
    susinput = "%s/%s" % (inputsdir, input(test))

  if mode == "dbg":
    if do_restart == "yes":
      malloc_stats_file = "restart_malloc_stats"
    else:
      malloc_stats_file = "malloc_stats"
    environ['MALLOC_STATS'] = malloc_stats_file

  rc = system("%s %s > sus.log 2>&1" % (command, susinput))

  if mode == "dbg":
    environ['MALLOC_STATS'] = "compare_uda_malloc_stats"

  if rc != 0:
    print "\t*** Test %s failed with code %d" % (testname, rc)
    if do_restart == "yes":
	print "\t\tMake sure the problem makes checkpoints before finishing"
    print "\tSee %s/sus.log for details" % (getcwd())
    return 1
  else:
    if do_restart == "yes":
      chdir("..")
      system("rm *.uda")
      system("cp -d --symbolic-link restart/*.uda .")
      chdir("restart")
    print "\tComparing udas on %s" % (date())
    replace_msg = "\tTo replace the gold standard uda and memory usage with these results,\n\trun: %s/replace_gold_standard" % (getcwd())
    rc = system("compare_sus_runs %s %s %s %s > compare_sus_runs.log 2>&1" % (testname, getcwd(), compare_root, susdir))
    if rc != 0:
	if rc == 5 * 256:
     	    print "\t*** Warning, %s has changed.  You must update the gold standard." % (input(test))
 	    print "%s" % replace_msg
	    return 1
	elif rc == 1 * 256:
    	    print "\t*** Warning, test %s failed uda comparison with error code %s" % (testname, rc)
    	    print "\tSee %s/compare_sus_runs.log for details" % (getcwd())
	    if do_restart != "yes":
 	    	print "%s" % replace_msg
	    return 1
	elif rc == 65280: # (-1 return code)
	    print "\tComparison tests passed.  (Note: No dat files to compare.)"
        else:
	    print "\tComparison tests passed.  (Note: No previous gold standard.)"
    else:
        print "\tComparison tests passed."

    if mode == "dbg":
	rc = system("mem_leak_check %s %s %s/%s/%s %s > mem_leak_check.log" % (testname, malloc_stats_file, compare_root, testname, malloc_stats_file, "."))
        if rc == 0:
	    print "\tMemory leak tests passed."
	elif rc == 5 * 256:
	    print "\t* Warning, no malloc_stats file created.  No memory leak test performed."
	elif rc == 256:
	    print "\t*** Warning, test %s failed memory leak test" % (testname)
	    print "\tSee %s/mem_leak_check.log" % (getcwd())
	    return 1
	elif rc == 2*256:
	    print "\t*** Warning, test %s failed memory highwater test" % (testname)
	    print "\tSee %s/mem_leak_check.log" % (getcwd())
 	    print "%s" % replace_msg
	    return 1
	else:
	    print "\tMemory leak tests passed. (Note: no previous memory usage stats)."

  return 0



