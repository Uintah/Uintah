#!/usr/local/bin/python

from os import environ,putenv,mkdir,path,system,chdir
from time import asctime,localtime,time
from string import upper

def input (test):
    return test[0]
def modes (test):
    return test[1]
def num_processes (test):
    return test[2]
def date ():
    return asctime(localtime(time()))

def runSusTest(test, mode, susdir, algo, do_restart = "no"):
  datmode = mode
  ALGO = upper(algo)
  testname = path.splitext(input(test))[0];

  if do_restart == "yes":
    print "%s-%s: Running restart test for %s on %s" % (ALGO, mode, testname, date())
  else:
    print "%s-%s: Running test %s on %s" % (ALGO, mode, testname, date())
  if mode in ('mpi', 'dbgmpi'):
    command = "mpirun -np %s %s/sus -%s" % (num_processes(test), susdir, algo)
    if mode == "mpi":
    	datmode = "opt"
    else:
	datmode = "dbg"
  else:
    command = "%s/sus -%s" % (susdir, algo)

  if do_restart == "yes":
    susinput = "-restart *.uda.000 -t 0 -move"
    log = "sus_restart.log"
  else:
    susinput = "%s/inputs/%s/%s" % (susdir, ALGO, input(test))
    log = "sus.log"

  if datmode == "dbg":
    if d_restart == "yes":
      environ['MALLOC_STATS'] = "restart_malloc_stats"
    else:
      environ['MALLOC_STATS'] = "malloc_stats"

  test_root = "%s/%s-%s" % (environ['BUILDROOT'], ALGO, mode)
  compare_root = "%s/%s-%s" % (environ['TEST_DATA'], ALGO, datmode)

  rc = system("%s %s > %s 2>&1" % (command, susinput, log))

  if datmode == "dbg":
    environ['MALLOC_STATS'] = "compare_uda_malloc_stats"

  if rc != 0:
    print "\t*** Test %s failed with code %d" % (testname, rc)
    if do_restart == "yes":
	print "\t*** Make sure the problem makes checkpoints before finishing"
    return 1
  else:
    print "\tComparing udas on %s" % (date())
    errors_to = environ['ERRORS_TO']
    if environ['ERRORMAIL'] != "yes":
	errors_to = ""
    rc = system("compare_sus_runs %s %s %s %s '%s' > compare_sus_runs.log 2>&1" % (testname, test_root, compare_root, susdir, errors_to))
    if rc != 0:
	if rc != 65280:
    	    print "\t*** Warning, test %s failed uda comparison with error code %s" % (testname, rc)
	    return 1
	else:
	    print "\tComparison tests passed.  (Note: No dat files to compare.)"
    else:
        print "\tComparison tests passed."

    if mode in ('dbg', 'dbgmpi'):
	rc = system("mem_leak_check %s %s %s/%s/%s %s %s > mem_leak_check.log" % (testname, environ['MALLOC_STATS'], compare_root, testname, environ['MALLOC_STATS'], ".", errors_to))
        if rc == 0:
	    print "\tMemory leak tests passed."
	elif rc == 5 * 256:
	    print "\t*** Warning, no malloc_stats file created.  Memory leak test failed."
	    return 1
	elif rc == 256:
	    print "\t*** Warning, test %s failed memory leak test" % (testname)
	    return 1
	else:
	    print "\tMemory leak tests passed (Note: no highwater comparison made)."

  return 0



