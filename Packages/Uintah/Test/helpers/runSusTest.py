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
    environ['MALLOC_STATS'] = "malloc_stats"

  rc = system("%s %s > %s 2>&1" % (command, susinput, log))
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
    rc = system("compare_sus_runs %s %s/%s-%s %s/%s-%s %s '%s' > compare_sus_runs.log 2>&1" % (testname, environ['BUILDROOT'], ALGO, mode, environ['TEST_DATA'], ALGO, datmode, susdir, errors_to))
    if rc != 0:
	if rc != 65280:
    	    print "\t*** Warning, test %s failed uda comparison with error code %s" % (testname, rc)
	    return 1
	else:
	    print "\tComparison tests passed.  (Note: No dat files to compare.)"
    else:
        print "\tComparison tests passed."

    if mode in ('dbg', 'dbgmpi'):
	rc = system("mem_leak_check %s %s %s %s" % (testname, "malloc_stats", ".", errors_to))
        if rc == 0:
	    print "\tMemory leak test (only tests scinew leaks) passed."
	elif rc == 5:
	    print "\t*** Warning, no malloc_stats file created.  Memory leak test failed."
	    return 1
	else:
	    print "\t*** Warning, test %s failed memory leak test" % (testname)
	    return 1

  return 0



