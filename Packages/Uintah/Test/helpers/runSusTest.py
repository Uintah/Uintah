#!/usr/local/bin/python

from os import environ,mkdir,path,system,chdir
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
  if mode == "mpi":
    command = "mpirun -np %s %s/sus -%s" % (num_processes(test), susdir, algo)
    datmode = "opt"
  else:
    command = "%s/sus -%s" % (susdir, algo)

  if do_restart == "yes":
    susinput = "-restart *.uda.000 -t 0 -move"
    log = "sus_restart.log"
  else:
    susinput = "%s/inputs/%s/%s" % (susdir, ALGO, input(test))
    log = "sus.log"

  rc = system("%s %s > %s 2>&1" % (command, susinput, log))
  if rc != 0:
    print "\t*** Test %s failed with code %d" % (testname, rc)
    return 1;
  else:
    print "\tComparing dat files"
    errors_to = environ['ERRORS_TO']
    if environ['ERRORMAIL'] != "yes":
	errors_to = ""
    rc = system("dat_test %s %s/%s-%s %s/Uintah_testdata/%s/%s-%s '%s' > dat_test.log 2>&1" % (testname, environ['BUILDROOT'], ALGO, mode, environ['BUILDROOT'], environ['OS'], ALGO, datmode, errors_to))
    if rc != 0:
	if rc != 65280:
    	    print "\t*** Warning, test %s failed dat comparision with error code %s" % (testname, rc)
	    return 1
	else:
	    print "\tNo dat files to compare."
    else:
	print "\tComparison tests passed."

  return 0



