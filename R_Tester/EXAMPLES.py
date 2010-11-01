#!/usr/bin/python

from sys import argv,exit
from os import environ
from helpers.runSusTests import runSusTests

#______________________________________________________________________

NIGHTLYTESTS = [   ("poisson1", "poisson1.ups", 1, "ALL") ]

# Tests that are run during local regression testing
LOCALTESTS   = [   ("poisson1", "poisson1.ups", 1, "ALL") ]

UNUSED_TESTS = [   ("poisson2", "poisson2.ups", 1) ]

#__________________________________

def getNightlyTests() :
  return NIGHTLYTESTS

def getLocalTests() :
  return LOCALTESTS

#__________________________________

if __name__ == "__main__":
  if environ['LOCAL_OR_NIGHTLY_TEST'] == "local":
    TESTS = LOCALTESTS
  else:
    TESTS = NIGHTLYTESTS
  result = runSusTests(argv, TESTS, "Examples")
  exit( result )
