# -*- python -*-
# ex: set syntax=python:

import numpy as np
import os
import shutil
import glob
import re                # regular expressions
import pprint
# 
from buildbot.plugins               import *
from buildbot.process               import results
from buildbot.process.results       import FAILURE
from buildbot.process.results       import SKIPPED
from buildbot.process.results       import statusToString
# from datetime                       import timedelta
from buildbot.process.build         import Build
#______________________________________________________________________
#
#  Common steps for the factories


m_git_url        ='https://github.com/Uintah/Uintah.git'

#______________________________________________________________________
#   returns value from either input argv or the property dictionary

def getValue( Q, Qname):
    Q_prop = util.Property( Qname )

    # bulletproofing
    if Q == None and Q_prop == None:
      print( "ERROR the variable %s has not been set!!! " % Qname )

    if Q == None:
      return  util.Property( Qname )
    else:
      return Q

#______________________________________________________________________
#   returns value from either input argv or the property dictionary

def getValueWithDefault( Q_default, Qname):
    Q = util.Interpolate('%(prop:'+Qname+')s') 

    if Q == None:
      return  Q_default
    else:
      return Q

#______________________________________________________________________
#   Update the branch code and remove doc/ directory
def gitUpdate( factory ):

    gitPoller = steps.Git(
                     repourl          = m_git_url,
                     branch           = util.Property( 'myBranch' ),
                     workdir          = 'build/',
                     mode             = "incremental",
                     shallow          = True,
                     retry            = (10,2),
                     clobberOnFailure = True,
                     warnOnFailure    = True )

    rmDocDir = steps.ShellCommand(
                     description   = [" remove doc/ directory"],
                     name          = "rm -rf doc",
                     command       = ["/bin/sh", "-c", "/bin/rm -rf doc/"],
                     workdir       = 'build/',
                     logEnviron    = False,
                     warnOnWarnings= True,
                     warnOnFailure = True,
                     haltOnFailure = True )

    factory.addStep( gitPoller)
    factory.addStep( rmDocDir)

#______________________________________________________________________
#   configure step
def configure( factory, wrkDir=None, configCmd=None):

    configCmd = getValue( configCmd, 'configCmd' )
    workDir   = getValue( wrkDir,     'wrkDir' )

    rmBuildDir = steps.ShellCommand(
                     description   = ["removing build directory"],
                     name          = "remove build directory",
                     command       = ["/bin/sh", "-c", "/bin/rm -rf *"],
                     workdir       = workDir,
                     hideStepIf    = False,
                     logEnviron    = False,
                     warnOnWarnings= True,
                     warnOnFailure = True )

    configCmd = steps.Configure(
                     command        = configCmd,
                     env            = util.Property( 'compiler_env' ),
                     workdir        = workDir,
                     warnOnFailure  = True,
                     haltOnFailure  = True )

    factory.addStep( rmBuildDir )
    factory.addStep( configCmd )

#______________________________________________________________________
#   compile step
def compile( factory, numProcs, wrkDir=None ):

    workDir = getValue( wrkDir, 'wrkDir' )

    runMake = steps.Compile(
                     command        = ["python", "../src/scripts/buildbot/make.py", " 16"],
                     workdir        = workDir,
                     logEnviron     = False,
                     warnOnFailure  = True,
                     haltOnFailure  = False)

    factory.addStep( runMake )


#______________________________________________________________________
# determine if a step has failed return true if one has.
def hasStepFailed(step):

    myResults = statusToString( step.build.results )

    print("__________________________________hasStepFailed: %s" %(myResults) )

    if  myResults == "failure" or myResults == "exception":
      return True
    else:
      return False

    return False

#__________________________________
def skipped(results, s):
    return (results ==  SKIPPED)


#______________________________________________________________________
#   run make cleanrelly
def makeclean( factory, wrkDir= None ):

  wrkDir = getValue( wrkDir, 'wrkDir' )

  makeClean = steps.ShellCommand(
                   description   = ["make clean"],
                   name          = "make clean",
                   command       = ["make", "reallyclean"],
                   workdir       = util.Property( 'wrkDir' ),
                   logEnviron    = False,
                   warnOnWarnings= True,
                   warnOnFailure = True )

  factory.addStep( makeClean )

#______________________________________________________________________
#   copy build directory to appropriate /data  directory
def copyBuildDir( factory ):

  mkdir = steps.MakeDirectory(
                   dir        = util.Interpolate( '/data/buildbot/%(prop:buildername)s/%(prop:buildnumber)s/' ),
                   name       ="mkdir",
                   doStepIf   = hasStepFailed,
                   hideStepIf = skipped )

  copyDir = steps.CopyDirectory(
                   src        = "build",
                   dest       = util.Interpolate( '/data/buildbot/%(prop:buildername)s/%(prop:buildnumber)s/' ),
                   name       = "cp build ",
                   doStepIf   = hasStepFailed,
                   hideStepIf = skipped )

  changePerms = steps.ShellCommand(
                   description   = ["change permissions"],
                   name          = "Change permissions",
                   command       = ["/bin/sh", "-c", "chgrp -R users . ; chmod -R g+rwX ."],
                   doStepIf      = hasStepFailed,
                   hideStepIf    = skipped ,
                   workdir       = util.Interpolate( '/data/buildbot/%(prop:buildername)s/%(prop:buildnumber)s/' ),
                   logEnviron    = False,
                   warnOnWarnings= True,
                   warnOnFailure = True )


  factory.addStep( mkdir )
  factory.addStep( copyDir )
  factory.addStep( changePerms )



#______________________________________________________________________
#   configure step
def rm_localRT_dir( factory, wrkDir=None,):

    workDir   = getValue( wrkDir, 'wrkDir' )

    rm_localRT_Dir = steps.ShellCommand(
                     description   = ["removing localRT directory"],
                     name          = "remove localRT directory",
                     command       = ["/bin/sh", "-c", "/bin/rm -rf local_RT"],
                     workdir       = workDir,
                     hideStepIf    = False,
                     logEnviron    = False,
                     warnOnWarnings= True,
                     warnOnFailure = True )

    factory.addStep( rm_localRT_Dir )

#______________________________________________________________________

def runComponentTests(factory):

    runLocalRT = steps.ShellCommand(
                     description    = ["Running  tests"],
                     command        = ["make", "runLocalRT"],
                     workdir        = util.Property( 'wrkDir' ),
                     timeout        = 60*60,  # timeout after 1 hour
                     name           = "localRT_test",
                     warnOnWarnings = True,
                     warnOnFailure  = True,
                     haltOnFailure  = False )

    factory.addStep( runLocalRT )
    
#______________________________________________________________________
#  Utilities:
#_______________________________________________________________________
#  Logic for determining if the buildBot should be run
def runOrSkip_BuildBot( change ):

    #__________________________________
    #   skip if the string "skipRT" or "skip-rt" or skip_rt" is in the comments
    LC_comments = change.comments.lower()  # convert comments to lower case

    if "skiprt" in LC_comments or "skip_rt" in LC_comments  or "skip_rt" in LC_comments:
        print ("  runOrSkip_BuildBot:  __________________________________ Skipping buildbot")
        return False

    #__________________________________
    # skip if a changed file lives in these dirs
    ignoreDirs = ['doc',
                'src/parametricStudies',
                'src/scripts',
                'src/testprograms',
                'src/CCA/Components/FVM',
                'src/CCA/Components/Arches/Attic',
                'src/CCA/Components/ICE/Docs',
                'src/CCA/Components/ICE/Matlab',
                'src/CCA/Components/ICE/PressureSolve',
                'src/CCA/Components/MPMICE/Docs',
                'src/CCA/Components/PhaseField'
              ]

    # if the path of a committed files is inside of ignoreDirs[*] then tag it
    tag = np.array([], int)
    count = -1

    # loop over all changed files
    for name in change.files:
      count +=1

      tag = np.append(tag, 0)             # default
      path1 = os.path.dirname( name )

      #__________________________________
      # Logic
      for pattern in ignoreDirs:

        test = re.match(pattern, path1)

        if test :
          tag[count] = 1
          break

    # how many of the commited files are tagged
    mySum = np.sum( tag )

    # every file must be tagged to skip the tests
    if (mySum == tag.size ):
      print ("  runOrSkip_BuildBot:  __________________________________ Skipping buildbot")
      return False
    else:
      return True
