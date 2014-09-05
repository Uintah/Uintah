#! /usr/bin/env python
import os
import shutil
import platform
from optparse import OptionParser
from sys import argv, exit
from string import upper


# bulletproofing
if os.sys.version_info <= (2,4):
  print ""
  print "ERROR: Your python version [" + str( os.sys.version_info ) + "] is too old.\n" + \
        "       You must use version 2.5 or greater (but NOT version 3.x!). \n\n"
  print ""
  exit( 1 )

import subprocess

from helpers.runSusTests import nameoftest, testOS, input, num_processes, testOS, setGeneratingGoldStandards

####################################################################################

sus    = ""   # full path to sus executable
inputs = ""   # full path to src/Standalone/inputs/
OS     = platform.system()

####################################################################################

#script_dir=os.sys.path[0]

usage = "%prog [options]\n\n" \
        "   generateGoldStandards creates a sub-directory for the test_file.\n" \
        "   Note, multiple tests may be specified: -t ICE -t MPM etc."

parser = OptionParser( usage, add_help_option=False )

parser.set_defaults( verbose=False, 
                     parallelism=1 )

parser.add_option( "-v", action="store_true", dest="verbose", help="Enable verbosity" )
parser.add_option( "-j", type="int", dest="parallelism",      help="Set make parallelism" )
parser.add_option( "-h", "--help", action="help",             help="Show this help message" )

parser.add_option( "-t", dest="test_file",                    help="Name of specific test script (eg: ICE) [REQUIRED/Multiple allowed]",
                   action="append", type="string" )

parser.add_option( "-b", dest="build_directory",              help="Uintah build directory [REQUIRED]",
                   action="store", type="string" )

parser.add_option( "-s", dest="src_directory",                help="Uintah src directory [defaults to .../bin/../src]",
                   action="store", type="string" )

####################################################################################

def error( error_msg ) :
    print ""
    print "ERROR: " + error_msg
    print ""
    parser.print_help()
    print ""
    exit( 1 )

####################################################################################

def validateArgs( options, args ) :
    global sus, inputs

    if len( args ) > 0 :
        error( "Unknown command line args: " + str( args ) )

    if not options.build_directory :
        error( "Uintah build directory is required..." )
    elif options.build_directory[0] != "/" : 
        error( "Uintah build directory must be an absolute path (ie, it must start with '/')." )
    elif options.build_directory[-1] == "/" : 
        # Cut off the trailing '/'
        options.build_directory = options.build_directory[0:-1]

    if not options.test_file :
        error( "A test file must be specified..." )

    if not os.path.isdir( options.build_directory ) :
        error( "Build directory '" + options.build_directory + "' does not exist." )

    sus = options.build_directory + "/StandAlone/sus"

    if not os.path.isfile( sus ) :
        error( "'sus' not here: '" + sus + "'" )

    if not options.src_directory :
        # Cut off the <bin> and replace it with 'src'
        last_slash = options.build_directory.rfind( "/" )
        options.src_directory = options.build_directory[0:last_slash] + "/src"

    if not os.path.isdir( options.src_directory ) :
        error( "Src directory '" + options.src_directory + "' does not exist." )

    inputs = options.src_directory + "/StandAlone/inputs"

    if not os.path.isdir( inputs ) :
        error( "'inputs' directory not found here: '" + inputs )

    setGeneratingGoldStandards( inputs )

####################################################################################

def generateGS() :

    global sus, inputs
    try :
        (options, leftover_args ) = parser.parse_args()
    except :
        print "" # Print an extra newline at end of output for clarity
        exit( 1 )

    validateArgs( options, leftover_args )
    
    #__________________________________
    # Does mpirun command exist or has the environmental variable been set?
    try :
      MPIRUN = os.environ['MPIRUN']    # first try the environmental variable
    except :
      MPIRUN = "mpirun"
      rc = os.system("which mpirun>&/dev/null")

      if rc == 256:
        print "ERROR:generateGoldStandards.py "
        print "      mpirun command was not found and the environmental variable MPIRUN was not set."
        print "      You must either put mpirun in your path or set the environmental variable"
        exit (1)
    print "Using mpirun: %s " % MPIRUN
        
    if options.verbose :
        print "Building Gold Standards in " + os.getcwd()

    ##############################################################
    # Determine if the code has been modified (svn stat)

    process = subprocess.Popen( "svn stat " + options.src_directory, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE )
    ( stdout, sterr ) = process.communicate()
    result = process.returncode

    if result != 0 :
        answer = ""
        while answer != "n" and answer != "y" :
            print ""
            print "WARNING:  SVN 'stat' failed to run correctly, so generateGoldStandards.py cannot tell"
            print "          if your tree is 'up to date'.  Are you sure you want to continue generating"
            print "          new gold standards at this time? [y/n]"
            print ""

            answer = os.sys.stdin.readline()[:-1]
            if answer == "n" :
                print ""
                print "Goodbye."
                print ""
                exit( 0 )
    #
    # !!!FIXME!!!: if svn fails to run, or returns differences, then
    # ask the user what they want to do.  This is not complete yet!!!
    ##############################################################

    ##############################################################
    # !!!FIXME!!!: Determine if configVars has changed... if not, let the user
    # know this...
    #
    #configVars = options.build_directory + "/configVars.mk"
    #shutil.copy( configVars, "." )
    ##############################################################

    ##############################################################
    # !!!FIXME!!!:
    # - Determine if/where mpirun is...
    # - Determine if sus was built with MPI...
    # - Determine (ask the user?) if the (sus) binary is up to date.
    ##############################################################

    components = options.test_file
    
    # Exit if the component hasn't been compiled.  Note, not all components
    # are listed in the configVars.mk file 
    configVars = options.build_directory + "/configVars.mk"
    for component in components :
      
      searchString = "BUILD_%s=no" % upper(component)  # search for BUILD_<COMPONENT>=no
      for line in open(configVars):
        if searchString in line:
          print "\n ERROR: the component (%s) was not compiled.  You must compile it before you can generate the gold standards\n" % component
          exit( 1 ) 

    # Warn user if directories already exist
    some_dirs_already_exist = False

    for component in components :
        if os.path.isdir( component ) :
            if not some_dirs_already_exist :
                some_dirs_already_exist = True
                print ""
                print "Note, the following gold standards already exist: ",
            else :
                print ", ",
            os.sys.stdout.write( component )

    if some_dirs_already_exist :
        answer = ""
        while answer != "n" and answer != "y" :
            print ""
            print "Delete existing gold standards?  (If 'no', script will exit.) [y/n]"
            answer = os.sys.stdin.readline()[:-1]
            if answer == "n" :
                print ""
                print "Goodbye."
                print ""
                exit( 0 )

        for component in components :
            if os.path.isdir( component ) :
                print "Deleting " + component
                shutil.rmtree( component )

    for component in components :

        os.mkdir( component )
        os.chdir( component )

        # Create a symbolic link to the 'inputs' directory so some .ups files will be able
        # to find what they need...
        if not os.path.islink( "inputs" ) :
            os.symlink( inputs, "inputs" )

        # Pull the list of tests from the the 'component's python module's 'TESTS' variable:
        # (Need to 'import' the module first.)
        if options.verbose :
            print "Python importing " + component + ".py"
        THE_COMPONENT = __import__( component )

        # determine which tests (local/nightly) to run default is local
        whichTests = os.getenv( 'WHICH_TESTS', "local" )
        
        print "Which_tests: %s " % whichTests
        if whichTests == "local" :
            tests = THE_COMPONENT.getLocalTests()
        elif whichTests == "nightly" :
            tests = THE_COMPONENT.getNightlyTests()
        else :
            print "\nThe environmental variable WHICH_TESTS:(%s) is not valid" % whichTests
            print "the valid options are local or nightly. \n"
            exit (-1)
          
                  
        if options.verbose :
            print "About to run tests for: " + component

        for test in tests :
            if testOS( test ) != upper( OS ) and testOS( test ) != "ALL":
                continue
              
            # FIXME: NOT SURE IF THIS IS RIGHT, BUT IT APPEARS TO MATCH WHAT THE RUN TESTS SCRIPT NEEDS:
            print "About to run test: " + nameoftest( test )
            os.mkdir( nameoftest( test ) )
            os.chdir( nameoftest( test ) )

            # Create (yet) another symbolic link to the 'inputs' directory so some .ups files will be able
            # to find what they need...  (Needed for, at least, methane8patch (ARCHES) test.)
            if not os.path.islink( "inputs" ) :
                os.symlink( inputs, "inputs" )

            np = float( num_processes( test ) )
            mpirun = ""
            if np > 1.0 :
                np = int( np )
                mpirun = "%s -np %s  " % (MPIRUN,np)

                command = mpirun + sus + " -svnStat -svnDiff " + inputs + "/" + component + "/" + input( test )  + " >> sus_log.txt " 
            else :
                command = sus + " -svnStat -svnDiff " + inputs + "/" + component + "/" + input( test )  + " >> sus_log.txt " 

            print "Running command: " + command

            rc = os.system( command )
            
            # catch if sus doesn't run to completion
            if rc != 0:
              print "\nERROR: %s: Test (%s) failed to complete\n" % (component,test)
              exit(-1)
            
            os.chdir( ".." ) # Back to the component (eg: 'ICE') directory

        os.chdir( ".." ) # Back to the TestData directory


####################################################################################

if __name__ == "__main__":
    generateGS()
    exit( 0 )
