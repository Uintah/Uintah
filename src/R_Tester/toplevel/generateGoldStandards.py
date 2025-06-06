#! /usr/bin/env python3

import os

# bulletproofing
if os.sys.version_info <= (3,0):
  print( "" )
  print( "ERROR: Your python version [" + str( os.sys.version_info ) + "] is too old.\n" + \
        "       You must use version 3.0 or greater. \n" )
  exit( 1 )

import shutil
import platform
import socket
import resource
import subprocess # needed to accurately get return codes

from os                       import system,environ,pathsep,path
from optparse                 import OptionParser
from sys                      import argv, exit
from helpers.runSusTests_git  import getTestName, getTestOS, getUpsFile, getMPISize, getTestOS, setInputsDir, getTestFlags, cmdline, isValid_inputFile
from helpers.runSusTests_git  import runPreProcessCmd, parse_addtlpath
from helpers.modUPS           import modUPS
from helpers.utilities        import writeDividerLine
####################################################################################

sus           = ""   # full path to sus executable
inputs        = ""   # full path to src/Standalone/inputs/
OS            = platform.system()
debug_build   = ""
no_sci_malloc = ""


####################################################################################

#script_dir=os.sys.path[0]

usage = "%prog [options]\n\n" \
        "   generateGoldStandards creates a sub-directory for the test_file.\n" \
        "   Note, multiple tests may be specified: -t ICE -t MPM etc."

parser = OptionParser( usage, add_help_option=False )

parser.set_defaults( verbose=False,
                     parallelism=1 )

parser.add_option( "-b", dest="build_directory",              help="Uintah build directory [REQUIRED]",
                   action="store", type="string" )

parser.add_option( "-d", dest="is_debug",                     help="Whether this is a debug build (use 'yes' or 'no')",
                   action="store", type="string" )

parser.add_option( "-h", "--help", action="help",             help="Show this help message" )

parser.add_option( "-j", type="int", dest="parallelism",      help="Set make parallelism" )

parser.add_option( "-m", dest="sci_malloc_on",                help="Whether this is build has sci-malloc turned on (use 'yes' or 'no')",
                   action="store", type="string" )

parser.add_option( "-s", dest="src_directory",                help="Uintah src directory [defaults to .../bin/../src]",
                   action="store", type="string" )

parser.add_option( "-t", dest="test_file",                    help="Name of specific test script (eg: ICE) [REQUIRED/Multiple allowed]",
                   action="append", type="string" )

parser.add_option( "-v", action="store_true", dest="verbose", help="Enable verbosity" )

####################################################################################

def error( error_msg ) :
    print( "" )
    print( "ERROR: " + error_msg )
    print( "" )
    parser.print_help()
    print( "" )
    exit( 1 )

####################################################################################

def validateArgs( options, args ) :
    global sus, inputs, debug_build, no_sci_malloc

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

    if not options.sci_malloc_on :
        error( "Whether this is build has sci-malloc turned on is not specified.  Please use <-m yes/no>." )
    else :
        if options.sci_malloc_on == "yes" :
            no_sci_malloc = False
        elif options.sci_malloc_on == "no" :
            no_sci_malloc = True
        else :
            error( "-d requires 'yes' or 'no'." )

    if not options.is_debug :
        error( "debug/optimized not specified.  Please use <-d yes/no>." )
    else :
        if options.is_debug == "yes" :
            debug_build = True
        elif options.is_debug == "no" :
            debug_build = False
        else :
            error( "-d requires 'yes' or 'no'." )

    if not options.src_directory :
        # Cut off the <bin> and replace it with 'src'
        last_slash = options.build_directory.rfind( "/" )
        options.src_directory = options.build_directory[0:last_slash] + "/src"

    if not os.path.isdir( options.src_directory ) :
        error( "Src directory '" + options.src_directory + "' does not exist." )

    inputs = options.src_directory + "/StandAlone/inputs"

    if not os.path.isdir( inputs ) :
        error( "'inputs' directory not found here: '" + inputs )

    setInputsDir( inputs )

####################################################################################

def generateGS() :

    global sus, inputs, debug_build, no_sci_malloc
    try :
        (options, leftover_args ) = parser.parse_args()
    except :
        print( "" ) # Print an extra newline at end of output for clarity
        exit( 1 )

    validateArgs( options, leftover_args )

    #__________________________________
    # define the maximum run time
    Giga = 2**30
    Kilo = 2**10
    Mega = 2**20
    #resource.setrlimit(resource.RLIMIT_AS, (90 * Mega,100*Mega) )  If we ever want to limit the memory

    if debug_build :
      maxAllowRunTime = 30*60   # 30 minutes
    else:
      maxAllowRunTime = 15*60   # 15 minutes

    resource.setrlimit(resource.RLIMIT_CPU, (maxAllowRunTime,maxAllowRunTime) )

    #__________________________________
    # Does mpirun command exist or has the environmental variable been set?
    try :
      MPIRUN = os.environ['MPIRUN']    # first try the environmental variable
    except :
      try:
        MPIRUN = shutil.which("mpirun")
      except:
        print( "ERROR:generateGoldStandards.py ")
        print( "      mpirun command was not found and the environmental variable MPIRUN was not set." )
        print( "      You must either add mpirun to your path, or set the 'MPIRUN' environment variable." )
        exit (1)

    print( "Using mpirun: %s " % MPIRUN )
    print( "If this is not the correct MPIRUN, please indicate the desired one with the MPIRUN environment variable" )

    if options.verbose :
        print( "Building Gold Standards in " + os.getcwd() )

    ##############################################################
    # Determine if the code has been modified (git status)
    cmd = "cd %s; git status" % options.src_directory

    (stdout,stderr,rc) = cmdline( cmd )

    if rc != 0 :
        answer = ""
        while answer != "n" and answer != "y" :
            print( "" )
            print( "WARNING:  'git status' failed to run correctly, so generateGoldStandards.py cannot tell" )
            print( "          if your tree is 'up to date'.  Are you sure you want to continue generating" )
            print( "          new gold standards at this time? [y/n]" )
            print( "" )

            answer = os.sys.stdin.readline()[:-1]
            if answer == "n" :
                print( "" )
                print( "Goodbye." )
                print( "" )
                exit( 0 )

    allComponentsTests = options.test_file

    # parse the inputs which are in the form ( <component>:<test> )
    components      = []          # list that contains each test name
    componentTests  = []          # contains the name list of tests to run


    #______________________________________________________________________
    #  Bulletproofing
    #  clean out old gold standards

    for component in allComponentsTests :
      me = component.split(':')
      c = me[0]
      t = me[1]
      components.append( c )
      componentTests.append( t )

    print( "\nComponents (%s), tests(%s) " % (components,componentTests) )


    # Exit if the component hasn't been compiled.  Note, not all components
    # are listed in the configVars.mk file
    configVars = options.build_directory + "/configVars.mk"
    for component in components :

      searchString = "BUILD_%s=no" % component.upper()  # search for BUILD_<COMPONENT>=no
      for line in open(configVars):
        if searchString in line:
          print( "\n ERROR: the component (%s) was not compiled.  You must compile it before you can generate the gold standards\n" % component )
          exit( 1 )

    # Warn user if directories already exist
    some_dirs_already_exist = False

    for component in components :
        if os.path.isdir( component ) :
            if not some_dirs_already_exist :
                some_dirs_already_exist = True
                print( "" )
                print( "Note, the following gold standards already exist: " )
            else :
                print( ", " )
            os.sys.stdout.write( component )

    if some_dirs_already_exist :
        answer = ""
        while answer != "n" and answer != "y" :
            print( "" )
            print( "Delete existing gold standards?  (If 'no', script will exit.) [y/n]" )
            answer = os.sys.stdin.readline()[:-1]
            if answer == "n" :
                print( "" )
                print( "Goodbye." )
                print( "" )
                exit( 0 )

        for component in components :
            if os.path.isdir( component ) :
                print( "Deleting " + component )
                shutil.rmtree( component )

    #__________________________________
    #
    counter = -1;
    for component in components :
        counter = counter + 1

        # Pull the list of tests from the the 'component's python module's 'TESTS' variable:
        # (Need to 'import' the module first.)
        if options.verbose :
            print( "Python importing " + component + ".py" )

        try :
          THE_COMPONENT = __import__( component )
        except :
          print( "" )
          print( "Error: loading the component '%s'." % component )
          print( "       Either that python file does not exist or there is a syntax error in the tests that have been defined.  Goodbye." )
          print( "" )
          exit( -1 )

        os.mkdir( component )
        os.chdir( component )

        #__________________________________
        # Create a symbolic link to the 'inputs' directory so some .ups files will be able
        # to find what they need...
        if not os.path.islink( "inputs" ) :
            os.symlink( inputs, "inputs" )

        # find the list of tests (local/nightly/debug/......)
        tests = THE_COMPONENT.getTestList( componentTests[counter] )

        #__________________________________
        # update path for preProcessing cmds
        susdir    = path.dirname(sus)
        addtlPath = THE_COMPONENT.ADDTL_PATH
        addtlPath = parse_addtlpath( addtlPath, susdir )

        environ['PATH']  =  addtlPath + pathsep + environ['PATH']
        print ( "path: %s" % environ['PATH'])

        if options.verbose :
            print( "" )
            print( "______________________________________________________________________" )
            print( "About to run tests for component: " + component )

        #__________________________________
        #     Loop over tcomponent tests

        nTestsFinished = 0
        for test in tests :
            if getTestOS( test ) != OS.upper() and getTestOS( test ) != "ALL":
                continue

            #  Defaults
            sus_options    = ""
            preProcess_cmd = ""
            do_restart     = 1
            do_gpu         = 0    # run test if gpu is supported
            testname       = getTestName( test )
            upsFile        = inputs + "/" + component + "/" + getUpsFile( test )

            #__________________________________
            # parse user flags for the gpu and sus_options
            # override defaults if the flags have been specified
            print( "__________________" )

            if len(test) == 5:
              flags = getTestFlags(test)
              print( "User Flags:" )

              #  parse the user flags
              for i in range(len(flags)):
                if flags[i] == "gpu":
                  do_gpu = 1

                if flags[i] == "no_restart":
                  do_restart = 0

                tmp = flags[i].rsplit('=')
                if tmp[0] == "sus_options":
                  sus_options = tmp[1]
                  print( "sus_option: %s \n"%(sus_options) )

                if tmp[0] == "preProcessCmd":
                  preProcess_cmd = tmp[1]
                  print( "preProcess_cmd: %s \n"%(preProcess_cmd) )

            #__________________________________
            # check if code/machine is gpu enabled
            if do_gpu == 1:

              print( "Running command to see if GPU is active: " + sus + " -gpucheck" )

              (stdout,stderr,rc) = cmdline( "%s -gpucheck" % sus)

              if rc == 1:
                print( "GPU found!" )
              else:
                print( "\nWARNING: skipping this test.  This machine is not configured to run gpu tests\n" )
                continue

            print( "About to run test: " + testname )

            #__________________________________
            #
            if isValid_inputFile( upsFile, "null", do_restart ) == False:
              print ("    Now skipping test %s \n" % testname)
              continue

            os.mkdir( testname )
            os.chdir( testname )

            # Create (yet) another symbolic link to the 'inputs' directory so some .ups files will be able
            # to find what they need...  (Needed for, at least, methane8patch (ARCHES) test.)
            if not os.path.islink( "inputs" ) :
                os.symlink( inputs, "inputs" )


            MALLOC_FLAG = None

            if debug_build :
                if no_sci_malloc :
                    print( "" )
                    print( "WARNING!!! The build was not built with SCI Malloc on...  Memory tests will not be run." )
                    print( "WARNING!!! If you wish to perform memory checks, you must re-configure your debug build" )
                    print( "WARNING!!! with '--enable-sci-malloc', run 'make cleanreally', and re-compile everything." )
                    print( "" )
                else :
                    os.environ['MALLOC_STRICT'] = "set"
                    os.environ['MALLOC_STATS'] = "malloc_stats"
                    MALLOC_FLAG = " -x MALLOC_STATS "

            GIT_FLAGS = " -gitStatus -gitDiff "
            #GIT_FLAGS = "" # When debugging, if you don't want to spend time waiting for git, uncomment this line.

            #__________________________________
            #  run preprocess command on ups file
            sus_log_msg = '\tSee sus.log.txt for details'
            rc=runPreProcessCmd( preProcess_cmd, upsFile, 'sus_log.txt', sus_log_msg )
            if rc==1:
                os.chdir( ".." )
                continue;

            #__________________________________
            MPIHEAD="%s -n" % MPIRUN       #default

            # pass in environmental variables to mpirun
            if MALLOC_FLAG is not None:

              if environ['OS'] == "Linux":
                MPIHEAD="%s %s -n" % (MPIRUN, MALLOC_FLAG)

                                             # openmpi
              rc = system("%s -x TERM echo 'hello' > /dev/null 2>&1" % MPIRUN)
              if rc == 0:
                MPIHEAD="%s %s -n" % (MPIRUN, MALLOC_FLAG)

                                             #  mvapich and mpich
              rc = system("%s -genvlist TERM echo 'hello' > /dev/null 2>&1" % MPIRUN)
              if rc == 0:
                MPIHEAD="%s -genvlist MALLOC_STATS -n" % MPIRUN

            #__________________________________
            #  Run sus and check return codes
            np = int( getMPISize( test ) )
            my_mpirun = "%s %s  " % (MPIHEAD, np)

            command = my_mpirun + sus + " " + GIT_FLAGS + " " + sus_options + " " + upsFile  + " >> sus_log.txt 2>&1 "

            print( "Running command: " + command )

            rc = os.system( command )

            print( "\t*** Test return code %i" % rc )
            # catch if sus doesn't run to completion

            if rc == 35072 or rc == 36608 :
              print( "\t*** Test exceeded maximum allowable run time ***" )
              print( "" )

            if rc != 0:
              print( "\nERROR: %s: Test (%s) failed to complete\n" % (component,test) )

            os.chdir( ".." ) # Back to the component (eg: 'ICE') directory
            nTestsFinished += 1

        if nTestsFinished == 0:
          print( "\nERROR: Zero %s regression tests ran to completion.  Hint: is the OS you're using appropriate for the component's tests?  \n" % (component) )

        os.chdir( ".." ) # Back to the TestData directory


####################################################################################

if __name__ == "__main__":
    generateGS()
    exit( 0 )
