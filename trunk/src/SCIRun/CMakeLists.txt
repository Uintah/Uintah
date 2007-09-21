#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#

###############################################################################
# Don't allow the user to build in the src directory...
#
IF(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  MESSAGE( "" )
  MESSAGE( "Building SCIRun in the source directory is not allowed." )
  MESSAGE( "(In order to clean up, you now need to run: rm -rf CMakeFiles CMakeCache.txt" )
  MESSAGE( "" )
  MESSAGE( FATAL_ERROR "Goodbye." )
ENDIF(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})

###############################################################################
# This is the root SCIRun CMakeLists file.
IF(WIN32)
  CMAKE_MINIMUM_REQUIRED(VERSION 2.4.4)

  # for the moment, force the user to select the build type here.  Maybe 
  # we can enable all types again if dynamic compilation goes away.
  # This needs to be before PROJECT, or changing the value will append to CMAKE_CONFIGURATION_TYPES, instead
  # of replacing it. - BW
  SET(BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
  SET(CMAKE_CONFIGURATION_TYPES ${BUILD_TYPE} CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
ELSE(WIN32)
  CMAKE_MINIMUM_REQUIRED(VERSION 2.2.3)
ENDIF(WIN32)

PROJECT(SCIRUN_CORE)

#-----------------------------------------------------------------------------
# Test for some required system information.
INCLUDE (CMakeBackwardCompatibilityC)

#-----------------------------------------------------------------------------
# SCIRUN version number.  Must reflect the value of SCIRUN_RCFILE_VERSION
# in src/scirunrc.  Also, make sure it matches the number in src/main/sci_version.h,
# and in src/main/scirun.rc.

SET(SCIRUN_VERSION_MAJOR "3")
SET(SCIRUN_VERSION_MINOR "1")
SET(SCIRUN_VERSION_PATCH "0")

SET(SCIRUN_VERSION_STRING "${SCIRUN_VERSION_MAJOR}.${SCIRUN_VERSION_MINOR}.${SCIRUN_VERSION_PATCH}")

#-----------------------------------------------------------------------------
# Configure testing support.  This should be done before any
# MESSAGE(FATAL_ERROR ...) commands are invoked.
INCLUDE( CTest )
IF(BUILD_TESTING)
  ENABLE_TESTING()
  SET(BUILDNAME "${BUILDNAME}" CACHE STRING "Name of build on the dashboard")
  MARK_AS_ADVANCED(BUILDNAME)
ENDIF(BUILD_TESTING)

SET(SCIRUN_CORE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

#----------------------------------------------------------------------------
# shared/static
OPTION(BUILD_SHARED_LIBS "Build shared libraries" ON)
MARK_AS_ADVANCED(BUILD_SHARED_LIBS)

#----------------------------------------------------------------------------
# Output directories.
SET (LIBRARY_OUTPUT_PATH ${SCIRUN_CORE_BINARY_DIR}/lib CACHE INTERNAL "Single output directory for building all libraries.")
SET (EXECUTABLE_OUTPUT_PATH ${SCIRUN_CORE_BINARY_DIR} CACHE INTERNAL "Single output directory for building all executables.")

#  FIX ME: NOT SURE ABOUT THIS: SET(CXX_TEST_PATH ${EXECUTABLE_OUTPUT_PATH})  
#      AND THE FOLLOWING LINE... which one should be used?

MARK_AS_ADVANCED(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)
#MARK_AS_ADVANCED(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH CXX_TEST_PATH)

SET(BUILD_FROM_TOP_LEVEL 1)

IF(NOT WIN32 AND NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF(NOT WIN32 AND NOT CMAKE_BUILD_TYPE)

IF(CMAKE_BUILD_TYPE STREQUAL Debug)
  SET(CMAKE_VERBOSE_MAKEFILE on)  
ENDIF(CMAKE_BUILD_TYPE STREQUAL Debug)

IF(CMAKE_C_COMPILER STREQUAL gcc)
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
ENDIF(CMAKE_C_COMPILER STREQUAL gcc)


################################################################################################
# Number of Bits (32 | 64) Check
#
# DV Fix checking bit of thirdparty with size of pointer check
# sci_defs/bits_defs.h
# DV clean up (if user tries to configure 64 bit machine to 32, flag error?)
# DV also maybe need to add binary flags here
# Set NBITS (64 or 32) based on the size of a void pointer.
# Handle if machine defaults to one bit size but user is trying to do another
IF(CMAKE_SIZEOF_VOID_P MATCHES  8)
  SET(NBITS 64 CACHE INTERNAL "64 or 32 bit build")
ELSE(CMAKE_SIZEOF_VOID_P MATCHES 8)
  SET(NBITS 32 CACHE INTERNAL "64 or 32 bit build")
ENDIF(CMAKE_SIZEOF_VOID_P MATCHES 8)
SET (SCIRUN_ENABLE_64BIT 0 CACHE BOOL "Enable 64 bit build.")
IF(${SCIRUN_ENABLE_64BIT})
  SET(NBITS 64 CACHE INTERNAL "64 bit build")
ENDIF(${SCIRUN_ENABLE_64BIT})
MARK_AS_ADVANCED(SCIRUN_ENABLE_64BIT)

################################################################################################
# Under OSX, use single_module (as opposed to multiple_module) linking.

if(APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "-single_module" CACHE STRING "Flags used to link a shared library." FORCE)
endif(APPLE)

################################################################################################
# SCIRun Thirdparty Location/Validation Check
#
#   Force user to use our Thirdparty, and verify for looking for
#   SCIRUN_THIRDPARTY_VERSION file

IF(NOT SCIRUN_THIRDPARTY_DIR)
  IF(NOT WIN32)
    # Check in default build location, specified by of ../thirdparty.src/install_command.txt
    FIND_PATH(SCIRUN_THIRDPARTY_STANDARD_INSTALL install_command.txt ../thirdparty.src/)

    IF(SCIRUN_THIRDPARTY_STANDARD_INSTALL)

      # Grab full path from install_command.txt
      FILE(READ "${SCIRUN_THIRDPARTY_STANDARD_INSTALL}/install_command.txt" 3p_full_path_string)
      STRING(REGEX MATCH "SCIRUN_THIRDPARTY_DIR=.+/" 3p_full_path_line ${3p_full_path_string})
      STRING(REGEX MATCH "[^SCIRUN_THIRDPARTY_DIR=].+" 3p_path ${3p_full_path_line})
      SET(SCIRUN_THIRDPARTY_DIR ${3p_path} CACHE PATH "Path to SCIRun Thirdparty")
      MESSAGE(STATUS "Found and using SCIRun Thirdparty ${SCIRUN_THIRDPARTY_DIR}")

    ELSE(SCIRUN_THIRDPARTY_STANDARD_INSTALL)

      # Figure out if SCIRun is in the standard /usr/sci location: 
      SET(COMPILER_VERS unknown)
      SET(COMPILER      unknown)

      IF(CMAKE_COMPILER_IS_GNUCXX)

          ## Currently only know how to use gcc... :(
          SET( COMPILER gcc )

          ## Pipe "gcc --version" to "head -1" (first line) and then do a regular expression look for #.#.#
          EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                          COMMAND head -1
                          OUTPUT_VARIABLE COMPILER_VERS )
          IF( COMPILER_VERS )
              STRING( REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" COMPILER_VERS ${COMPILER_VERS} )
          ENDIF( COMPILER_VERS )

      ENDIF(CMAKE_COMPILER_IS_GNUCXX) 

      INCLUDE( CMake/sciThirdparty.cmake )
      IF( COMPILER_VERS )
          SEARCH_FOR_SYSTEM_3P( "${SCIRUN_VERSION_MAJOR}.${SCIRUN_VERSION_MINOR}.${SCIRUN_VERSION_PATCH}"
                                ${COMPILER}
                                ${COMPILER_VERS}
                                ${NBITS} )
      ENDIF( COMPILER_VERS )

    ENDIF(SCIRUN_THIRDPARTY_STANDARD_INSTALL)

  ELSE(NOT WIN32)
      ## Check environment variable SCIRUN_THIRDPARTY_DIR for 3p
      IF( NOT SCIRUN_THIRDPARTY_DIR )
          SET(SCIRUN_THIRDPARTY_DIR_ENV "$ENV{SCIRUN_THIRDPARTY_DIR}")
          IF(SCIRUN_THIRDPARTY_DIR_ENV)
              MESSAGE(STATUS "Using Thirdparty directory specification in shell environment..." )
              SET(SCIRUN_THIRDPARTY_DIR ${SCIRUN_THIRDPARTY_DIR_ENV} CACHE PATH "Path to SCIRun's Thirdparty")
          ENDIF(SCIRUN_THIRDPARTY_DIR_ENV)
      ENDIF( NOT SCIRUN_THIRDPARTY_DIR )

  ENDIF(NOT WIN32)

ENDIF(NOT SCIRUN_THIRDPARTY_DIR)


IF(SCIRUN_THIRDPARTY_DIR)

    IF( NOT EXISTS ${SCIRUN_THIRDPARTY_DIR}/SCIRUN_THIRDPARTY_VERSION )
        MESSAGE( "\nError: The SCIRun Thirdparty directory (${SCIRUN_THIRDPARTY_DIR}) does not appear to be valid." )
        MESSAGE( "If you specified a directory, please verify that the directory exists and has the correct permissions." )
        MESSAGE( FATAL_ERROR "If you did not specify a directory, you may need to remove the CMakeCache.txt file and then run cmake again.\n" )
    ENDIF( NOT EXISTS ${SCIRUN_THIRDPARTY_DIR}/SCIRUN_THIRDPARTY_VERSION )


ELSE(SCIRUN_THIRDPARTY_DIR)

    SET(SCIRUN_THIRDPARTY_DIR "." CACHE PATH "Path to SCIRun's Thirdparty")
    MESSAGE(FATAL_ERROR "\n\nSCIRUN_THIRDPARTY_DIR must be specified to build. Please re-run cmake "
                        "and specify a valid path to the SCIRun Thirdparty (version ${SCIRUN_VERSION_STRING}) "
                        "using the SCIRUN_THIRDPARTY_DIR variable. This path must contain the SCIRUN_THIRDPARTY_VERSION file.\n\n")

ENDIF(SCIRUN_THIRDPARTY_DIR)

# Verify that 3P is of the correct version:

FILE(READ ${SCIRUN_THIRDPARTY_DIR}/SCIRUN_THIRDPARTY_VERSION SCIRUN_THIRDPARTY_VERSION_STRING)
STRING(REGEX MATCH "SCIRUN_THIRDPARTY_VERSION=${SCIRUN_VERSION_STRING}" THIRDPARTY_VERSION_CHECK ${SCIRUN_THIRDPARTY_VERSION_STRING})
IF( NOT THIRDPARTY_VERSION_CHECK )
    MESSAGE(FATAL_ERROR "SCIRUN_THIRDPARTY_VERSION file does not contain correct version of ${SCIRUN_VERSION_STRING}")
ENDIF( NOT THIRDPARTY_VERSION_CHECK )

SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_THIRDPARTY_DIR}/include)

# Force SCIRUN_THIRDPARTY_DIR to be cached
SET(SCIRUN_THIRDPARTY_DIR ${SCIRUN_THIRPDARTY_PATH} CACHE PATH "Path to SCIRun's Thirdparty.")

#-----------------------------------------------------------------------------
# SCIRun Packages to enable (must enable packages when building core for now)
# DV Fix need error checking to make sure SCIRun gets included
SET(LOAD_PACKAGE "SCIRun,BioPSE,Teem,MatlabInterface" CACHE STRING "Building SCIRun, BioPSE, Teem, and MatlabInterface packages by default.")

# Extension so collaborators can set this up easily
IF(WITH_MODELCREATION_PACKAGE)
	STRING(REGEX MATCH "ModelCreation" MODELCREATION_INCLUDED ${LOAD_PACKAGE})
	IF (NOT MODELCREATTION_INCLUDED)
		SET(LOAD_PACKAGE "${LOAD_PACKAGE},ModelCreation")
		MESSAGE(STATUS "LoadPackage=${LOAD_PACKAGE}")		
	ENDIF (NOT MODELCREATTION_INCLUDED)	
ENDIF(WITH_MODELCREATION_PACKAGE)

# ----------------------------------------------------------------------------
# SCIRun build configuration options
SET(DEF_DISABLE_DYNAMIC_COMPILATION "" CACHE INTERNAL "Whether or not to disable dynamic compilation (for binary releases)")

# Add correct flags for win32
IF(WIN32)
  IF(NOT BUILD_SHARED_LIBS)
    ADD_DEFINITIONS(-DBUILD_CORE_STATIC)
    ADD_DEFINITIONS(-DGLEW_STATIC)
    # builds against static teem
    ADD_DEFINITIONS(-DTEEM_STATIC)
  ENDIF(NOT BUILD_SHARED_LIBS)
  #  This defines M_PI for windows
  ADD_DEFINITIONS(-D_USE_MATH_DEFINES)
  # defines Windows 2000 features like mouse leaving window
  ADD_DEFINITIONS(-D_WIN32_WINNT=0x0500)
  # turns off VS 2005 warnings about strcpy et.al. being unsafe
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE)
  
  # win32 l3 warnings are a little ridiculous
  SET(CMAKE_CXX_WARNING_LEVEL 1)
  IF(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    STRING(REGEX REPLACE "/W[0-4]" "/W1"
            CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  ELSE(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W1")
  ENDIF(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
  
  IF(NOT CYGWIN)
    IF(CMAKE_GENERATOR MATCHES "Visual Studio 7")
      STRING(REGEX REPLACE "Common7/IDE/devenv.com" "Vc7" VC "${CMAKE_MAKE_PROGRAM}")
    ENDIF(CMAKE_GENERATOR MATCHES "Visual Studio 7")
    IF(CMAKE_GENERATOR MATCHES "Visual Studio 8")
      STRING(REGEX REPLACE "Common7/IDE/VCExpress.exe" "VC" VC "${CMAKE_MAKE_PROGRAM}")
    ENDIF(CMAKE_GENERATOR MATCHES "Visual Studio 8")
    SET (PATH_TO_VC "${VC}" CACHE PATH "Path to Visual Studio")
    MARK_AS_ADVANCED(PATH_TO_VC)
    
    # Look for PlatformSDK, for dynamic compilation
    IF(WIN32)
      # Find Platform SDK - try 3 default locations before asking the user
      IF(NOT PATH_TO_PSDK)
        SET(PATH_TO_PSDK ${PATH_TO_VC}/PlatformSDK)
        IF(NOT EXISTS ${PATH_TO_PSDK}/Include/windows.h)
          SET(PATH_TO_PSDK "C:/Program Files/Microsoft Platform SDK")
          IF(NOT EXISTS ${PATH_TO_PSDK}/Include/windows.h)
            SET(PATH_TO_PSDK "C:/Program Files/Microsoft Platform SDK for Windows Server 2003 R2")
          ENDIF(NOT EXISTS ${PATH_TO_PSDK}/Include/windows.h)
        ENDIF(NOT EXISTS ${PATH_TO_PSDK}/Include/windows.h)
      ENDIF(NOT PATH_TO_PSDK)

      IF (PATH_TO_PSDK)
        IF(NOT EXISTS ${PATH_TO_PSDK}/Include/windows.h)
          SET(PATH_TO_PSDK "." CACHE PATH "Path to Microsoft Platform SDK")
          MESSAGE(FATAL_ERROR "\nMicrosoft Platform SDK could not be found. Please re-run cmake "
                              "and specify a valid path to the Platform SDK "
                              "using the PATH_TO_PSDK variable. This path must contain Include/windows.h.\n")
        
        ENDIF(NOT EXISTS ${PATH_TO_PSDK}/Include/windows.h)
      ENDIF(PATH_TO_PSDK)
    ENDIF(WIN32)

  ENDIF(NOT CYGWIN)

  OPTION(ENABLE_DYNAMIC_COMPILATION "Toggle dynamic compilation.  Disable for binary installations" ON)
  MARK_AS_ADVANCED(ENABLE_DYNAMIC_COMPILATION)
  IF(NOT ENABLE_DYNAMIC_COMPILATION)
    SET(DEF_DISABLE_DYNAMIC_COMPILATION "#define DISABLE_DYNAMIC_COMPILATION 1" CACHE INTERNAL "blah")
  ENDIF(NOT ENABLE_DYNAMIC_COMPILATION)

ENDIF(WIN32)

#------------------------------------------------------------------------------
# enable/disable sci malloc
OPTION(ENABLE_SCI_MALLOC "Enable sci malloc system" OFF)
IF(ENABLE_SCI_MALLOC)
  SET(DEF_MALLOC "")
ELSE(ENABLE_SCI_MALLOC)
  SET(DEF_MALLOC "#define DISABLE_SCI_MALLOC 1")
ENDIF(ENABLE_SCI_MALLOC)
SET(DEF_MALLOC_LINENUMBERS "")
MARK_AS_ADVANCED(ENABLE_SCI_MALLOC)

#-----------------------------------------------------------------------------
# Teem in Thirdparty
SET (CMAKE_MODULE_PATH ${SCIRUN_CORE_SOURCE_DIR}/CMake)
FIND_PACKAGE( Teem )
IF (TEEM_LIBRARY)
  IF (TEEM_INCLUDE_DIR)
    SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} ${TEEM_INCLUDE_DIR})
  ENDIF(TEEM_INCLUDE_DIR)
ENDIF(TEEM_LIBRARY)

#-----------------------------------------------------------------------------
# Z lib in Thirdparty
FIND_LIBRARY(ZLIB_LIBRARY 
  NAMES z zlib zdll
  PATHS ${SCIRUN_THIRDPARTY_DIR}/lib/
  NO_DEFAULT_PATH)

FIND_PATH(ZLIB_INCLUDE zlib.h
  ${SCIRUN_THIRDPARTY_DIR}/include/
  NO_DEFAULT_PATH)

IF(ZLIB_LIBRARY)
#   MESSAGE(STATUS "Found ZLIB: ${ZLIB_LIBRARY}")
ELSE(ZLIB_LIBRARY)
   MESSAGE(FATAL_ERROR "z library required to be in SCIRun's Thirdparty. Please re-install your Thirdparty and verify that the z library installed propery.")
ENDIF(ZLIB_LIBRARY)

#-----------------------------------------------------------------------------
# png lib in Thirdparty
FIND_LIBRARY(PNG_LIBRARY png
             PATHS ${SCIRUN_THIRDPARTY_DIR}/lib/
             NO_DEFAULT_PATH)

FIND_PATH(PNG_INCLUDE png.h
          ${SCIRUN_THIRDPARTY_DIR}/include/
          NO_DEFAULT_PATH)

IF(PNG_LIBRARY)
#   MESSAGE(STATUS "Found PNG: ${PNG_LIBRARY}")
   SET(HAVE_PNG "YES")
   SET(DEF_PNG "#define HAVE_PNG 1")
ELSE(PNG_LIBRARY)
   MESSAGE(FATAL_ERROR "PNG library required to be in SCIRun's Thirdparty. Please re-install your Thirdparty and verify that the PNG library installed propery.")
ENDIF(PNG_LIBRARY)


#-----------------------------------------------------------------------------
# XML2 in Thirdparty
FIND_LIBRARY(XML2_LIBRARY xml2
            PATHS ${SCIRUN_THIRDPARTY_DIR}/lib/
            NO_DEFAULT_PATH)

FIND_PATH(XML2_INCLUDE libxml/xmlreader.h
           ${SCIRUN_THIRDPARTY_DIR}/include/libxml2/
           ${SCIRUN_THIRDPARTY_DIR}/include/
           NO_DEFAULT_PATH)
SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} ${XML2_INCLUDE})

#-----------------------------------------------------------------------------
# Tcl in Thirdparty
FIND_LIBRARY(TCL_LIBRARY tcl
             PATHS ${SCIRUN_THIRDPARTY_DIR}/lib/
             NO_DEFAULT_PATH)

FIND_PATH(TCL_INCLUDE1 tcl.h
           ${SCIRUN_THIRDPARTY_DIR}/include
           NO_DEFAULT_PATH)

FIND_PATH(TCL_INCLUDE2 tclMath.h
           ${SCIRUN_THIRDPARTY_DIR}/src/tcl/generic
           ${SCIRUN_THIRDPARTY_DIR}/include
           NO_DEFAULT_PATH)
MARK_AS_ADVANCED(TCL_INCLUDE1 TCL_INCLUDE2)

# Tk in Thirdparty
FIND_LIBRARY(TK_LIBRARY tk
             PATHS ${SCIRUN_THIRDPARTY_DIR}/lib/
             NO_DEFAULT_PATH)

# Tcl/Tk versions
IF(WIN32)
  # BJW - find a better way to do this
  SET(TK_VERSION 8.4)
  SET(TCL_VERSION 8.4)
ELSE(WIN32)
  SET(TK_VERSION 8.3)
  SET(TCL_VERSION 8.3)
ENDIF(WIN32)

#-----------------------------------------------------------------------------
# ITK in Thirdparty
FIND_LIBRARY(ITK_LIBRARY itk
             PATHS ${SCIRUN_THIRDPARTY_DIR}/lib/
             NO_DEFAULT_PATH)

FIND_PATH(ITK_INCLUDE itk.h
           ${SCIRUN_THIRDPARTY_DIR}/include
           NO_DEFAULT_PATH)

#-----------------------------------------------------------------------------
# ITCL library in Thirdparty
FIND_LIBRARY(ITCL_LIBRARY itcl
             PATHS ${SCIRUN_THIRDPARTY_DIR}/lib/
             NO_DEFAULT_PATH)

FIND_PATH(ITCL_INCLUDE itcl.h
           ${SCIRUN_THIRDPARTY_DIR}/include
           NO_DEFAULT_PATH)

#-----------------------------------------------------------------------------
# blt lib in Thirdparty
FIND_LIBRARY(BLT_LIBRARY BLT
             PATHS ${SCIRUN_THIRDPARTY_DIR}/lib/
             NO_DEFAULT_PATH)

FIND_PATH(BLT_INCLUDE blt.h
           ${SCIRUN_THIRDPARTY_DIR}/include
           NO_DEFAULT_PATH)

IF(WIN32)
  SET(BLT_LIBRARY ${BLT_LIBRARY} gdi32 winspool comdlg32 user32 advapi32)
ENDIF(WIN32)

#-----------------------------------------------------------------------------
# Freetype in Thirdparty
FIND_LIBRARY(FREETYPE_LIBRARY 
             NAMES freetype libfreetype.so libfreetype.a
             PATHS ${SCIRUN_THIRDPARTY_DIR}/lib/
             NO_DEFAULT_PATH)

FIND_PATH(FREETYPE_INCLUDE freetype/freetype.h
           ${SCIRUN_THIRDPARTY_DIR}/include/freetype2/
           ${SCIRUN_THIRDPARTY_DIR}/include/
           NO_DEFAULT_PATH)

IF(FREETYPE_LIBRARY)
  SET(DEF_FREETYPE "#define HAVE_FREETYPE 1")
  SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} ${FREETYPE_INCLUDE})
ELSE(FREETYPE_LIBRARY)
  SET(DEF_FREETYPE "")
ENDIF(FREETYPE_LIBRARY)

#-----------------------------------------------------------------------------
# MPEG sometimes in Thirdparty (not in Windows?)
IF(WIN32)
  OPTION(WITH_MPEG "Build SCIRun with MPEG support" OFF)
ELSE(WIN32)
  OPTION(WITH_MPEG "Build SCIRun with MPEG support" ON)
ENDIF(WIN32)

IF(WITH_MPEG)
  FIND_LIBRARY(MPEG_LIBRARY mpege
              PATHS ${SCIRUN_THIRDPARTY_DIR}/lib/
              /usr/local/lib
              /usr/lib
              /lib
              NO_DEFAULT_PATH)

  FIND_PATH(MPEG_INCLUDE mpege.h
             ${SCIRUN_THIRDPARTY_DIR}/include
             /usr/local/include
             /usr/include
             /include
             NO_DEFAULT_PATH)

  IF(MPEG_LIBRARY)
    SET(HAVE_MPEG "1")
    SET(DEF_MPEG "#define HAVE_MPEG 1")
#    MESSAGE(STATUS "Found MPEG: ${MPEG_LIBRARY}")
    SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${MPEG_INCLUDE})
  ELSE(MPEG_LIBRARY)
    SET(HAVE_MPEG "0")
#    MESSAGE(STATUS "MPEG not found")
  ENDIF(MPEG_LIBRARY)
ENDIF(WITH_MPEG)

#-----------------------------------------------------------------------------
# Thread library
FIND_PACKAGE( Threads )
SET(THREAD_LIBRARY ${CMAKE_THREAD_LIBS} CACHE INTERNAL "Thread library.")
# DV Define DEF_THREAD?

IF(CMAKE_USE_PTHREADS)
  SET(DEF_THREAD "#define SCI_PTHREAD 1")
  # Check for sempahore libs w/in pthread library
  # DV can I scale down the lib like configure.ac for loop
  TRY_COMPILE(HAS_SEMAPHORE_LIBS
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeTmp
    ${SCIRUN_CORE_SOURCE_DIR}/CMake/testSemaphore.cc
      COMPILE_DEFINITIONS
      -l${THREAD_LIBRARY}
      -I/usr/local/include -I/usr/include -I/include
    )
    IF(HAS_SEMAPHORE_LIBS)
      SET(SEMAPHORE_LIBRARY ${THREAD_LIBRARY} CACHE INTERNAL "Semaphore library in pthreads.")
    ENDIF(HAS_SEMAPHORE_LIBS)
  ELSE(CMAKE_USE_PTHREADS)
    SET(DEF_THREAD "") 
ENDIF(CMAKE_USE_PTHREADS)

#-----------------------------------------------------------------------------
# Socket library
# DV Not sure if socket library needs to be defined for cygwin
IF(WIN32 AND NOT CYGWIN)
  SET(SOCKET_LIBRARY ws2_32 CACHE INTERNAL "Socket library ws2_32")
  SET(THREAD_LIBRARY winmm CACHE INTERNAL "Time functionality library winmm")
ENDIF(WIN32 AND NOT CYGWIN)

#-----------------------------------------------------------------------------
# math library
IF(UNIX)
  FIND_LIBRARY(M_LIBRARY m
             PATHS /usr/local/lib
             /usr/lib 
             /lib)
ENDIF(UNIX)

#-----------------------------------------------------------------------------
# dl library
IF(UNIX)
  FIND_LIBRARY(DL_LIBRARY dl)
ENDIF(UNIX)

#-----------------------------------------------------------------------------
# lapack library
OPTION(WITH_LAPACK "Build with lapack library" OFF)
MARK_AS_ADVANCED(WITH_LAPACK)		   
IF(WITH_LAPACK OR LAPACK_DIR)		   
  IF(CMAKE_SYSTEM MATCHES "IRIX.*" OR CMAKE_SYSTEM MATCHES "OSF.*" OR CMAKE_SYSTEM MATCHES "AIX.*" OR APPLE)
    # these guys have one lib for both blas and lapack - find it in the blas section
    SET(WITH_BLAS="ON")
  ELSE(CMAKE_SYSTEM MATCHES "IRIX.*" OR CMAKE_SYSTEM MATCHES "OSF.*" OR CMAKE_SYSTEM MATCHES "AIX.*" OR APPLE)
    # everybody else - probably linux
    MESSAGE(STATUS "Looking for LAPACK")
    SET(LAPACK_DIR ${LAPACK_DIR} CACHE PATH "Path to Lapack" FORCE)
    MARK_AS_ADVANCED(CLEAR LAPACK_DIR)

    FIND_LIBRARY(LAPACK_LIBRARY NAMES lapack clapack flapack 
      PATHS ${LAPACK_DIR}/lib NO_DEFAULT_PATH)
    FIND_LIBRARY(LAPACK_LIBRARY NAMES lapack clapack flapack 
      PATHS /usr/lib /usr/local/lib /lib)
    IF(LAPACK_LIBRARY)
      MESSAGE(STATUS "Looking for LAPACK - found")
      SET(HAVE_LAPACK "YES")
    ELSE(LAPACK_LIBRARY)
      MESSAGE(STATUS "Looking for LAPACK - not found")
      SET(HAVE_LAPACK "NO")
      MESSAGE(FATAL_ERROR "WITH_LAPACK OR LAPACK_DIR specified and lapack was not found")
    ENDIF(LAPACK_LIBRARY)
  ENDIF(CMAKE_SYSTEM MATCHES "IRIX.*" OR CMAKE_SYSTEM MATCHES "OSF.*" OR CMAKE_SYSTEM MATCHES "AIX.*" OR APPLE)
ELSE(WITH_LAPACK OR LAPACK_DIR)
   SET(HAVE_LAPACK "NO")
ENDIF(WITH_LAPACK OR LAPACK_DIR)

#-----------------------------------------------------------------------------
# Search for SGI's or Win32's traceback library
# traceback_lib exc_lib_dir_flag exc_lib_flag
# DV Make required and look for it, also fix for SGI
INCLUDE (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
IF(WIN32)
  CHECK_INCLUDE_FILE("windows.h"       HAVE_WINDOWS_H)
  #CHECK_INCLUDE_FILE("dbghlp"       HAVE_DBGHLP_H)
  IF(HAVE_WINDOWS_H)
  #    IF(HAVE_DBGHLP_H)
        SET(TRACEBACK_LIB imagehlp psapi CACHE INTERNAL "Trace back library.")
        SET(HAVE_TRACEBACK 1)
        SET(DEF_TRACEBACK "#define HAVE_TRACEBACK 1")
  #    ELSE(HAVE_DBGHELP_H)
  #      MESSAGE(SEND_ERROR "Trace back library requires dbghlp.h for Windows.")
  #    ENDIF(HAVE_DBGHLP_H)
  ELSE(HAVE_WINDOWS_H)
     MESSAGE(SEND_ERROR "Trace back library requires windows.h for Windows.")
  ENDIF(HAVE_WINDOWS_H)
ELSE(WIN32)
  CHECK_INCLUDE_FILE("libexc.h"       HAVE_TRACEBACK_H)
  IF(HAVE_TRACEBACK_H)
    SET(TRACEBACK_LIB exc CACHE INTERNAL "Trace back library.")
    SET(HAVE_TRACEBACK 1)
    SET(DEF_TRACEBACK "#define HAVE_TRACEBACK 1")
  ENDIF(HAVE_TRACEBACK_H)
ENDIF(WIN32)

#-----------------------------------------------------------------------------
# OpenGL library - win32 finds the OPENGL_gl_LIBRARY but for some reason this 
#   if check does not complete
# Use our own find OpenGL module
SET (CMAKE_MODULE_PATH ${SCIRUN_CORE_SOURCE_DIR}/CMake)
FIND_PACKAGE( OpenGL )
IF(OPENGL_LIBRARIES AND OPENGL_INCLUDE_DIR)
  SET(GL_LIBRARY ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} CACHE INTERNAL "OpenGL Libs")
  SET(GL_INCLUDE ${OPENGL_INCLUDE_DIR})
  SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${GL_INCLUDE})
ENDIF(OPENGL_LIBRARIES AND OPENGL_INCLUDE_DIR)
IF (WIN32)
  SET(GL_LIBRARY ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} gdi32 CACHE INTERNAL "OpenGL Libs")
  SET(GL_INCLUDE ${OPENGL_INCLUDE_DIR})
  SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${GL_INCLUDE})
ENDIF(WIN32)

# if we are a debug build then define our debug opengl code.
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  ADD_DEFINITIONS(-DOGL_DBG)
endif(CMAKE_BUILD_TYPE STREQUAL Debug)

#----------------------------------------------------------------------------
# X11
OPTION(WITH_X11 "Build SCIRun with X11 support" ON)

SET(X11_FOUND 0)

IF(WITH_X11)
  FIND_PACKAGE( X11 )
  IF(X11_FOUND)
    SET(X_LIBRARY ${X11_LIBRARIES})
    SET(DEF_X11 "#define HAVE_X11 1")
  ENDIF(X11_FOUND)
ENDIF(WITH_X11)

MARK_AS_ADVANCED(WITH_X11)

# Fixing a problem on OSX for now: the standard CMake script finds the wrong OpenGL
# libraries. Hence they are hard coded here, however these require X11 to be installed
# and this is not checked here. We need to come up with a better fix here.
 
IF (APPLE AND X11_FOUND)
  SET(GL_LIBRARY /usr/X11R6/lib/libGL.dylib /usr/X11R6/lib/libGLU.dylib "-framework AGL" CACHE INTERNAL "OpenGL Libs")
  SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${GL_INCLUDE})
ENDIF(APPLE AND X11_FOUND)

#----------------------------------------------------------------------------
# sci_defs variables

# sci_defs/compile_defs.h
# DV Not needed eventually?

# Set Make for MAKE_COMMAND in compile_defs.h
# DV Fix Make sure this works for win
SET(MAKE ${CMAKE_BUILD_TOOL})

# sci_defs/environment_defs.h
SET(scirun_src_dir "${SCIRUN_CORE_SOURCE_DIR}" CACHE INTERNAL "Path to SCIRun's src directory")
# DV Technically, I don't have access to this variable once we fix building
# these projets separately!!!
SET(scirun_obj_dir ${SCIRUN_CORE_BINARY_DIR} CACHE INTERNAL "Path to SCIRun's binary directory")

SET(ITCL_WIDGETS ${SCIRUN_THIRDPARTY_DIR}/iwidgets/scripts CACHE INTERNAL "Path to iwidgets scripts")
SET(ITCL_LIB_DIR ${SCIRUN_THIRDPARTY_DIR}/lib)

# sci_defs/error_defs.h
SET(THROW_LEVEL 0 CACHE STRING "Set throw level N(0-3)")
SET(ASSERTION_LEVEL 2 CACHE STRING "Set assertion level N(0-3)")
IF(CMAKE_BUILD_TYPE STREQUAL Debug)
  SET(ASSERTION_LEVEL 3)
ENDIF(CMAKE_BUILD_TYPE STREQUAL Debug)


# sci_defs/lapack_defs.h
# DV FIX hard coding of lapack library
IF(HAVE_LAPACK)
  SET(DEF_LAPACK "#define HAVE_LAPACK 1" CACHE INTERNAL "Have Lapack")
ENDIF(HAVE_LAPACK)

# sci_defs/hashmap_defs.h
SET(DEF_HASHMAP "None" CACHE INTERNAL "Determining hash map")

# DV Fix to figure out value of DEF_HASHMAP
TRY_COMPILE(DEF_HASHMAP
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeTmp
  ${SCIRUN_CORE_SOURCE_DIR}/CMake/SCIRunTestHashMapSTD.cc
  OUTPUT_VARIABLE OUTPUT)
IF(DEF_HASHMAP)
  SET(DEF_HASHMAP "#define HAVE_STD_HASHMAP 1" CACHE INTERNAL "Using STD hash map")
ELSE(DEF_HASHMAP)
  TRY_COMPILE(DEF_HASHMAP 
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeTmp
  ${SCIRUN_CORE_SOURCE_DIR}/CMake/SCIRunTestHashMapEXT.cc
  OUTPUT_VARIABLE OUTPUT)

  # include <ext/hash_map> instead
  IF(DEF_HASHMAP)
    SET(DEF_HASHMAP "#define HAVE_EXT_HASHMAP 1" CACHE INTERNAL "Using EXT hash map")
  ELSE(DEF_HASHMAP)
   TRY_COMPILE(DEF_HASHMAP 
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeTmp
    ${SCIRUN_CORE_SOURCE_DIR}/CMake/SCIRunTestHashMapSTDEXT.cc
    OUTPUT_VARIABLE OUTPUT)

    IF(DEF_HASHMAP)
      SET(DEF_HASHMAP "#define HAVE_STDEXT_HASHMAP 1" CACHE INTERNAL "Using STD EXT hash map")
    ELSE(DEF_HASHMAP)
      TRY_COMPILE(DEF_HASHMAP 
      ${CMAKE_CURRENT_BINARY_DIR}/CMakeTmp
      ${SCIRUN_CORE_SOURCE_DIR}/CMake/SCIRunTestHashMapGNU.cc
      OUTPUT_VARIABLE OUTPUT)
      # include <ext/hash_map> instead

      IF(DEF_HASHMAP)
        SET(DEF_HASHMAP "#define HAVE_GNU_HASHMAP 1" CACHE INTERNAL "Using GNU hash map")
      ELSE(DEF_HASHMAP)
        SET(DEF_HASHMAP "" CACHE INTERNAL "No hash map found")
        MESSAGE(STATUS "Cannot find hash_map, will use map instead") 
      ENDIF(DEF_HASHMAP)
    ENDIF(DEF_HASHMAP)
  ENDIF(DEF_HASHMAP)
ENDIF(DEF_HASHMAP)

# DV Figure out how to make these not even configurable
# sci_defs/template_defs.h
SET(TEMPLATE_TAG " " CACHE STRING "NOT SURE")
# DV This should be an empty string on some platforms
SET(TEMPLATE_BOX "<>" CACHE STRING "NOT SURE")

MARK_AS_ADVANCED(THROW_LEVEL ASSERTION_LEVEL DEF_HASH TEMPLATE_TAG TEMPLATE_BOX)
# DV Add with_exc stuff from configure.ac
INCLUDE (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
CHECK_INCLUDE_FILE(limits.h HAVE_LIMITS)
IF(HAVE_LIMITS)
  SET(HAVE_LIMITS_H 1)
ELSE(HAVE_LIMITS)
  SET(HAVE_LIMITS_H 0)
ENDIF(HAVE_LIMITS)

CHECK_INCLUDE_FILE(sys/select.h HAVE_SYS_SELECT)
IF(HAVE_SYS_SELECT)
  SET(HAVE_SYS_SELECT_H 1)
ELSE(HAVE_SYS_SELECT)
  SET(HAVE_SYS_SELECT_H 0)
ENDIF(HAVE_SYS_SELECT)

CHECK_INCLUDE_FILE(sys/time.h HAVE_SYS_TIME)
IF(HAVE_SYS_TIME)
  SET(HAVE_SYS_TIME_H 1)
ELSE(HAVE_SYS_TIME)
  SET(HAVE_SYS_TIME_H 0)
ENDIF(HAVE_SYS_TIME)

# BW Fix to figure out if we have ext/algorithm
TRY_COMPILE(HAVE_EXT_ALGORITHM
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeTmp
  ${SCIRUN_CORE_SOURCE_DIR}/CMake/SCIRunTestExtAlgorithm.cc
  OUTPUT_VARIABLE OUTPUT)
IF(HAVE_EXT_ALGORITHM)
  SET(HAVE_EXT_ALGORITHM_H 1)
ELSE(HAVE_EXT_ALGORITHM)
  SET(HAVE_EXT_ALGORITHM_H 0)
ENDIF(HAVE_EXT_ALGORITHM)

CHECK_INCLUDE_FILE(stdint.h HAVE_STDINT)
IF(HAVE_STDINT)
  SET(HAVE_STDINT_H 1)
ELSE(HAVE_STDINT)
  SET(HAVE_STDINT_H 0)
ENDIF(HAVE_STDINT)

CHECK_INCLUDE_FILE(inttypes.h HAVE_INTTYPES)
IF(HAVE_INTTYPES)
  SET(HAVE_INTTYPES_H 1)
ELSE(HAVE_INTTYPES)
  SET(HAVE_INTTYPES_H 0)
ENDIF(HAVE_INTTYPES)

SET(HAVE_SYS_UUID_H 0)
SET(HAVE_UUID_UUID_H 0)

#SET(DEF_FTGL)

#SET(DEF_OGL_EXT_HDRS)

# sci_defs/mpi_defs.h
# mpi libray

OPTION(WITH_MPI "Build with mpi library" OFF)
MARK_AS_ADVANCED(WITH_MPI)		   
IF(WITH_MPI OR MPI_DIR)		   
  MESSAGE(STATUS "Looking for MPI")
  SET(MPI_DIR ${MPI_DIR} CACHE PATH "Path to MPI" FORCE)
  MARK_AS_ADVANCED(CLEAR MPI_DIR)
  # general
  FIND_LIBRARY(MPI_LIBRARY 
    NAMES mpi
    PATHS ${MPI_DIR}/lib NO_DEFAULT_PATH)
  MARK_AS_ADVANCED(MPI_LIBRARY)

  IF(MPI_LIBRARY)
    SET(MPI_LIBS ${MPI_LIBRARY} CACHE INTERNAL "All MPI libs needed")
    # test for lam
    FIND_LIBRARY(LAM_LIBRARY 
      NAMES lam 
      PATHS ${MPI_DIR}/lib NO_DEFAULT_PATH)
    MARK_AS_ADVANCED(LAM_LIBRARY)
    IF(LAM_LIBRARY)
      SET(MPI_LIBS ${MPI_LIBS} ${LAM_LIBRARY} CACHE INTERNAL "All MPI libs needed")
      FIND_LIBRARY(UTIL_LIBRARY NAMES util)
      MARK_AS_ADVANCED(UTIL_LIBRARY)
      IF(UTIL_LIBRARY)
        SET(MPI_LIBS ${MPI_LIBS} ${UTIL_LIBRARY} CACHE INTERNAL "All MPI libs needed")
      ENDIF(UTIL_LIBRARY)
    ENDIF(LAM_LIBRARY)
  ENDIF(MPI_LIBRARY)

  IF(NOT MPI_LIBRARY)
    # mpich
    FIND_LIBRARY(MPICH_LIBRARY 
      NAMES mpich
      PATHS ${MPI_DIR}/lib NO_DEFAULT_PATH)

    MARK_AS_ADVANCED(MPICH_LIBRARY)
    IF(MPICH_LIBRARY)
      # mpich (with pmpich
      SET(MPI_LIBS ${MPICH_LIBRARY} CACHE INTERNAL "All MPI libs needed")
      FIND_LIBRARY(PMPICH_LIBRARY 
        NAMES pmpich
        PATHS ${MPI_DIR}/lib NO_DEFAULT_PATH)
      MARK_AS_ADVANCED(PMPICH_LIBRARY)
      IF(PMPICH_LIBRARY)
        SET(MPI_LIBS ${MPI_LIBS} ${PMPICH_LIBRARY} CACHE INTERNAL "All MPI libs needed")
      ENDIF(PMPICH_LIBRARY)
    ENDIF(MPICH_LIBRARY)
  ENDIF(NOT MPI_LIBRARY)
  
  IF(MPI_LIBRARY OR MPICH_LIBRARY)
    MESSAGE(STATUS "Looking for MPI - found")
    SET(HAVE_MPI "YES")
    SET(DEF_MPI "#define HAVE_MPI 1")
    IF(MPICH_LIBRARY)
      SET(DEF_MPICH "#define HAVE_MPICH 1")
    ENDIF(MPICH_LIBRARY)
  ELSE(MPI_LIBRARY OR MPICH_LIBRARY)
  MESSAGE(STATUS "Looking for MPI - not found")
    SET(HAVE_MPI "NO")
  ENDIF(MPI_LIBRARY OR MPICH_LIBRARY)

  FIND_PATH(MPI_INCLUDE mpi.h
    ${MPI_DIR}/include)

  MARK_AS_ADVANCED(MPI_INCLUDE)
  IF(MPI_INCLUDE)
    SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${MPI_INCLUDE})
  ENDIF(MPI_INCLUDE)
  
ELSE(WITH_MPI OR MPI_DIR)
   SET(HAVE_MPI "NO")
ENDIF(WITH_MPI OR MPI_DIR)


# sci_defs/petsc_defs.h
# petsc library
OPTION(WITH_PETSC "Build with petsc library" OFF)
MARK_AS_ADVANCED(WITH_PETSC)		   
IF(WITH_PETSC OR PETSC_DIR)		   
  MESSAGE(STATUS "Looking for Petsc")
  SET(PETSC_DIR ${PETSC_DIR} CACHE PATH "Path to Petsc" FORCE)
  MARK_AS_ADVANCED(CLEAR PETSC_DIR)
  FIND_LIBRARY(PETSC_LIBRARY
    NAMES petsc PATHS /usr/local/lib /usr/lib /lib)
  FIND_LIBRARY(PETSCVEC_LIBRARY
    NAMES petscvec PATHS /usr/local/lib /usr/lib /lib)
  FIND_LIBRARY(PETSCMAT_LIBRARY
    NAMES petscmat PATHS /usr/local/lib /usr/lib /lib)
  FIND_LIBRARY(PETSCDM_LIBRARY
    NAMES petscdm PATHS /usr/local/lib /usr/lib /lib)
  FIND_LIBRARY(PETSCSNES_LIBRARY
    NAMES petscsnes PATHS /usr/local/lib /usr/lib /lib)
  FIND_LIBRARY(PETSCTS_LIBRARY
    NAMES petscts PATHS /usr/local/lib /usr/lib /lib)
  FIND_LIBRARY(PETSCKSP_LIBRARY
    NAMES petscksp PATHS /usr/local/lib /usr/lib /lib)


  FIND_PATH(PETSC_INCLUDE petsc.h
    /usr/local/include/
    /usr/include
    /include)
  FIND_PATH(PETSCCONF_INCLUDE petscconf.h
    /usr/local/include/
    /usr/include
    /include)

  IF(NOT PETSC_LIBRARY OR NOT PETSCVEC_LIBRARY OR NOT PETSCMAT_LIBRARY OR NOT PETSCDM_LIBRARY 
     OR NOT PETSCSNES_LIBRARY OR NOT PETSCTS_LIBRARY OR NOT PETSCKSP_LIBRARY OR NOT PETSC_INCLUDE 
     OR NOT PETSCCONF_INCLUDE)
    # Look in PETSC_DIR
    SET (PETSC_ARCH linux CACHE STRING "PETSC_ARCH")
    FIND_LIBRARY(PETSC_LIBRARY
      NAMES petsc PATHS ${PETSC_DIR}/lib ${PETSC_DIR}/lib/${PETSC_ARCH} NO_DEFAULT_PATH)
    FIND_LIBRARY(PETSCVEC_LIBRARY
      NAMES petscvec PATHS ${PETSC_DIR}/lib ${PETSC_DIR}/lib/${PETSC_ARCH} NO_DEFAULT_PATH)
    FIND_LIBRARY(PETSCMAT_LIBRARY
      NAMES petscmat PATHS ${PETSC_DIR}/lib ${PETSC_DIR}/lib/${PETSC_ARCH} NO_DEFAULT_PATH)
    FIND_LIBRARY(PETSCDM_LIBRARY
      NAMES petscdm PATHS ${PETSC_DIR}/lib ${PETSC_DIR}/lib/${PETSC_ARCH} NO_DEFAULT_PATH)
    FIND_LIBRARY(PETSCSNES_LIBRARY
      NAMES petscsnes PATHS ${PETSC_DIR}/lib ${PETSC_DIR}/lib/${PETSC_ARCH} NO_DEFAULT_PATH)
    FIND_LIBRARY(PETSCTS_LIBRARY
      NAMES petscts PATHS ${PETSC_DIR}/lib ${PETSC_DIR}/lib/${PETSC_ARCH} NO_DEFAULT_PATH)
    FIND_LIBRARY(PETSCKSP_LIBRARY
      NAMES petscksp PATHS ${PETSC_DIR}/lib ${PETSC_DIR}/lib/${PETSC_ARCH} NO_DEFAULT_PATH)

    MESSAGE("PETSC DIR: " ${PETSC_DIR} "/include")
    FIND_PATH(PETSC_INCLUDE petsc.h PATHS ${PETSC_DIR}/include NO_DEFAULT_PATH)
    FIND_PATH(PETSCCONF_INCLUDE petscconf.h PATHS ${PETSC_DIR}/bmake/${PETSC_ARCH} NO_DEFAULT_PATH)
  ENDIF(NOT PETSC_LIBRARY OR NOT PETSCVEC_LIBRARY OR NOT PETSCMAT_LIBRARY OR NOT PETSCDM_LIBRARY 
     OR NOT PETSCSNES_LIBRARY OR NOT PETSCTS_LIBRARY OR NOT PETSCKSP_LIBRARY OR NOT PETSC_INCLUDE
     OR NOT PETSCCONF_INCLUDE)

  IF(PETSC_LIBRARY AND PETSCVEC_LIBRARY AND PETSCMAT_LIBRARY AND PETSCDM_LIBRARY 
     AND PETSCSNES_LIBRARY AND PETSCTS_LIBRARY AND PETSCKSP_LIBRARY AND PETSC_INCLUDE)
    MESSAGE(STATUS "Looking for Petsc - found")
    MARK_AS_ADVANCED(PETSC_LIBRARY)		   
    MARK_AS_ADVANCED(PETSCVEC_LIBRARY)		   
    MARK_AS_ADVANCED(PETSCMAT_LIBRARY)		   
    MARK_AS_ADVANCED(PETSCDM_LIBRARY)		   
    MARK_AS_ADVANCED(PETSCSNES_LIBRARY)		   
    MARK_AS_ADVANCED(PETSCTS_LIBRARY)		   
    MARK_AS_ADVANCED(PETSCKSP_LIBRARY)		   
    MARK_AS_ADVANCED(PETSC_INCLUDE)		   
    MARK_AS_ADVANCED(PETSCCONF_INCLUDE)		   

    SET(HAVE_PETSC "YES")
    SET(DEF_PETSC "#define HAVE_PETSC 1")
    SET(PETSC_LIBRARIES ${PETSC_LIBRARY} ${PETSCVEC_LIBRARY} ${PETSCMAT_LIBRARY} ${PETSCDM_LIBRARY} 
     ${PETSCSNES_LIBRARY} ${PETSCTS_LIBRARY} ${PETSCKSP_LIBRARY} CACHE INTERNAL "All petsc libs")
    SET(PETSC_INCLUDES ${PETSC_INCLUDE} ${PETSCCONF_INCLUDE} CACHE INTERNAL "petsc include")
  ELSE(PETSC_LIBRARY AND PETSCVEC_LIBRARY AND PETSCMAT_LIBRARY AND PETSCDM_LIBRARY 
     AND PETSCSNES_LIBRARY AND PETSCTS_LIBRARY AND PETSCKSP_LIBRARY AND PETSC_INCLUDE)
    MESSAGE(STATUS "Looking for Petsc - not found")
    SET(HAVE_PETSC "NO")
    MESSAGE(FATAL_ERROR "WITH_PETSC or PETSC_DIR specified and petsc was not found")
  ENDIF(PETSC_LIBRARY AND PETSCVEC_LIBRARY AND PETSCMAT_LIBRARY AND PETSCDM_LIBRARY 
     AND PETSCSNES_LIBRARY AND PETSCTS_LIBRARY AND PETSCKSP_LIBRARY AND PETSC_INCLUDE)
ELSE(WITH_PETSC OR PETSC_DIR)
   SET(HAVE_PETSC "NO")
ENDIF(WITH_PETSC OR PETSC_DIR)

# sci_defs/hypre_defs.h
# hypre library
OPTION(WITH_HYPRE "Build with hypre library" OFF)
MARK_AS_ADVANCED(WITH_HYPRE)		   
IF(WITH_HYPRE OR HYPRE_DIR)
  MESSAGE(STATUS "Looking for Hypre")
  SET(HYPRE_DIR ${HYPRE_DIR} CACHE PATH "Path to Hypre" FORCE)
  MARK_AS_ADVANCED(CLEAR HYPRE_DIR)

  FIND_LIBRARY(HYPRE_LIBRARY NAMES HYPRE PATHS ${HYPRE_DIR}/lib)
  FIND_PATH(HYPRE_INCLUDE HYPRE.h PATHS ${HYPRE_DIR}/include)

  IF(HYPRE_LIBRARY AND HYPRE_INCLUDE)
    MARK_AS_ADVANCED(HYPRE_LIBRARY)
    MARK_AS_ADVANCED(HYPRE_INCLUDE)
    MESSAGE(STATUS "Looking for hypre - found")
    SET(HAVE_HYPRE "YES")
    SET(DEF_HYPRE "#define HAVE_HYPRE 1")
  ELSE(HYPRE_LIBRARY AND HYPRE_INCLUDE)
    MESSAGE(STATUS "Looking for hypre - not found")
    SET(HAVE_HYPRE "NO")
  ENDIF(HYPRE_LIBRARY AND HYPRE_INCLUDE)
ELSE(WITH_HYPRE OR HYPRE_DIR)
   SET(HAVE_HYPRE "NO")
ENDIF(WITH_HYPRE OR HYPRE_DIR)

# DV FIX check for unipetsc if petsc not found

#----------------------------------------------------------------------------
# Search for Mesquite

#----------------------------------------------------------------------------
# Search for Mesquite

IF(MESQUITE_DIR)
  SET(MESQUITE_DIR ${MESQUITE_DIR} CACHE PATH "Path to MESQUITE" FORCE)
  MARK_AS_ADVANCED(CLEAR MESQUITE_DIR)
ENDIF(MESQUITE_DIR)

OPTION(WITH_MESQUITE "compile with mesquite" OFF)
MARK_AS_ADVANCED(WITH_MESQUITE)

IF(WITH_MESQUITE)
  MARK_AS_ADVANCED(CLEAR WITH_MESQUITE)
ENDIF(WITH_MESQUITE)

IF(WITH_MESQUITE OR MESQUITE_DIR)

  MESSAGE(STATUS "Looking for MESQUITE")

  SET(MESQUITE_LIBRARY MESQUITE_LIBRARY-NOTFOUND CACHE FILEPATH "Mesquite library" FORCE)
  SET(MESQUITE_INCLUDE1 MESQUITE_INCLUDE1-NOTFOUND CACHE FILEPATH "Mesquite include1" FORCE)
  SET(MESQUITE_INCLUDE2 MESQUITE_INCLUDE2-NOTFOUND CACHE FILEPATH "Mesquite include2" FORCE)

  IF(MESQUITE_DIR)
    FIND_LIBRARY(MESQUITE_LIBRARY 
      NAMES mesquite
      PATHS ${MESQUITE_DIR}
      ${MESQUITE_DIR}/lib)

    FIND_PATH(MESQUITE_INCLUDE1 AddQualityMetric.hpp
      ${MESQUITE_DIR}
			${MESQUITE_DIR}/includeLinks
      ${MESQUITE_DIR}/../includeLinks)

    FIND_PATH(MESQUITE_INCLUDE2 Mesquite.hpp
      ${MESQUITE_DIR}
			${MESQUITE_DIR}/include
      ${MESQUITE_DIR}/../include)
  ELSE (MESQUITE_DIR)
    FIND_LIBRARY(MESQUITE_LIBRARY 
      NAMES mesquite
      PATHS ${SCIRUN_THIRDPARTY_DIR}/lib/
      /usr/local/lib
      /usr/lib
      /lib)

    FIND_PATH(MESQUITE_INCLUDE1 AddQualityMetric.hpp
      ${SCIRUN_THIRDPARTY_DIR}/includeLinks
      /usr/local/include/
      /usr/include
      /include)

    FIND_PATH(MESQUITE_INCLUDE2 Mesquite.hpp
      ${SCIRUN_THIRDPARTY_DIR}/include
      /usr/local/include/
      /usr/include
      /include)  
  ENDIF(MESQUITE_DIR)

  IF(MESQUITE_LIBRARY AND MESQUITE_INCLUDE1)
    SET(HAVE_MESQUITE "YES" CACHE INTERNAL "Do not edit" FORCE)
    SET(DEF_MESQUITE "#define HAVE_MESQUITE 1" CACHE INTERNAL "Do not edit" FORCE)
    SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} ${MESQUITE_INCLUDE1} ${MESQUITE_INCLUDE2})
  ELSE(MESQUITE_LIBRARY AND MESQUITE_INCLUDE1)
    MESSAGE(FATAL_ERROR "Cannot find mesquite library.") 
  ENDIF(MESQUITE_LIBRARY AND MESQUITE_INCLUDE1)

  MARK_AS_ADVANCED(MESQUITE_LIBRARY)
  MARK_AS_ADVANCED(MESQUITE_INCLUDE1)
  MARK_AS_ADVANCED(MESQUITE_INCLUDE2)
ELSE(WITH_MESQUITE OR MESQUITE_DIR)
  SET(HAVE_MESQUITE "NO" CACHE INTERNAL "Do not edit" FORCE)
  SET(DEF_MESQUITE "" CACHE INTERNAL "Do not edit" FORCE)
ENDIF(WITH_MESQUITE OR MESQUITE_DIR)

#----------------------------------------------------------------------------
# Search for Verdict 

IF(VERDICT_DIR)
  SET(VERDICT_DIR ${VERDICT_DIR} CACHE PATH "Path to verdict" FORCE)
  MARK_AS_ADVANCED(CLEAR VERDICT_DIR)
ENDIF(VERDICT_DIR)

OPTION(WITH_VERDICT "compile with verdict" OFF)
MARK_AS_ADVANCED(WITH_VERDICT)

IF(WITH_VERDICT)
  MARK_AS_ADVANCED(CLEAR WITH_VERDICT)
ENDIF(WITH_VERDICT)


IF(WITH_VERDICT OR VERDICT_DIR)

  MESSAGE(STATUS "Looking for VERDICT")

  SET(VERDICT_LIBRARY VERDICT_LIBRARY-NOTFOUND CACHE FILEPATH "Verdict library" FORCE)
  SET(VERDICT_INCLUDE VERDICT_INCLUDE-NOTFOUND CACHE FILEPATH "Verdict include" FORCE)

  IF(VERDICT_DIR)
    FIND_LIBRARY(VERDICT_LIBRARY 
      NAMES verdict110 
      PATHS ${VERDICT_DIR}
      ${VERDICT_DIR}/lib)

    FIND_PATH(VERDICT_INCLUDE verdict.h
      ${VERDICT_DIR}
      ${VERDICT_DIR}/include
			${VERDICT_DIR}/../include)

  ELSE(VERDICT_DIR)
    FIND_LIBRARY(VERDICT_LIBRARY 
      NAMES verdict110 
      PATHS ${SCIRUN_THIRDPARTY_DIR}/lib
      /usr/local/lib
      /usr/lib
      /lib)

    FIND_PATH(VERDICT_INCLUDE verdict.h
      ${SCIRUN_THIRDPARTY_DIR}/include
      /usr/local/include/
      /usr/include
      /include)
  
  ENDIF(VERDICT_DIR)

  IF(VERDICT_LIBRARY AND VERDICT_INCLUDE)
    SET(HAVE_VERDICT "YES" CACHE INTERNAL "Do not edit" FORCE)
    SET(DEF_VERDICT "#define HAVE_VERDICT 1" CACHE INTERNAL "Do not edit" FORCE)
    SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} ${VERDICT_INCLUDE})
  ELSE(VERDICT_LIBRARY AND VERDICT_INCLUDE)
    MESSAGE(FATAL_ERROR "Cannot find verdict library.") 
  ENDIF(VERDICT_LIBRARY AND VERDICT_INCLUDE)

  MARK_AS_ADVANCED(VERDICT_LIBRARY)
  MARK_AS_ADVANCED(VERDICT_INCLUDE)
ELSE(WITH_VERDICT OR VERDICT_DIR)
  SET(HAVE_VERDICT "NO" CACHE INTERNAL "Do not edit" FORCE)
  SET(DEF_VERDICT "" CACHE INTERNAL "Do not edit" FORCE)
ENDIF(WITH_VERDICT OR VERDICT_DIR)

#----------------------------------------------------------------------------
# Search for Tetgen (http://tetgen.berlios.de)

OPTION(WITH_TETGEN "Compile with TetGen" OFF)
MARK_AS_ADVANCED(WITH_TETGEN)

IF(WITH_TETGEN)
  MESSAGE(STATUS "Looking for TETGEN")

  FIND_PATH(TETGEN_DIR tetgen.h)
  
  IF (TETGEN_DIR) 
    FIND_LIBRARY(TETGEN_LIBRARY tet ${TETGEN_DIR}	${TETGEN_DIR}/lib)
    FIND_PATH(TETGEN_INCLUDE tetgen.h 
      ${TETGEN_DIR} 
      ${TETGEN_DIR}/include
      ${TETGEN_DIR}/../include)
  
    IF(TETGEN_LIBRARY AND TETGEN_INCLUDE)
      SET(HAVE_TETGEN "YES" CACHE INTERNAL "Do not edit" FORCE)
      SET(DEF_TETGEN "#define HAVE_TETGEN 1" CACHE INTERNAL "Do not edit" FORCE)
      SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} 
        ${TETGEN_INCLUDE})
    ELSE(TETGEN_LIBRARY AND TETGEN_INCLUDE)
      MESSAGE(FATAL_ERROR "Cannot find tetgen library.") 
    ENDIF(TETGEN_LIBRARY AND TETGEN_INCLUDE)
  ENDIF(TETGEN_DIR)

ELSE(WITH_TETGEN)
  SET(HAVE_TETGEN "NO" CACHE INTERNAL "Do not edit" FORCE)
  SET(DEF_TETGEN "" CACHE INTERNAL "Do not edit" FORCE)
ENDIF(WITH_TETGEN)


#----------------------------------------------------------------------------
# Search for Camal libraries.


IF(CAMAL_DIR)
  SET(CAMAL_DIR ${CAMAL_DIR} CACHE PATH "Path to camal" FORCE)
  MARK_AS_ADVANCED(CLEAR CAMAL_DIR)
ENDIF(CAMAL_DIR)

IF(CAMAL_DIR)

  MESSAGE(STATUS "Looking for CAMAL")

  SET(CAMAL_INCLUDE CAMAL_INCLUDE-NOTFOUND CACHE FILEPATH 
      "camal include" FORCE)

  SET(CAMAL_LIBRARY1 CAMAL_LIBRARY1-NOTFOUND CACHE FILEPATH 
      "one camal library" FORCE)
  FIND_LIBRARY(CAMAL_LIBRARY1 camal_tet 
               ${CAMAL_DIR}/lib/Linux NO_DEFAULT_PATH)

  IF (CAMAL_LIBRARY1)
    FIND_LIBRARY(CAMAL_LIBRARY2 CMLShr300 
                ${CAMAL_DIR}/lib/Linux NO_DEFAULT_PATH)
    FIND_LIBRARY(CAMAL_LIBRARY3 CMLUtl300 
               ${CAMAL_DIR}/lib/Linux NO_DEFAULT_PATH)

    FIND_LIBRARY(FORTRAN_LIBRARY g2c)

    SET(CAMAL_LIBRARIES ${CAMAL_LIBRARY1} 
                        ${CAMAL_LIBRARY2} ${CAMAL_LIBRARY3} 
                        ${FORTRAN_LIBRARY} CACHE FILEPATH 
        "camal libraries" FORCE)

    

  ELSE (CAMAL_LIBRARY1)
    SET(CAMAL_LIBRARIES CAMAL_LIBRARIES-NOTFOUND CACHE FILEPATH 
        "camal libraries" FORCE)
  ENDIF (CAMAL_LIBRARY1)


  FIND_PATH(CAMAL_INCLUDE CMLMesher.hpp
			${CAMAL_DIR}/include)
	
  IF(CAMAL_LIBRARIES AND CAMAL_INCLUDE)
    SET(HAVE_CAMAL "YES" CACHE INTERNAL "Do not edit" FORCE)
    SET(DEF_CAMAL "#define HAVE_CAMAL 1" CACHE INTERNAL "Do not edit" FORCE)
    SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} 
        ${CAMAL_INCLUDE})
  ELSE(CAMAL_LIBRARIES AND CAMAL_INCLUDE)
    MESSAGE(FATAL_ERROR "Cannot find camal libraries.") 
  ENDIF(CAMAL_LIBRARIES AND CAMAL_INCLUDE)

  MARK_AS_ADVANCED(CAMAL_LIBRARIES)
  MARK_AS_ADVANCED(CAMAL_INCLUDE)
ELSE(CAMAL_DIR)
  SET(HAVE_CAMAL "NO" CACHE INTERNAL "Do not edit" FORCE)
  SET(DEF_CAMAL "" CACHE INTERNAL "Do not edit" FORCE)
ENDIF(CAMAL_DIR)

#----------------------------------------------------------------------------
# Search for libgeotiff

IF(GEOTIFF_DIR)
  SET(GEOTIFF_DIR ${GEOTIFF_DIR} CACHE PATH "Path to libgeotiff" FORCE)
  MARK_AS_ADVANCED(CLEAR GEOTIFF_DIR)
ENDIF(GEOTIFF_DIR)

IF(GEOTIFF_DIR)

  MESSAGE(STATUS "Looking for GEOTIFF")

  SET(GEOTIFF_LIBRARY GEOTIFF_LIBRARY-NOTFOUND CACHE FILEPATH 
      "libgeotiff library" FORCE)
  SET(GEOTIFF_INCLUDE GEOTIFF_INCLUDE-NOTFOUND CACHE FILEPATH 
      "libgeotiff include" FORCE)

  FIND_LIBRARY(GEOTIFF_LIBRARY geotiff
	       ${GEOTIFF_DIR}
	       ${GEOTIFF_DIR}/lib)
  FIND_PATH(GEOTIFF_INCLUDE geotiff.h 
	    ${GEOTIFF_DIR}
	    ${GEOTIFF_DIR}/include
            ${GEOTIFF_DIR}/../include)
	
  IF(GEOTIFF_LIBRARY AND GEOTIFF_INCLUDE)
    SET(HAVE_GEOTIFF "YES" CACHE INTERNAL "Do not edit" FORCE)
    SET(DEF_GEOTIFF "#define HAVE_GEOTIFF 1" CACHE INTERNAL "Do not edit" FORCE)
    SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} 
        ${GEOTIFF_INCLUDE})
  ELSE(GEOTIFF_LIBRARY AND GEOTIFF_INCLUDE)
    MESSAGE(FATAL_ERROR "Cannot find libgeotiff library.") 
  ENDIF(GEOTIFF_LIBRARY AND GEOTIFF_INCLUDE)

  MARK_AS_ADVANCED(GEOTIFF_LIBRARY)
  MARK_AS_ADVANCED(GEOTIFF_INCLUDE)
ELSE(GEOTIFF_DIR)
  SET(HAVE_GEOTIFF "NO" CACHE INTERNAL "Do not edit" FORCE)
  SET(DEF_GEOTIFF "" CACHE INTERNAL "Do not edit" FORCE)
ENDIF(GEOTIFF_DIR)

#----------------------------------------------------------------------------
# Search for Blas

IF(BLAS_DIR)
  SET(BLAS_DIR ${BLAS_DIR} CACHE PATH "Path to Blas" FORCE)
  MARK_AS_ADVANCED(CLEAR BLAS_DIR)
ENDIF(BLAS_DIR)

IF(APPLE OR HAVE_LAPACK)
  OPTION(WITH_BLAS "compile with blas" ON)
  MARK_AS_ADVANCED(CLEAR WITH_BLAS)
ELSE(APPLE OR HAVE_LAPACK)
  OPTION(WITH_BLAS "compile with blas" OFF)
  MARK_AS_ADVANCED(WITH_BLAS)
ENDIF(APPLE OR HAVE_LAPACK)

IF(WITH_BLAS)
  MARK_AS_ADVANCED(CLEAR WITH_BLAS)
ENDIF(WITH_BLAS)

IF(WITH_BLAS OR BLAS_DIR)
  MESSAGE(STATUS "Looking for BLAS")

  IF(BLAS_DIR)
    FIND_LIBRARY(CBLAS_LIBRARY NAMES cblas PATHS ${BLAS_DIR} ${BLAS_DIR}/lib)
    FIND_LIBRARY(FBLAS_LIBRARY NAMES fblas f77blas blas PATHS ${BLAS_DIR} ${BLAS_DIR}/lib)
    FIND_LIBRARY(ATLAS_LIBRARY NAMES atlas PATHS ${BLAS_DIR} ${BLAS_DIR}/lib)

    FIND_PATH(CBLAS_INCLUDE cblas.h ${BLAS_DIR} ${BLAS_DIR}/include 
               ${BLAS_DIR}/../include)
  ELSE(BLAS_DIR)
    IF(LAPACK_DIR)
      # blas and lapack are traditionally in the same place
      FIND_LIBRARY(CBLAS_LIBRARY NAMES cblas PATHS ${LAPACK_DIR} ${LAPACK_DIR}/lib)
      FIND_LIBRARY(FBLAS_LIBRARY NAMES fblas f77blas blas PATHS ${LAPACK_DIR} ${LAPACK_DIR}/lib)
      FIND_LIBRARY(ATLAS_LIBRARY NAMES atlas PATHS ${LAPACK_DIR} ${LAPACK_DIR}/lib)
      FIND_PATH(CBLAS_INCLUDE cblas.h ${LAPACK_DIR} ${LAPACK_DIR}/include)
    ELSE(LAPACK_DIR)
      FIND_LIBRARY(CBLAS_LIBRARY NAMES cblas PATHS /usr/local/lib /usr/lib /lib)
      FIND_LIBRARY(FBLAS_LIBRARY NAMES fblas f77blas blas PATHS /usr/local/lib /usr/lib /lib)
      FIND_LIBRARY(ATLAS_LIBRARY NAMES atlas PATHS /usr/local/lib /usr/lib /lib)
      FIND_PATH(CBLAS_INCLUDE cblas.h /usr/local/include/ /usr/include /include)
    ENDIF(LAPACK_DIR)
  ENDIF(BLAS_DIR)

  IF(CBLAS_LIBRARY AND CBLAS_INCLUDE)
    MESSAGE(STATUS "Looking for BLAS - found CBLAS")
    SET(HAVE_CBLAS "YES" CACHE INTERNAL "Do not edit" FORCE)
    SET(DEF_CBLAS "#define HAVE_BLAS 1" CACHE INTERNAL "Do not edit" FORCE)
    SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} ${BLAS_INCLUDE})
  ELSE(CBLAS_LIBRARY AND CBLAS_INCLUDE)
    MESSAGE(STATUS "Looking for BLAS - did not find CBLAS")
  ENDIF(CBLAS_LIBRARY AND CBLAS_INCLUDE)

  IF(FBLAS_LIBRARY)
    MESSAGE(STATUS "Looking for BLAS - found FBLAS")
  ELSE(FBLAS_LIBRARY)
    MESSAGE(STATUS "Looking for BLAS - did not find FBLAS")
  ENDIF(FBLAS_LIBRARY)

  IF(ATLAS_LIBRARY)
    MESSAGE(STATUS "Looking for BLAS - found atlas")
  ELSE(ATLAS_LIBRARY)
    MESSAGE(STATUS "Looking for BLAS - did not find atlas")
  ENDIF(ATLAS_LIBRARY)

  IF(ATLAS_LIBRARY AND FBLAS_LIBRARY)
    SET(FBLAS_LIBRARY ${FBLAS_LIBRARY} ${ATLAS_LIBRARY} CACHE INTERNAL "fortran blas with atlas")
  ENDIF(ATLAS_LIBRARY AND FBLAS_LIBRARY)
  IF(ATLAS_LIBRARY AND CBLAS_LIBRARY)
    SET(CBLAS_LIBRARY ${CBLAS_LIBRARY} ${ATLAS_LIBRARY} CACHE INTERNAL "c blas with atlas")
  ENDIF(ATLAS_LIBRARY AND CBLAS_LIBRARY)

  IF(CMAKE_SYSTEM MATCHES "IRIX.*")
    FIND_LIBARY(FBLAS_LIBRARY complib.sgimath PATHS /usr/local/lib /usr/lib /lib)
  ELSE(CMAKE_SYSTEM MATCHES "IRIX.*")
    # dec needs cxml
    IF(CMAKE_SYSTEM MATCHES "OSF.*")
      FIND_LIBARY(FBLAS_LIBRARY cxml PATHS /usr/local/lib /usr/lib /lib)
    ELSE(CMAKE_SYSTEM MATCHES "OSF.*")
      # aix needs essl
      IF(CMAKE_SYSTEM MATCHES "AIX.*")
        FIND_LIBARY(FBLAS_LIBRARY essl PATHS /usr/local/lib /usr/lib /lib)
      ELSE(CMAKE_SYSTEM MATCHES "AIX.*")
        IF(APPLE)
          SET(CBLAS_LIBRARY "-framework vecLib" CACHE STRING "CBLAS library" FORCE)
          SET(FBLAS_LIBRARY "-framework vecLib" CACHE STRING "FBLAS library" FORCE)
          SET(HAVE_BLAS "YES" CACHE INTERNAL "Do not edit" FORCE)
          SET(DEF_BLAS "#define HAVE_BLAS 1" CACHE INTERNAL "Do not edit" FORCE)
        ENDIF(APPLE)
      ENDIF(CMAKE_SYSTEM MATCHES "AIX.*")
    ENDIF(CMAKE_SYSTEM MATCHES "OSF.*")
  ENDIF(CMAKE_SYSTEM MATCHES "IRIX.*")

  MARK_AS_ADVANCED(CBLAS_LIBRARY)
  MARK_AS_ADVANCED(FBLAS_LIBRARY)
  MARK_AS_ADVANCED(ATLAS_LIBRARY)
  MARK_AS_ADVANCED(CBLAS_INCLUDE)
ELSE(WITH_BLAS OR BLAS_DIR)
  SET(HAVE_BLAS "NO" CACHE INTERNAL "Do not edit" FORCE)
  SET(DEF_BLAS "" CACHE INTERNAL "Do not edit" FORCE)
ENDIF(WITH_BLAS OR BLAS_DIR)


#----------------------------------------------------------------------------
# Insight Toolkit

# Look for it in the Thirdparty
IF(EXISTS ${SCIRUN_THIRDPARTY_DIR}/lib/InsightToolkit)
  MESSAGE(STATUS "Found ITK in SCIRun Thirdparty: ${SCIRUN_THIRDPARTY_DIR}/lib/InsightToolkit")
  SET(ITK_DIR ${SCIRUN_THIRDPARTY_DIR}/lib/InsightToolkit CACHE PATH "Path to IInsight Toolkit")
  MESSAGE(STATUS "Setting ITK_DIR=${ITK_DIR}")
  SET(FIND_ITK 1 CACHE INTERNAL "Do not edit" FORCE)
ENDIF(EXISTS ${SCIRUN_THIRDPARTY_DIR}/lib/InsightToolkit)

STRING(REGEX MATCH "Insight" BUILDING_INSIGHT ${LOAD_PACKAGE})
IF(ITK_DIR)
  SET(ITK_DIR ${ITK_DIR} CACHE PATH "Path to Insight Toolkit")
  # Check if Insight in LOAD_PACKAGE
  IF(NOT BUILDING_INSIGHT)
    MESSAGE(STATUS "ITK_DIR specified as ${ITK_DIR} but Insight not included in LOAD_PACKAGE. Please add Insight to LOAD_PACKAGE.")
    #SET(LOAD_PACKAGE "${LOAD_PACKAGE},Insight")
    MESSAGE(STATUS "LoadPackage=${LOAD_PACKAGE}")		
  ELSE(NOT BUILDING_INSIGHT)

  ENDIF(NOT BUILDING_INSIGHT)
ELSE(ITK_DIR)
  # Force user to define it if they specified Insight in LOAD_PACKAGE
  IF(BUILDING_INSIGHT)
    MESSAGE(SEND_ERROR "Insight specified in LOAD_PACKAGE, but ITK_DIR not set and can't find it in Thirdparty. Please set ITK to the directory that contains the ITKConfig.cmake file")
    SET(FIND_ITK 1)
  ENDIF(BUILDING_INSIGHT)
ENDIF(ITK_DIR)

IF(ITK_DIR OR FIND_ITK)
  # Turn building Seg3D on
  FIND_PACKAGE(ITK)
	
	IF(NOT ITK_FOUND)
		IF(EXISTS ${ITK_DIR}/lib/InsightToolkit)
			SET(ITK_DIR ${ITK_DIR}/lib/InsightToolkit)
                        FIND_PACKAGE(ITK)
		ENDIF(EXISTS ${ITK_DIR}/lib/InsightToolkit)
	ENDIF(NOT ITK_FOUND)	
	
  IF(ITK_FOUND)
    INCLUDE(${ITK_USE_FILE})
    SET(DEF_INSIGHT "#define HAVE_INSIGHT 1" CACHE INTERNAL "Have insight")
    SET(HAVE_INSIGHT "yes" CACHE INTERNAL "Linking against ITK")
  ELSE(ITK_FOUND)
    MESSAGE(FATAL_ERROR "Cannot find ITK. Please set ITK_DIR.")
  ENDIF(ITK_FOUND)
ENDIF(ITK_DIR OR FIND_ITK)

# Set which components of wx we need, as there can be different parts
# compiled.
OPTION(WITH_WXWIDGETS "Compile Seg3D with WX Widgets support" ON)
MARK_AS_ADVANCED(WITH_WXWIDGETS)
IF(WITH_WXWIDGETS)
  SET(wxWidgets_USE_LIBS core base gl adv html xml xrc)
  FIND_PACKAGE(wxWidgets)

  IF(wxWidgets_FOUND)
    # Without this magic line wxWidgets fails to compile on OSX.  It dies
    # with a bad template error on wxString.
    SET(wxWidgets_INCLUDE_DIRS_NO_SYSTEM 1)

    SET(DEF_WX "#define HAVE_WX 1" CACHE INTERNAL "Have WX")
    INCLUDE(${wxWidgets_USE_FILE})
  ENDIF(wxWidgets_FOUND)
ENDIF(WITH_WXWIDGETS)

#----------------------------------------------------------------------------
# Search for HDF5 (http://www.hdfgroup.org)

IF(HDF5_DIR)
  SET(HDF5_DIR ${HDF5_DIR} CACHE PATH "Path to HDF5" FORCE)
  MARK_AS_ADVANCED(CLEAR HDF5_DIR)
ENDIF(HDF5_DIR)

IF(HDF5_DIR)

  MESSAGE(STATUS "Configuring SCIRun with HDF5")

  SET(HDF5_LIBRARY HDF5_LIBRARY-NOTFOUND CACHE FILEPATH 
      "HDF5 library" FORCE)
  SET(HDF5_INCLUDE HDF5_INCLUDE-NOTFOUND CACHE FILEPATH 
      "HDF5 include" FORCE)

	IF(HDF5_DIR)
    FIND_LIBRARY(HDF5_LIBRARY hdf5 
			${HDF5_DIR}
			${HDF5_DIR}/lib)
    FIND_PATH(HDF5_INCLUDE hdf5.h 
			${HDF5_DIR}
			${HDF5_DIR}/include
			${HDF5_DIR}/../include)
	ENDIF(HDF5_DIR)
	
  IF(HDF5_LIBRARY AND HDF5_INCLUDE)
    SET(HAVE_HDF5 "YES" CACHE INTERNAL "Do not edit" FORCE)
    SET(DEF_HDF5 "#define HAVE_HDF5 1" CACHE INTERNAL "Do not edit" FORCE)
    SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} 
        ${HDF5_INCLUDE})
  ELSE(HDF5_LIBRARY AND HDF5_INCLUDE)
    MESSAGE(FATAL_ERROR "Cannot find HDF5 library.") 
  ENDIF(HDF5_LIBRARY AND HDF5_INCLUDE)

  MARK_AS_ADVANCED(HDF5_LIBRARY)
  MARK_AS_ADVANCED(HDF5_INCLUDE)
ELSE(HDF5_DIR)
  SET(HAVE_HDF5 "NO" CACHE INTERNAL "Do not edit" FORCE)
  SET(DEF_HDF5 "" CACHE INTERNAL "Do not edit" FORCE)
ENDIF(HDF5_DIR)

#----------------------------------------------------------------------------
# GDCM Library (which could be pulled from ITK or specified individually)
IF(ITK_FOUND OR GDCM_DIR)
  OPTION(WITH_GDCM "Compile with GDCM support" ON)
ELSE(ITK_FOUND OR GDCM_DIR)
  OPTION(WITH_GDCM "Compile with GDCM support" OFF)
ENDIF(ITK_FOUND OR GDCM_DIR)
MARK_AS_ADVANCED(FORCE WITH_GDCM)

IF(WITH_GDCM)
  FIND_LIBRARY(GDCM_LIBRARY gdcm itkgdcm
    PATHS ${SCIRUN_THIRDPARTY_DIR}/lib/InsightToolkit
    ${ITK_DIR}/bin
    PATHS ${SCIRUN_THIRDPARTY_DIR}/lib/gdcm
    NO_DEFAULT_PATH)
  FIND_PATH(GDCM_INCLUDE gdcm.h
    PATHS ${SCIRUN_THIRDPARTY_DIR}/include/InsightToolkit/gdcm/src/
    ${ITK_SOURCE_DIR}/Utilities/gdcm/src
    ${SCIRUN_THIRDPARTY_DIR}/include/gdcm
    NO_DEFAULT_PATH)
  FIND_PATH(GDCM_CONFIGURE_INCLUDE gdcmConfigure.h
    PATHS ${SCIRUN_THIRDPARTY_DIR}/include/InsightToolkit/gdcm/src/
    ${ITK_SOURCE_DIR}/Utilities/gdcm/src
    ${SCIRUN_THIRDPARTY_DIR}/include/gdcm
    NO_DEFAULT_PATH)
  IF(GDCM_LIBRARY)
    SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} ${GDCM_INCLUDE} ${GDCM_CONFIGURE_INCLUDE})
    SET(DEF_GDCM "#define HAVE_GDCM 1")
  ELSE(GDCM_LIBRARY)
    SET(DEF_GDCM "#define HAVE_GDCM 0")
  ENDIF(GDCM_LIBRARY)
ENDIF(WITH_GDCM)

MARK_AS_ADVANCED(WITH_GDCM GDCM_INCLUDE GDCM_CONFIGURE_INCLUDE GDCM_LIBRARY)


#-----------------------------------------------------------------------------
# The entire SCIRun tree should use the same include path

# This should be the only INCLUDE_DIRECTORIES command in the entire
# tree.  We need to do this in one place to make sure the order is correct.
#-----------------------------------------------------------------------------
# Include directories for other projects installed on the system.
SET(SCIRUN_INCLUDE_DIRS_SYSTEM 
  ${SCIRUN_CORE_SOURCE_DIR}/
  ${SCIRUN_CORE_SOURCE_DIR}/include/
)

# Set include directories for sci_defs
SET(SCIRUN_INCLUDE_DIRS_BUILD_TREE 
  ${CMAKE_CURRENT_BINARY_DIR}/
  ${CMAKE_CURRENT_BINARY_DIR}/include/
  ${CMAKE_CURRENT_BINARY_DIR}/include/sci_defs/
  ${CMAKE_CURRENT_SOURCE_DIR}/externals/glew/include
  ${CMAKE_CURRENT_SOURCE_DIR}/externals/	
)

INCLUDE_DIRECTORIES(
  ${SCIRUN_INCLUDE_THIRDPARTY}
  ${SCIRUN_INCLUDE_DIRS_BUILD_TREE}
  ${SCIRUN_INCLUDE_DIRS_SYSTEM}
  ${SCIRUN_INCLUDE_ADDITIONAL}
)


#----------------------------------------------------------------------------
# for now set make install directory to current binary directory
SET(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})

# Help other projects use SCIRun Core

# Copy the UseSCIRUN.cmake file to the binary tree for backward compatability.
CONFIGURE_FILE(${SCIRUN_CORE_SOURCE_DIR}/UseSCIRUN.cmake.in
               ${SCIRUN_CORE_BINARY_DIR}/UseSCIRUN.cmake COPYONLY IMMEDIATE)
 
# Save the compiler settings so another project can import them.
INCLUDE(${CMAKE_ROOT}/Modules/CMakeExportBuildSettings.cmake)
CMAKE_EXPORT_BUILD_SETTINGS(${SCIRUN_CORE_BINARY_DIR}/SCIRUNBuildSettings.cmake)
INSTALL_FILES(/lib/SCIRun .cmake SCIRUNBuildSettings)

# Save library dependencies.
EXPORT_LIBRARY_DEPENDENCIES(${SCIRUN_CORE_BINARY_DIR}/SCIRUNLibraryDepends.cmake)

# Create the SCIRUNConfig.cmake file containing the SCIRUN configuration.
INCLUDE (${SCIRUN_CORE_SOURCE_DIR}/sciGenerateSCIRUNConfig.cmake)

# Install some files.
#INSTALL_FILES(/include/SCIRun .h sciConfigure)
INSTALL_FILES(/lib/SCIRun .cmake UseSCIRUN SCIRUNLibraryDepends)

OPTION(MINIMAL_CORE "Build only a small subset of Core" OFF)
MARK_AS_ADVANCED(MINIMAL_CORE) #shouldn't be exposed to most users.

# Build the external include library 
SET(SCI_GLEW_LIBRARY sci_glew)
IF(NOT MINIMAL_CORE)
  ADD_SUBDIRECTORY(externals/glew)
ENDIF(NOT MINIMAL_CORE)

SET(SLIVRG_LIBRARY slivrG)
SET(SLIVRV_LIBRARY slivrV)

SET(SCIRUN_SLIVR_BUILD true)
SET(SLIVR_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/externals/slivr)
ADD_SUBDIRECTORY(externals/slivr)

# traverse the tree
ADD_SUBDIRECTORY(include)
ADD_SUBDIRECTORY(Core)

OPTION(BUILD_UTILS "Build convert and util apps" ON)
MARK_AS_ADVANCED(BUILD_UTILS)

IF(BUILD_UTILS)
  ADD_SUBDIRECTORY(StandAlone)
ENDIF(BUILD_UTILS)

MARK_AS_ADVANCED(BLT_INCLUDE BLT_LIBRARY FREETYPE_INCLUDE FREETYPE_LIBRARY TCL_LIBRARY TCL_INCLUDE ITCL_INCLUDE ITCL_LIBRARY ITK_INCLUDE ITK_LIBRARY TK_INCLUDE TK_LIBRARY PNG_INCLUDE PNG_LIBRARY ZLIB_INCLUDE ZLIB_LIBRARY XML2_LIBRARY XML2_INCLUDE TEEM_LIBRARY TEEM_INCLUDE_DIR MPEG_LIBRARY MPEG_INCLUDE M_LIBRARY WITH_MPEG SLIVRG_LIBRARY SLIVRV_LIBRARY SLIVR_INCLUDE_DIR)

# Option for building Seg3D
SET(DEFAULT_SEG3D OFF)

IF(ITK_FOUND AND wxWidgets_FOUND)
  MESSAGE(STATUS "Building Seg3D by default because ITK and WX found.")
  SET(DEFAULT_SEG3D ON)
ENDIF(ITK_FOUND AND wxWidgets_FOUND)

OPTION(BUILD_SEG3D "Build Seg3D Application" ${DEFAULT_SEG3D})

IF(BUILD_SEG3D)
  SET(SCIRUN_DIR ${SCIRUN_CORE_BINARY_DIR})
  MARK_AS_ADVANCED(SCIRUN_DIR)
  ADD_SUBDIRECTORY(StandAlone/Apps/Seg3D)
  IF(ITK_FOUND)

  ELSE(ITK_FOUND)
    MESSAGE(SEND_ERROR "BUILD_SEG3D set to ON, but ITK not specified. Please make sure ITK_DIR is set to be the directory where the ITKConfig.cmake file is located or make sure to install ITK into your ${SCIRUN_THIRDPARTY_DIR} directory.")
  ENDIF(ITK_FOUND)
ENDIF(BUILD_SEG3D)
MARK_AS_ADVANCED(BUILD_SEG3D)

# Option for building scirun executable, dataflow, packages
OPTION(BUILD_DATAFLOW "Build SCIRun executable, dataflow, and packages" ON)
IF(BUILD_DATAFLOW)
  SET(SCIRUN_DIR ${SCIRUN_CORE_BINARY_DIR})
  MARK_AS_ADVANCED(SCIRUN_DIR)
  ADD_SUBDIRECTORY(Dataflow)
ENDIF(BUILD_DATAFLOW)
MARK_AS_ADVANCED(BUILD_DATAFLOW)

# Option for building scirun python interface
OPTION(BUILD_SR_PY "Build SCIRun Python interface" OFF)
IF(BUILD_SR_PY)
  FIND_PACKAGE(PythonLibs)
  ADD_SUBDIRECTORY(sr_py)
ENDIF(BUILD_SR_PY)
MARK_AS_ADVANCED(BUILD_SR_PY)

MARK_AS_ADVANCED(CMAKE_INSTALL_PREFIX DART_TESTING_TIMEOUT)
MARK_AS_ADVANCED(CMAKE_BACKWARDS_COMPATIBILITY)

# Traverse Testing directory
IF(BUILD_TESTING)
  ADD_SUBDIRECTORY(Testing)
ENDIF(BUILD_TESTING)

IF (NOT WIN32)
  ADD_CUSTOM_TARGET(copy_scripts ALL
    COMMAND cp ${SCIRUN_CORE_SOURCE_DIR}/scripts/run_unit_tests.sh ${SCIRUN_CORE_BINARY_DIR}/run_unit_tests.sh
    COMMAND chmod +x ${SCIRUN_CORE_BINARY_DIR}/run_unit_tests.sh
    COMMAND cp ${SCIRUN_CORE_SOURCE_DIR}/scripts/run_sample_tests.sh ${SCIRUN_CORE_BINARY_DIR}/run_sample_tests.sh
    COMMAND chmod +x ${SCIRUN_CORE_BINARY_DIR}/run_sample_tests.sh
    COMMAND cp ${SCIRUN_CORE_SOURCE_DIR}/scripts/run_all_tests.sh ${SCIRUN_CORE_BINARY_DIR}/run_all_tests.sh
    COMMAND chmod +x ${SCIRUN_CORE_BINARY_DIR}/run_all_tests.sh
#    COMMENT "Copying scripts for testing/using dashboard"
    )
ENDIF(NOT WIN32)

