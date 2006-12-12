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

# This is the root SCIRun CMakeLists file.
CMAKE_MINIMUM_REQUIRED(VERSION 2.2)

# DV Change to SCIRun Core or something?
PROJECT(SCIRUN_CORE)

#-----------------------------------------------------------------------------
# Test for some required system information.
INCLUDE (CMakeBackwardCompatibilityC)

#-----------------------------------------------------------------------------
# Configure testing support.  This should be done before any
# MESSAGE(FATAL_ERROR ...) commands are invoked.
IF(NOT BUILD_FROM_TOP_LEVEL)
  INCLUDE( CTest )
  IF(BUILD_TESTING)
    ENABLE_TESTING()
    SET(BUILDNAME "${BUILDNAME}" CACHE STRING "Name of build on the dashboard")
    MARK_AS_ADVANCED(BUILDNAME)
  ENDIF(BUILD_TESTING)
ENDIF(NOT BUILD_FROM_TOP_LEVEL)

SET(SCIRUN_CORE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

#----------------------------------------------------------------------------
# Output directories.
IF(NOT BUILD_FROM_TOP_LEVEL)
  SET (LIBRARY_OUTPUT_PATH ${SCIRUN_CORE_BINARY_DIR}/lib CACHE INTERNAL "Single output directory for building all libraries.")
  SET (EXECUTABLE_OUTPUT_PATH ${SCIRUN_CORE_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all executables.")
  #output directory for installing all include files.")
  MARK_AS_ADVANCED(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)
ENDIF(NOT BUILD_FROM_TOP_LEVEL)


#-----------------------------------------------------------------------------
# Force user to use our Thirdparty, and verify for looking for
# SCIRUN_THIRDPARTY_VERSION file

# If not defined, check environment
SET(SCIRUN_THIRDPARTY_PATH_SET 0)
IF(NOT SCIRUN_THIRDPARTY_PATH)
  SET(SCIRUN_THIRDPARTY_PATH_ENV "$ENV{SCIRUN_THIRDPARTY_PATH}")
  IF(SCIRUN_THIRDPARTY_PATH_ENV)
    SET(SCIRUN_THIRDPARTY_PATH ${SCIRUN_THIRDPARTY_PATH_ENV} CACHE PATH "Path to SCIRun's Thirdparty")
    SET(SCIRUN_THIRDPARTY_PATH_SET 1)
  ENDIF(SCIRUN_THIRDPARTY_PATH_ENV)
ELSE(NOT SCIRUN_THIRDPARTY_PATH)
  SET(SCIRUN_THIRDPARTY_PATH_SET 1)
ENDIF(NOT SCIRUN_THIRDPARTY_PATH)

IF(NOT SCIRUN_THIRDPARTY_PATH_SET)
  SET(SCIRUN_THIRDPARTY_PATH "." CACHE PATH "Path to SCIRun's Thirdparty")
  MESSAGE(FATAL_ERROR "\n\nSCIRUN_THIRDPARTY_PATH must be specified to build. Please re-run cmake and specify a valid path to the SCIRun Thirdparty (version ${SCIRUN_VERSION_STRING}) using the SCIRUN_THIRDPARTY_PATH variable. This path must contain the SCIRUN_THIRDPARTY_VERSION file.\n\n")
ENDIF(NOT SCIRUN_THIRDPARTY_PATH_SET)

FIND_PATH(SCIRUN_THIRDPARTY_PATH_CHECK SCIRUN_THIRDPARTY_VERSION ${SCIRUN_THIRDPARTY_PATH})
MARK_AS_ADVANCED(SCIRUN_THIRDPARTY_PATH_CHECK)

IF(NOT SCIRUN_THIRDPARTY_PATH_CHECK)
  MESSAGE(FATAL_ERROR "Invalid SCIRUN_THIRDPARTY_PATH ${SCIRUN_THIRDPARTY_PATH}. Cannot find ${SCIRUN_THIRDPARTY_PATH}/SCIRUN_THIRDPARTY_VERSION. Please re-run cmake and specify a valid path.")
ELSE(NOT SCIRUN_THIRDPARTY_PATH_CHECK)
  FILE(READ ${SCIRUN_THIRDPARTY_PATH}/SCIRUN_THIRDPARTY_VERSION 
    SCIRUN_THIRDPARTY_VERSION_STRING)
  STRING(REGEX MATCH "SCIRUN_THIRDPARTY_VERSION=${SCIRUN_VERSION_STRING}" THIRDPARTY_VERSION_CHECK ${SCIRUN_THIRDPARTY_VERSION_STRING})
  IF(THIRDPARTY_VERSION_CHECK)
  ELSE(THIRDPARTY_VERSION_CHECK)
    MESSAGE("SCIRUN_THIRDPARTY_VERSION file does not contain correct version of ${SCIRUN_VERSION_STRING}")
  ENDIF(THIRDPARTY_VERSION_CHECK)
ENDIF(NOT SCIRUN_THIRDPARTY_PATH_CHECK)

SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_THIRDPARTY_PATH}/include)

# Force SCIRUN_THIRDPARTY_PATH to be cached
SET(SCIRUN_THIRDPARTY_PATH ${SCIRUN_THIRPDARTY_PATH} CACHE PATH "Path to SCIRun's Thirdparty.")

#DV FIX HARD CODED VERSION
#SET(COMPILER_VERS unknown)
#
#IF(CMAKE_COMPILER_IS_GNUCXX)
#  EXEC_PROGRAM(${CMAKE_CXX_COMPILER} ARGS --version OUTPUT_VARIABLE CMAKE_CXX_COMPILER_VERSION)
#  IF(CMAKE_CXX_COMPILER_VERSION)
#    SET(COMPILER_VERS ${CMAKE_CXX_COMPILER_VERSION})
#  ENDIF(CMAKE_CXX_COMPILER_VERSION)
#ENDIF(CMAKE_COMPILER_IS_GNUCXX) 

#SCIRUN_THIRDPARTY_OPTION( 
#  "${SCIRUN_VERSION_MAJOR}.${SCIRUN_VERSION_MINOR}.${SCIRUN_VERSION_PATCH}"
#  ${COMPILER_VERS}
#  ${NBITS}
#)

#IF(SCIRUN_THIRDPARTY_FOUND)
#  SET(SCIRUN_INCLUDE_THIRDPARTY 
#    ${SCIRUN_THIRDPARTY_PATH}/include
#  )	
#ELSE(SCIRUN_THIRDPARTY_FOUND)
#  MESSAGE(FATAL_ERROR
#    "Cannot build SCIRun without Thirdparty. Please set SCIRUN_THIRDPARTY_PATH.")
#ENDIF(SCIRUN_THIRDPARTY_FOUND)

#-----------------------------------------------------------------------------
# SCIRun Packages to enable (must enable packages when building core for now)
# DV Fix need error checking to make sure SCIRun gets included
SET(LOAD_PACKAGE "SCIRun,BioPSE,Teem,MatlabInterface" CACHE STRING "Building SCIRun, BioPSE, Teem, and MatlabInterface packages by default.")

# ----------------------------------------------------------------------------
# SCIRun build configuration options
IF(NOT BUILD_FROM_TOP_LEVEL)
  OPTION(BUILD_SHARED_LIBS "Build SCIRun with shared libraries." ON)
  MARK_AS_ADVANCED(BUILD_SHARED_LIBS)
ENDIF(NOT BUILD_FROM_TOP_LEVEL)

# Add correct flags for win32
IF(WIN32)
  IF(NOT BUILD_SHARED_LIBS)
    ADD_DEFINITIONS(-DBUILD_CORE_STATIC)
  ENDIF(NOT BUILD_SHARED_LIBS)
#  This defines M_PI for windows
  ADD_DEFINITIONS(-D_USE_MATH_DEFINES)
  ADD_DEFINITIONS(-D_WIN32_WINNT=0x0500)
  # win32 l3 warnings are a little ridiculous
  SET(CMAKE_CXX_WARNING_LEVEL 1)
  IF(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    STRING(REGEX REPLACE "/W[0-4]" "/W1"
            CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  ELSE(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W1")
  ENDIF(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
ENDIF(WIN32)


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
  PATHS ${SCIRUN_THIRDPARTY_PATH}/lib/
  NO_DEFAULT_PATH)

FIND_PATH(ZLIB_INCLUDE zlib.h
  ${SCIRUN_THIRDPARTY_PATH}/include/
  NO_DEFAULT_PATH)

IF(ZLIB_LIBRARY)
   MESSAGE(STATUS "Found ZLIB: ${ZLIB_LIBRARY}")
ELSE(ZLIB_LIBRARY)
   MESSAGE(FATAL_ERROR ": z library required.")
ENDIF(ZLIB_LIBRARY)

#-----------------------------------------------------------------------------
# png lib in Thirdparty
FIND_LIBRARY(PNG_LIBRARY png
             PATHS ${SCIRUN_THIRDPARTY_PATH}/lib/
             NO_DEFAULT_PATH)

FIND_PATH(PNG_INCLUDE png.h
          ${SCIRUN_THIRDPARTY_PATH}/include/
          NO_DEFAULT_PATH)

IF(PNG_LIBRARY)
   MESSAGE(STATUS "Found PNG: ${PNG_LIBRARY}")
   SET(HAVE_PNG "YES")
   SET(DEF_PNG "#define HAVE_PNG 1")
ELSE(PNG_LIBRARY)
   MESSAGE(FATAL_ERROR ": png library required.")
ENDIF(PNG_LIBRARY)


#-----------------------------------------------------------------------------
# XML2 in Thirdparty
FIND_LIBRARY(XML2_LIBRARY xml2
            PATHS ${SCIRUN_THIRDPARTY_PATH}/lib/
            NO_DEFAULT_PATH)

FIND_PATH(XML2_INCLUDE libxml/xmlreader.h
           ${SCIRUN_THIRDPARTY_PATH}/include/libxml2/
           ${SCIRUN_THIRDPARTY_PATH}/include/
           NO_DEFAULT_PATH)
SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} ${XML2_INCLUDE})

#-----------------------------------------------------------------------------
# Tcl in Thirdparty
FIND_LIBRARY(TCL_LIBRARY tcl
             PATHS ${SCIRUN_THIRDPARTY_PATH}/lib/
             NO_DEFAULT_PATH)

FIND_PATH(TCL_INCLUDE1 tcl.h
           ${SCIRUN_THIRDPARTY_PATH}/include
           NO_DEFAULT_PATH)

FIND_PATH(TCL_INCLUDE2 tclMath.h
           ${SCIRUN_THIRDPARTY_PATH}/src/tcl/generic
           ${SCIRUN_THIRDPARTY_PATH}/include
           NO_DEFAULT_PATH)
MARK_AS_ADVANCED(TCL_INCLUDE1 TCL_INCLUDE2)

# Tk in Thirdparty
FIND_LIBRARY(TK_LIBRARY tk
             PATHS ${SCIRUN_THIRDPARTY_PATH}/lib/
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
             PATHS ${SCIRUN_THIRDPARTY_PATH}/lib/
             NO_DEFAULT_PATH)

FIND_PATH(ITK_INCLUDE itk.h
           ${SCIRUN_THIRDPARTY_PATH}/include
           NO_DEFAULT_PATH)

#-----------------------------------------------------------------------------
# ITCL library in Thirdparty
FIND_LIBRARY(ITCL_LIBRARY itcl
             PATHS ${SCIRUN_THIRDPARTY_PATH}/lib/
             NO_DEFAULT_PATH)

FIND_PATH(ITCL_INCLUDE itcl.h
           ${SCIRUN_THIRDPARTY_PATH}/include
           NO_DEFAULT_PATH)

#-----------------------------------------------------------------------------
# blt lib in Thirdparty
FIND_LIBRARY(BLT_LIBRARY BLT
             PATHS ${SCIRUN_THIRDPARTY_PATH}/lib/
             NO_DEFAULT_PATH)

FIND_PATH(BLT_INCLUDE blt.h
           ${SCIRUN_THIRDPARTY_PATH}/include
           NO_DEFAULT_PATH)

IF(WIN32)
  SET(BLT_LIBRARY ${BLT_LIBRARY} gdi32 winspool comdlg32 user32 advapi32)
ENDIF(WIN32)

#-----------------------------------------------------------------------------
# Freetype in Thirdparty
FIND_LIBRARY(FREETYPE_LIBRARY 
             NAMES freetype libfreetype.so libfreetype.a
             PATHS ${SCIRUN_THIRDPARTY_PATH}/lib/
             NO_DEFAULT_PATH)

FIND_PATH(FREETYPE_INCLUDE freetype/freetype.h
           ${SCIRUN_THIRDPARTY_PATH}/include/freetype2/
           ${SCIRUN_THIRDPARTY_PATH}/include/
           NO_DEFAULT_PATH)

SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} ${FREETYPE_INCLUDE})
SET(DEF_FREETYPE "#define HAVE_FREETYPE 1")

#-----------------------------------------------------------------------------
# Glew in Thirdparty
FIND_LIBRARY(GLEW_LIBRARY 
             NAMES GLEW glew32
             PATHS ${SCIRUN_THIRDPARTY_PATH}/lib/
             NO_DEFAULT_PATH)

FIND_PATH(GLEW_INCLUDE GL/glew.h
           ${SCIRUN_THIRDPARTY_PATH}/include
           NO_DEFAULT_PATH)

SET(DEF_GLEW "#define HAVE_GLEW 1")

#-----------------------------------------------------------------------------
# MPEG sometimes in Thirdparty (not in Windows?)
IF(WIN32)
  OPTION(WITH_MPEG "Build SCIRun with MPEG support" OFF)
ELSE(WIN32)
  OPTION(WITH_MPEG "Build SCIRun with MPEG support" ON)
ENDIF(WIN32)

IF(WITH_MPEG)
  FIND_LIBRARY(MPEG_LIBRARY mpege
              PATHS ${SCIRUN_THIRDPARTY_PATH}/lib/
              /usr/local/lib
              /usr/lib
              /lib
              NO_DEFAULT_PATH)

  FIND_PATH(MPEG_INCLUDE mpege.h
             ${SCIRUN_THIRDPARTY_PATH}/include
             /usr/local/include
             /usr/include
             /include
             NO_DEFAULT_PATH)

  IF(MPEG_LIBRARY)
    SET(HAVE_MPEG "1")
    SET(DEF_MPEG "#define HAVE_MPEG 1")
    MESSAGE(STATUS "Found MPEG: ${MPEG_LIBRARY}")
    SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${MPEG_INCLUDE})
  ELSE(MPEG_LIBRARY)
    SET(HAVE_MPEG "0")
    MESSAGE(STATUS "MPEG not found")
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
# lapack library
OPTION(WITH_LAPACK "Build with lapack library" OFF)
MARK_AS_ADVANCED(WITH_LAPACK)		   
IF(WITH_LAPACK)		   
  FIND_LIBRARY(LAPACK_LIBRARY 
    NAMES lapack clapack
    PATHS /usr/local/lib
    /usr/lib
    /lib)

   IF(LAPACK_LIBRARY)
     SET(HAVE_LAPACK "YES")
   ELSE(LAPACK_LIBRARY)
     SET(HAVE_LAPACK "NO")
   ENDIF(LAPACK_LIBRARY)
ELSE(WITH_LAPACK)
   SET(HAVE_LAPACK "NO")
ENDIF(WITH_LAPACK)

#-----------------------------------------------------------------------------
# blas library
# DV Fix finding blas/atlas stuff
SET(DEF_BLAS "#define HAVE_BLAS 0")
SET(DEF_CBLAS "#define HAVE_CLBLAS 0")

# DV Fix these libraries!
# blas, f, 



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

# Fixing a problem on OSX for now: the standard CMake script finds the wrong OpenGL
# libraries. Hence they are hard coded here, however these require X11 to be installed
# and this is not checked here. We need to come up with a better fix here.
 
IF (APPLE)
  SET(GL_LIBRARY /usr/X11R6/lib/libGL.dylib /usr/X11R6/lib/libGLU.dylib "-framework AGL" CACHE INTERNAL "OpenGL Libs")
  SET(SCIRUN_INCLUDE_ADDITIONAL ${SCIRUN_INCLUDE_ADDITIONAL} ${GL_INCLUDE})
ENDIF(APPLE)

#----------------------------------------------------------------------------
# X11
OPTION(WITH_X11 "Build SCIRun with X11 support" ON)

IF(WITH_X11)
  FIND_PACKAGE( X11 )
  IF(X11_FOUND)
    SET(X_LIBRARY ${X11_LIBRARIES})
    SET(DEF_X11 "#define HAVE_X11 1")
  ENDIF(X11_FOUND)
ENDIF(WITH_X11)

MARK_AS_ADVANCED(WITH_X11)

#----------------------------------------------------------------------------
# sci_defs variables

# sci_defs/compile_defs.h
# DV Not needed eventually?

# Set Make for MAKE_COMMAND in compile_defs.h
# DV Fix Make sure this works for win
SET(MAKE ${CMAKE_BUILD_TOOL})

IF(WIN32 AND NOT CYGWIN)
  SET (PATH_TO_VC "c:/Program Files/Microsoft Visual Studio .NET 2003/Vc7" CACHE PATH "Path to Visual Studio")
#  SET (PATH_TO_PSDK "c:/Program Files/Microsoft Visual Studio .NET 2003/Vc7/PlatformSDK" CACHE PATH "Path to PSDK")
ENDIF(WIN32 AND NOT CYGWIN)

# sci_defs/environment_defs.h
SET(scirun_src_dir "${SCIRUN_CORE_SOURCE_DIR}" CACHE INTERNAL "Path to SCIRun's src directory")
# DV Technically, I don't have access to this variable once we fix building
# these projets separately!!!
SET(scirun_obj_dir ${SCIRUN_BINARY_DIR} CACHE INTERNAL "Path to SCIRun's binary directory")

SET(ITCL_WIDGETS ${SCIRUN_THIRDPARTY_PATH}/iwidgets/scripts CACHE INTERNAL "Path to iwidgets scripts")
SET(ITCL_LIB_DIR ${SCIRUN_THIRDPARTY_PATH}/lib)

# sci_defs/error_defs.h
SET(THROW_LEVEL 0 CACHE STRING "Set throw level N(0-3)")
SET(ASSERTION_LEVEL 3 CACHE STRING "Set assertion level N(0-3)")

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

CHECK_INCLUDE_FILE(uinstd.h HAVE_UNISTD)
IF(HAVE_UNISTD)
  SET(HAVE_UNISTD_H 1)
ELSE(HAVE_UNISTD)
  SET(HAVE_UNISTD_H 0)
ENDIF(HAVE_UNISTD)

CHECK_INCLUDE_FILE(sys/time.h HAVE_SYS_TIME)
IF(HAVE_SYS_TIME)
  SET(HAVE_SYS_TIME_H 1)
ELSE(HAVE_SYS_TIME)
  SET(HAVE_SYS_TIME_H 0)
ENDIF(HAVE_SYS_TIME)

# DV Fix this one (ext/algorithm)
CHECK_INCLUDE_FILE(ext/algorithm  HAVE_EXT_ALGORITHM)
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

# sci_defs/oogl_defs.h
#INCLUDE (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
#INCLUDE (${CMAKE_ROOT}/Modules/CheckLibraryExists.cmake)
#CHECK_INCLUDE_FILE(oogl/basicTexture.h HAVE_BASIC_TEXTURE)
#CHECK_INCLUDE_FILE(oogl/shadedPrim.h HAVE_SHADED_PRIM)
#CHECK_LIBRARY_EXISTS(oogl )
SET(DEF_OOGL "")

#SET(DEF_FTGL)

#SET(DEF_OGL_EXT_HDRS)

#SET(OGL_ERR_MACRO)

# sci_defs/petsc_defs.h
# petsc library
OPTION(WITH_PETSC "Build with petsc library" OFF)
MARK_AS_ADVANCED(WITH_PETSC)		   
IF(WITH_PETSC)		   
  FIND_LIBRARY(PETSC_LIBRARY 
    NAMES petsc petscvec petscmat petscdm petscsnes petscts petscksp
    PATHS /usr/local/lib
    /usr/lib
    /lib)

  FIND_PATH(PETSC_INCLUDE petsc.h
    /usr/local/include/
    /usr/include
    /include)

   IF(PETSC_LIBRARY)
     SET(HAVE_PETSC "YES")
     SET(DEF_PETSC "#define HAVE_PETSC 1")
   ELSE(PETSC_LIBRARY)
     SET(HAVE_PETSC "NO")
   ENDIF(PETSC_LIBRARY)
ELSE(WITH_PETSC)
   SET(HAVE_PETSC "NO")
ENDIF(WITH_PETSC)

# DV FIX check for unipetsc if petsc not found

#----------------------------------------------------------------------------
# Search for Mesquite

IF(MESQUITE_PATH)
  SET(MESQUITE_PATH ${MESQUITE_PATH} CACHE PATH "Path to MESQUITE" FORCE)
  MARK_AS_ADVANCED(CLEAR MESQUITE_PATH)
ENDIF(MESQUITE_PATH)

OPTION(WITH_MESQUITE "compile with mesquite" OFF)
MARK_AS_ADVANCED(WITH_MESQUITE)

IF(WITH_MESQUITE)
  MARK_AS_ADVANCED(CLEAR WITH_MESQUITE)
ENDIF(WITH_MESQUITE)

IF(WITH_MESQUITE OR MESQUITE_PATH)

  MESSAGE(STATUS "Configuring SCIRun with MESQUITE")

  SET(MESQUITE_LIBRARY MESQUITE_LIBRARY-NOTFOUND CACHE FILEPATH "Mesquite library" FORCE)
  SET(MESQUITE_INCLUDE1 MESQUITE_INCLUDE1-NOTFOUND CACHE FILEPATH "Mesquite include1" FORCE)
  SET(MESQUITE_INCLUDE2 MESQUITE_INCLUDE2-NOTFOUND CACHE FILEPATH "Mesquite include2" FORCE)

  IF(MESQUITE_PATH)
    FIND_LIBRARY(MESQUITE_LIBRARY 
      NAMES mesquite
      PATHS ${MESQUITE_PATH}
      ${MESQUITE_PATH}/lib
      ${SCIRUN_THIRDPARTY_PATH}/lib/
      /usr/local/lib
      /usr/lib
      /lib)

    FIND_PATH(MESQUITE_INCLUDE1 AddQualityMetric.hpp
      ${MESQUITE_PATH}/includeLinks
      ${MESQUITE_PATH}/../includeLinks
      ${SCIRUN_THIRDPARTY_PATH}/includeLinks
      /usr/local/include/
      /usr/include
      /include)

    FIND_PATH(MESQUITE_INCLUDE2 Mesquite.hpp
      ${MESQUITE_PATH}/include
      ${MESQUITE_PATH}/../include
      ${SCIRUN_THIRDPARTY_PATH}/include
      /usr/local/include/
      /usr/include
      /include)
  ELSE (MESQUITE_PATH)
    FIND_LIBRARY(MESQUITE_LIBRARY 
      NAMES mesquite
      PATHS ${SCIRUN_THIRDPARTY_PATH}/lib/
      /usr/local/lib
      /usr/lib
      /lib)

    FIND_PATH(MESQUITE_INCLUDE1 AddQualityMetric.hpp
      ${SCIRUN_THIRDPARTY_PATH}/includeLinks
      /usr/local/include/
      /usr/include
      /include)

    FIND_PATH(MESQUITE_INCLUDE2 Mesquite.hpp
      ${SCIRUN_THIRDPARTY_PATH}/include
      /usr/local/include/
      /usr/include
      /include)  
  ENDIF(MESQUITE_PATH)

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
ELSE(WITH_MESQUITE OR MESQUITE_PATH)
  SET(HAVE_MESQUITE "NO" CACHE INTERNAL "Do not edit" FORCE)
  SET(DEF_MESQUITE "" CACHE INTERNAL "Do not edit" FORCE)
ENDIF(WITH_MESQUITE OR MESQUITE_PATH)

#----------------------------------------------------------------------------
# Search for Verdict (http://tetgen.berlios.de)

IF(VERDICT_PATH)
  SET(VERDICT_PATH ${VERDICT_PATH} CACHE PATH "Path to verdict" FORCE)
  MARK_AS_ADVANCED(CLEAR VERDICT_PATH)
ENDIF(VERDICT_PATH)

OPTION(WITH_VERDICT "compile with verdict" OFF)
MARK_AS_ADVANCED(WITH_VERDICT)

IF(WITH_VERDICT)
  MARK_AS_ADVANCED(CLEAR WITH_VERDICT)
ENDIF(WITH_VERDICT)


IF(WITH_VERDICT OR VERDICT_PATH)

  MESSAGE(STATUS "Configuring SCIRun with VERDICT")

  SET(VERDICT_LIBRARY VERDICT_LIBRARY-NOTFOUND CACHE FILEPATH "Verdict library" FORCE)
  SET(VERDICT_INCLUDE VERDICT_INCLUDE-NOTFOUND CACHE FILEPATH "Verdict include" FORCE)

  IF(VERDICT_PATH)
    FIND_LIBRARY(VERDICT_LIBRARY 
      NAMES verdict110 
      PATHS ${VERDICT_PATH}
      ${VERDICT_PATH}/lib
      ${SCIRUN_THIRDPARTY_PATH}/lib
      /usr/local/lib
      /usr/lib
      /lib)

    FIND_PATH(VERDICT_INCLUDE verdict.h
      ${VERDICT_PATH}
      ${VERDICT_PATH}/include
      ${SCIRUN_THIRDPARTY_PATH}/include
      /usr/local/include/
      /usr/include
      /include)

  ELSE(VERDICT_PATH)
    FIND_LIBRARY(VERDICT_LIBRARY 
      NAMES verdict110 
      PATHS ${SCIRUN_THIRDPARTY_PATH}/lib
      /usr/local/lib
      /usr/lib
      /lib)

    FIND_PATH(VERDICT_INCLUDE verdict.h
      ${SCIRUN_THIRDPARTY_PATH}/include
      /usr/local/include/
      /usr/include
      /include)
  
  ENDIF(VERDICT_PATH)

  IF(VERDICT_LIBRARY AND VERDICT_INCLUDE)
    SET(HAVE_VERDICT "YES" CACHE INTERNAL "Do not edit" FORCE)
    SET(DEF_VERDICT "#define HAVE_VERDICT 1" CACHE INTERNAL "Do not edit" FORCE)
    SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} ${VERDICT_INCLUDE})
  ELSE(VERDICT_LIBRARY AND VERDICT_INCLUDE)
    MESSAGE(FATAL_ERROR "Cannot find verdict library.") 
  ENDIF(VERDICT_LIBRARY AND VERDICT_INCLUDE)

  MARK_AS_ADVANCED(VERDICT_LIBRARY)
  MARK_AS_ADVANCED(VERDICT_INCLUDE)
ELSE(WITH_VERDICT OR VERDICT_PATH)
  SET(HAVE_VERDICT "NO" CACHE INTERNAL "Do not edit" FORCE)
  SET(DEF_VERDICT "" CACHE INTERNAL "Do not edit" FORCE)
ENDIF(WITH_VERDICT OR VERDICT_PATH)

#----------------------------------------------------------------------------
# Search for Tetgen (http://tetgen.berlios.de)

IF(TETGEN_PATH)
  SET(TETGEN_PATH ${TETGEN_PATH} CACHE PATH "Path to tetgen" FORCE)
  MARK_AS_ADVANCED(CLEAR TETGEN_PATH)
ENDIF(TETGEN_PATH)

OPTION(WITH_TETGEN "compile with tetgen" OFF)
MARK_AS_ADVANCED(WITH_TETGEN)

IF(WITH_TETGEN)
  MARK_AS_ADVANCED(CLEAR WITH_TETGEN)
ENDIF(WITH_TETGEN)

IF(WITH_TETGEN OR TETGEN_PATH)

  MESSAGE(STATUS "Configuring SCIRun with TETGEN")

  SET(TETGEN_LIBRARY TETGEN_LIBRARY-NOTFOUND CACHE FILEPATH "Tetgen library" FORCE)
  SET(TETGEN_INCLUDE TETGEN_INCLUDE-NOTFOUND CACHE FILEPATH "Tetgen include" FORCE)
  
  IF(TETGEN_PATH)
    FIND_LIBRARY(TETGEN_LIBRARY 
      NAMES tet 
      PATHS ${TETGEN_PATH}
      ${TETGEN_PATH}/lib
      ${SCIRUN_THIRDPARTY_PATH}/lib
      /usr/local/lib
      /usr/lib
      /lib)

    FIND_PATH(TETGEN_INCLUDE tetgen.h
      ${TETGEN_PATH}
      ${TETGEN_PATH}/include
      ${SCIRUN_THIRDPARTY_PATH}/include
      /usr/local/include/
      /usr/include
      /include)
  
  ELSE(TETGEN_PATH)
    FIND_LIBRARY(TETGEN_LIBRARY 
      NAMES tet 
      PATHS ${SCIRUN_THIRDPARTY_PATH}/lib
      /usr/local/lib
      /usr/lib
      /lib)

    FIND_PATH(TETGEN_INCLUDE tetgen.h
      ${SCIRUN_THIRDPARTY_PATH}/include
      /usr/local/include/
      /usr/include
      /include)
  
  ENDIF(TETGEN_PATH)

  IF(TETGEN_LIBRARY AND TETGEN_INCLUDE)
    SET(HAVE_TETGEN "YES" CACHE INTERNAL "Do not edit" FORCE)
    SET(DEF_TETGEN "#define HAVE_TETGEN 1" CACHE INTERNAL "Do not edit" FORCE)
    SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} ${TETGEN_INCLUDE})
  ELSE(TETGEN_LIBRARY AND TETGEN_INCLUDE)
    MESSAGE(FATAL_ERROR "Cannot find tetgen library.") 
  ENDIF(TETGEN_LIBRARY AND TETGEN_INCLUDE)

  MARK_AS_ADVANCED(TETGEN_LIBRARY)
  MARK_AS_ADVANCED(TETGEN_INCLUDE)
ELSE(WITH_TETGEN OR TETGEN_PATH)
  SET(HAVE_TETGEN "NO" CACHE INTERNAL "Do not edit" FORCE)
  SET(DEF_TETGEN "" CACHE INTERNAL "Do not edit" FORCE)
ENDIF(WITH_TETGEN OR TETGEN_PATH)

#----------------------------------------------------------------------------
# Search for CBlas

IF(CBLAS_PATH)
  SET(CBLAS_PATH ${CBLAS_PATH} CACHE PATH "Path to tetgen" FORCE)
  MARK_AS_ADVANCED(CLEAR CBLAS_PATH)
ENDIF(CBLAS_PATH)

IF(APPLE)
  OPTION(WITH_CBLAS "compile with cblas" ON)
  MARK_AS_ADVANCED(CLEAR WITH_CBLAS)
ELSE(APPLE)
  OPTION(WITH_CBLAS "compile with cblas" OFF)
  MARK_AS_ADVANCED(WITH_CBLAS)
ENDIF(APPLE)

IF(WITH_CBLAS)
  MARK_AS_ADVANCED(CLEAR WITH_CBLAS)
ENDIF(WITH_CBLAS)

IF(WITH_CBLAS OR CBLAS_PATH)

  MESSAGE(STATUS "Configuring SCIRun with CBLAS")

  SET(CBLAS_LIBRARY CBLAS_LIBRARY-NOTFOUND CACHE FILEPATH "CBLAS library" FORCE)
  SET(CBLAS_INCLUDE CBLAS_INCLUDE-NOTFOUND CACHE FILEPATH "CBLAS include" FORCE)

  IF(CBLAS_PATH)

    FIND_LIBRARY(CBLAS_LIBRARY 
      NAMES cblas 
      PATHS ${CBLAS_PATH}
      ${CBLAS_PATH}/lib
      /usr/local/lib
      /usr/lib
      /lib)

    FIND_PATH(CBLAS_INCLUDE cblas.h
      ${CBLAS_PATH}
      ${CBLAS_PATH}/include
      /usr/local/include/
      /usr/include
      /include)

  ELSE(CBLAS_PATH)

    FIND_LIBRARY(CBLAS_LIBRARY 
      NAMES cblas 
      PATHS /usr/local/lib
      /usr/lib
      /lib)

    FIND_PATH(CBLAS_INCLUDE cblas.h
      /usr/local/include/
      /usr/include
      /include)
  ENDIF(CBLAS_PATH)


  IF(CBLAS_LIBRARY AND CBLAS_INCLUDE)
    SET(HAVE_CBLAS "YES" CACHE INTERNAL "Do not edit" FORCE)
    SET(DEF_CBLAS "#define HAVE_CBLAS 1" CACHE INTERNAL "Do not edit" FORCE)
    SET(SCIRUN_INCLUDE_THIRDPARTY ${SCIRUN_INCLUDE_THIRDPARTY} ${CBLAS_INCLUDE})
  ELSE(CBLAS_LIBRARY AND CBLAS_INCLUDE)
    MESSAGE(STATUS "Could not find CBLAS, configuring SCIRun without CBLAS")
  ENDIF(CBLAS_LIBRARY AND CBLAS_INCLUDE)

  IF (APPLE)
    SET(CBLAS_LIBRARY "-framework vecLib" CACHE STRING "CBLAS library" FORCE)
    SET(HAVE_CBLAS "YES" CACHE INTERNAL "Do not edit" FORCE)
    SET(DEF_CBLAS "#define HAVE_CBLAS 1" CACHE INTERNAL "Do not edit" FORCE)
  ENDIF (APPLE)

  MARK_AS_ADVANCED(CBLAS_LIBRARY)
  MARK_AS_ADVANCED(CBLAS_INCLUDE)
ELSE(WITH_CBLAS OR CBLAS_PATH)
  SET(HAVE_CBLAS "NO" CACHE INTERNAL "Do not edit" FORCE)
  SET(DEF_CBLAS "" CACHE INTERNAL "Do not edit" FORCE)
ENDIF(WITH_CBLAS OR CBLAS_PATH)


#----------------------------------------------------------------------------
# Insight Toolkit
IF(EXISTS ${SCIRUN_THIRDPARTY_PATH}/lib/InsightToolkit)
  SET(ITK_DIR ${SCIRUN_THIRDPARTY_PATH}/lib/InsightToolkit CACHE PATH "Path to IInsight Toolkit")
ENDIF(EXISTS ${SCIRUN_THIRDPARTY_PATH}/lib/InsightToolkit)

IF(ITK_DIR)
  OPTION(WITH_ITK "Build SCIRun with support for the Insight Toolkit." ON)
ELSE(ITK_DIR)
  OPTION(WITH_ITK "Build SCIRun with support for the Insight Toolkit." OFF)
ENDIF(ITK_DIR)

IF(WITH_ITK)
  FIND_PACKAGE( ITK )
  IF(ITK_FOUND)
    INCLUDE(${ITK_USE_FILE})
    SET(DEF_INSIGHT "#define HAVE_INSIGHT 1" CACHE INTERNAL "Have insight")
    SET(HAVE_INSIGHT "yes" CACHE INTERNAL "Linking against ITK")
    # add Insight to LOAD_PACKAGE if not all ready there
    STRING(REGEX MATCH "Insight" BUILDING_INSIGHT ${LOAD_PACKAGE})
    IF(NOT BUILDING_INSIGHT)
      SET(LOAD_PACKAGE "${LOAD_PACKAGE},Insight")
    ENDIF(NOT BUILDING_INSIGHT)
  ELSE(ITK_FOUND)
    MESSAGE(FATAL_ERROR "Cannot find ITK. Please set ITK_DIR.")
  ENDIF(ITK_FOUND)
ENDIF(WITH_ITK)

MARK_AS_ADVANCED(WITH_ITK)

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

# traverse the tree
SUBDIRS(include Core)

MARK_AS_ADVANCED(BLT_INCLUDE BLT_LIBRARY FREETYPE_INCLUDE FREETYPE_LIBRARY TCL_LIBRARY TCL_INCLUDE ITCL_INCLUDE ITCL_LIBRARY ITK_INCLUDE ITK_LIBRARY TK_INCLUDE TK_LIBRARY PNG_INCLUDE PNG_LIBRARY ZLIB_INCLUDE ZLIB_LIBRARY XML2_LIBRARY XML2_INCLUDE TEEM_LIBRARY TEEM_INCLUDE_DIR MPEG_LIBRARY MPEG_INCLUDE GLEW_LIBRARY GLEW_INCLUDE M_LIBRARY WITH_MPEG)
