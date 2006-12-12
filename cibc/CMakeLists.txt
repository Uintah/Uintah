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

PROJECT(SCIRUN)

# This is the root SCIRun CMakeLists file.
CMAKE_MINIMUM_REQUIRED(VERSION 2.2)

#-----------------------------------------------------------------------------
# Test for some required system information.
INCLUDE (CMakeBackwardCompatibilityC)

#-----------------------------------------------------------------------------
# SCIRUN version number.  Must reflect the value of SCIRUN_RCFILE_VERSION 
# in src/scirunrc
SET(SCIRUN_VERSION_MAJOR "3")
SET(SCIRUN_VERSION_MINOR "0")
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

#----------------------------------------------------------------------------
# Output directories.
SET (LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib CACHE INTERNAL "Single output directory for building all libraries.")
SET (EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all executables.")
#output directory for installing all include files.")
MARK_AS_ADVANCED(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)

SET(BUILD_FROM_TOP_LEVEL 1 CACHE INTERNAL "Building from top level")

OPTION(BUILD_SHARED_LIBS "Build shared libraries" ON)
MARK_AS_ADVANCED(BUILD_SHARED_LIBS)

# Automatically build core
# DV Need to fix building Testing
SUBDIRS(src)

# Option for building scirun executable, dataflow, packages
OPTION(BUILD_DATAFLOW "Build SCIRun executable, dataflow, and packages" ON)
IF(BUILD_DATAFLOW)
  SET(SCIRUN_DIR ${SCIRUN_BINARY_DIR}/src)
  SUBDIRS(src/Dataflow)
ENDIF(BUILD_DATAFLOW)
MARK_AS_ADVANCED(BUILD_DATAFLOW)

# Option for building Seg3D
OPTION(BUILD_SEG3D "Build Seg3D Application" OFF)
IF(BUILD_SEG3D)
#  SET(SCIRUN_DIR ${SCIRUN_BINARY_DIR}/src)
#  MARK_AS_ADVANCED(SCIRUN_DIR)
#  SUBDIRS(src/Dataflow)
ENDIF(BUILD_SEG3D)
MARK_AS_ADVANCED(BUILD_SEG3D)

MARK_AS_ADVANCED(CMAKE_INSTALL_PREFIX DART_TESTING_TIMEOUT)
