#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
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

PROJECT(SCIRUN)

#-----------------------------------------------------------------------------
# Test for some required system information.
INCLUDE (CMakeBackwardCompatibilityC)

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
SET (LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib CACHE INTERNAL "Single output directory for building all libraries.")
SET (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all executables.")
#output directory for installing all include files.")
MARK_AS_ADVANCED(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)
SET(ENV{SCIRUN_OBJDIR} "${PROJECT_BINARY_DIR}")


# DV Find SCIRun Core stuff
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
FIND_PACKAGE( SCIRUN )
IF(SCIRUN_FOUND)
  INCLUDE(${SCIRUN_USE_FILE})
  SET(SCIRUN_DIR . CACHE PATH "Path to SCIRun Core")
ELSE(SCIRUN_FOUND)
  MESSAGE(FATAL_ERROR "Cannot build SCIRun Dataflow without SCIRUN Core. Please set SCIRUN_DIR.")
ENDIF(SCIRUN_FOUND)

# Create on-the-fly-libs stuff
CONFIGURE_FILE (
  "${SCIRUN_SOURCE_DIR}/src/Makefile.in"
  "${PROJECT_BINARY_DIR}/on-the-fly-libs/Makefile"
)

# TCL include directories
SET(TCL_SRC_PATH ${SCIRUN_THIRDPARTY_PATH}/src/tcl CACHE PATH "Path to tcl source")
SET(TCL_INCLUDE1 ${TCL_SRC_PATH}/generic)
IF(CMAKE_SYSTEM_NAME MATCHES Darwin)
  SET(TCL_INCLUDE2 ${TCL_SRC_PATH}/mac)
ELSE(CMAKE_SYSTEM_NAME MATCHES Darwin)
  SET(TCL_INCLUDE2 ${TCL_SRC_PATH}/unix)
ENDIF(CMAKE_SYSTEM_NAME MATCHES Darwin)

FIND_PATH(TCL_INCLUDE tcl.h
           ${SCIRUN_THIRDPARTY_PATH}/include
           /usr/local/include
           /usr/include
           /include)

INCLUDE_DIRECTORIES(${TCL_INCLUDE1} ${TCL_INCLUDE2})

# TK include directories
SET(TK_SRC_PATH ${SCIRUN_THIRDPARTY_PATH}/src/tk CACHE PATH "Path to tk source")
SET(TK_INCLUDE1 ${TK_SRC_PATH}/generic)
IF(CMAKE_SYSTEM_NAME MATCHES Darwin)
  SET(TK_INCLUDE2 ${TK_SRC_PATH}/mac)
ELSE(CMAKE_SYSTEM_NAME MATCHES Darwin)
  SET(TK_INCLUDE2 ${TK_SRC_PATH}/unix)
ENDIF(CMAKE_SYSTEM_NAME MATCHES Darwin)

FIND_PATH(TK_INCLUDE tk.h
           ${SCIRUN_THIRDPARTY_PATH}/include
           /usr/local/include
           /usr/include
           /include)

INCLUDE_DIRECTORIES(${TK_INCLUDE1} ${TK_INCLUDE2})


SUBDIRS(Comm Constraints GuiInterface Modules Network TkExtensions TCLThread
Widgets)

# createTclIndex and tclIndex files
# DV Fix for each packages, create their tclIndex file
SET(CREATETCLINDEX ${CMAKE_BINARY_DIR}/createTclIndex CACHE INTERNAL "createTclIndex script file")
SET(TCLSH ${SCIRUN_THIRDPARTY_PATH}/bin/tclsh${TCL_VERSION})
SET(TCL_INIT_DIR ${SCIRUN_THIRDPARTY_PATH}/lib/tcl/)
SET(ITCL_INIT_DIR ${SCIRUN_THIRDPARTY_PATH}/lib/itcl/)
MESSAGE(STATUS "Creating createTclIndex")
CONFIGURE_FILE (
  "${SCIRUN_SOURCE_DIR}/src/scripts/cmake_createTclIndex.in"
  "${CREATETCLINDEX}"
)

# Change permissions on createTclIndex file and generate
# tclIndex files for Core/GUI and Dataflow/GUI
ADD_CUSTOM_TARGET(tclIndex ALL
  chmod a+x ${CREATETCLINDEX}
  COMMAND ${CREATETCLINDEX} ${SCIRUN_SOURCE_DIR}/src/Core/GUI 
  COMMAND ${CREATETCLINDEX} ${SCIRUN_SOURCE_DIR}/src/Dataflow/GUI
)

# scirun executable
ADD_EXECUTABLE(scirun
${SCIRUN_SOURCE_DIR}/src/main/main.cc)

TARGET_LINK_LIBRARIES(scirun 
  Core_Basis         
  Core_Comm          
  Core_Containers    
  Core_Exceptions    
  Core_Geom          
  Dataflow_GuiInterface  
  Core_ICom          
  Core_Init         
  Core_Malloc 
  Core_Services      
  Core_SystemCall    
  Core_Thread        
  Dataflow_TkExtensions  
  Core_Util          
  Core_Volume        
  Core_XMLUtil       
  Dataflow_Network   
  Dataflow_TCLThread 
  ${GLEW_LIBRARY}
)



IF(CMAKE_SYSTEM_NAME MATCHES Darwin)
    TARGET_LINK_LIBRARIES(scirun Core_Datatypes Core_ImportExport Core_Persistent)
ENDIF(CMAKE_SYSTEM_NAME MATCHES Darwin)


IF(WIN32)
  IF(NOT BUILD_SHARED_LIBS)
    ADD_DEFINITIONS(-DBUILD_STATIC)
  ENDIF(NOT BUILD_SHARED_LIBS)
ENDIF(WIN32)




