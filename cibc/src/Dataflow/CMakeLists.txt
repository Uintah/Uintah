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


# DV Find SCIRun Core stuff
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
FIND_PACKAGE( SCIRUN )
IF(SCIRUN_FOUND)
  INCLUDE(${SCIRUN_USE_FILE})
  SET(SCIRUN_DIR . CACHE PATH "Path to SCIRun Core")
ELSE(SCIRUN_FOUND)
  MESSAGE(FATAL_ERROR "Cannot build SCIRun Dataflow without SCIRUN Core. Please set SCIRUN_DIR.")
ENDIF(SCIRUN_FOUND)

IF(WITH_ITK)
  IF(NOT ITK_DIR)
    SET(ITK_DIR ${SCIRUN_ITK_DIR})
  ENDIF(NOT ITK_DIR)

  FIND_PACKAGE( ITK )
  IF(ITK_FOUND)
    INCLUDE(${ITK_USE_FILE})
  ELSE(ITK_FOUND)
    MESSAGE(FATAL_ERROR "SCIRun Core build with the Insight Toolkit. Please set ITK_DIR.")
  ENDIF(ITK_FOUND)
ENDIF(WITH_ITK)

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
           /include
           NO_DEFAULT_PATH)
MARK_AS_ADVANCED(TCL_SRC_PATH TCL_INCLUDE1 TCL_INCLUDE2 TCL_INCLUDE)
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
           /include
           NO_DEFAULT_PATH)
MARK_AS_ADVANCED(TK_INCLUDE TK_SRC_PATH TK_INCLUDE1 TK_INCLUDE2)

INCLUDE_DIRECTORIES(${TK_INCLUDE1} ${TK_INCLUDE2})

# Create on-the-fly-libs stuff
# DV Fix - these shouldn't necesarily point to thirdparty
SET(INC_SCI_THIRDPARTY_H "-I${SCIRUN_THIRDPARTY_PATH}/include")
SET(INC_TCL_H "-I${SCIRUN_THIRDPARTY_PATH}/include")
IF(CMAKE_SYSTEM_NAME MATCHES Darwin)
  SET(INC_TK_H "-I${SCIRUN_THIRDPARTY_PATH}/src/tk/generic -I${SCIRUN_THIRDPARTY_PATH}/src/tk/mac")
ELSE(CMAKE_SYSTEM_NAME MATCHES Darwin)
  SET(INC_TK_H "-I${SCIRUN_THIRDPARTY_PATH}/src/tk/generic -I${SCIRUN_THIRDPARTY_PATH}/src/tk/unix")
ENDIF(CMAKE_SYSTEM_NAME MATCHES Darwin)
SET(INC_ITCL_H "-I${SCIRUN_THIRDPARTY_PATH}/include")
SET(INC_BLT_H "-I${SCIRUN_THIRDPARTY_PATH}/include")
SET(INC_XML2_H "-I${SCIRUN_THIRDPARTY_PATH}/include/libxml2")
IF(WITH_TETGEN)
  SET(INC_TETGEN_H "-I${TEGEN_INCLUDE}")
ENDIF(WITH_TETGEN)

SET(scirun_obj_dataflow_dir ${PROJECT_BINARY_DIR})
#SET(scirun_core_obj_dir ${scirun_obj_dir})

SET(IS_OSX "no")
SET(IS_WIN "no")
SET(IS_AIX "no")

SET(CFLAGS "-Wall -g ")

IF(CMAKE_SYSTEM_NAME MATCHES Darwin)
  SET(SO_OR_A_FILE "dylib" CACHE STRING "library extension")
  SET(IS_OSX "yes")
  SET(SOFLAGS "${CFLAGS} -shared -L${LIBRARY_OUTPUT_PATH}")
ENDIF(CMAKE_SYSTEM_NAME MATCHES Darwin)
IF(CMAKE_SYSTEM_NAME MATCHES AIX)
  SET(SO_OR_A_FILE "a" CACHE STRING "library extension")
  SET(IS_AIX "yes")
  SET(SOFLAGS "${CFLAGS} -shared -L${LIBRARY_OUTPUT_PATH}")
ENDIF(CMAKE_SYSTEM_NAME MATCHES AIX)
IF(WIN32 AND NOT CYGWIN)
  SET(SO_OR_A_FILE "dll" CACHE STRING "library extension")
  SET(IS_WIN "yes")
  SET(SOFLAGS "${CFLAGS} -shared -LIBPATH:${LIBRARY_OUTPUT_PATH}")
ELSE(WIN32 AND NOT CYGWIN)
  SET(SO_OR_A_FILE "so" CACHE STRING "library extension")
  SET(SOFLAGS "${CFLAGS} -shared -L${LIBRARY_OUTPUT_PATH}")
ENDIF(WIN32 AND NOT CYGWIN)

MARK_AS_ADVANCED(SO_OR_A_FILE)

IF(WIN32)
  CONFIGURE_FILE (
    "${SCIRUN_SOURCE_DIR}/src/on-the-fly-libs/cmake_win32make.in"
    "${PROJECT_BINARY_DIR}/on-the-fly-libs/win32make.bat"
  )
ELSE(WIN32)
  CONFIGURE_FILE (
    "${SCIRUN_SOURCE_DIR}/src/on-the-fly-libs/cmake_Makefile.in"
    "${PROJECT_BINARY_DIR}/on-the-fly-libs/Makefile"
  )
ENDIF(WIN32)

SET(BUILD_SHARED_LIBS ON)


# Add correct flags for win32
IF(WIN32)
  IF(NOT BUILD_CORE_LIBS_SHARED)
    ADD_DEFINITIONS(-DBUILD_CORE_STATIC)
  ENDIF(NOT BUILD_CORE_LIBS_SHARED)
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

SUBDIRS(Comm Constraints GuiInterface Modules Network TkExtensions TCLThread
Widgets)

# createTclIndex and tclIndex files
# DV Fix for each packages, create their tclIndex file
SET(CREATETCLINDEX ${CMAKE_BINARY_DIR}/createTclIndex CACHE INTERNAL "createTclIndex script file")
SET(TCLSH ${SCIRUN_THIRDPARTY_PATH}/bin/tclsh${TCL_VERSION})
SET(TCL_INIT_DIR ${SCIRUN_THIRDPARTY_PATH}/lib/tcl/)
SET(ITCL_INIT_DIR ${SCIRUN_THIRDPARTY_PATH}/lib/itcl/)
MESSAGE(STATUS "Creating createTclIndex")

IF(WIN32) 
  # On Windows, we would just make a batch file to do the same thing as the createTCLIndex sh script,
  # but the << operator is not accepted.  So we must create a file per tcl index
  set (CREATE_TCL_PACKAGE ${SCIRUN_SOURCE_DIR}/src/Core/GUI)
  CONFIGURE_FILE (
    "${SCIRUN_SOURCE_DIR}/src/scripts/cmake_win32createTclIndex.in"
    "${PROJECT_BINARY_DIR}/createCoreTclIndex.tcl"
  )
  set (CREATE_TCL_PACKAGE ${SCIRUN_SOURCE_DIR}/src/Dataflow/GUI)
  CONFIGURE_FILE (
    "${SCIRUN_SOURCE_DIR}/src/scripts/cmake_win32createTclIndex.in"
    "${PROJECT_BINARY_DIR}/createDataflowTclIndex.tcl"
  )
  
  ADD_CUSTOM_TARGET(create_tclIndex_SCIRun ALL
    COMMAND ${TCLSH} createCoreTclIndex.tcl
    COMMAND ${TCLSH} createDataflowTclIndex.tcl
  )  
ELSE(WIN32)
  CONFIGURE_FILE (
    "${SCIRUN_SOURCE_DIR}/src/scripts/cmake_createTclIndex.in"
    "${CREATETCLINDEX}"
  )
  # Change permissions on createTclIndex file and generate
  # tclIndex files for Core/GUI and Dataflow/GUI
  ADD_CUSTOM_TARGET(create_tclIndex_SCIRun ALL
    chmod a+x ${CREATETCLINDEX}
    COMMAND ${CREATETCLINDEX} ${SCIRUN_SOURCE_DIR}/src/Core/GUI 
    COMMAND ${CREATETCLINDEX} ${SCIRUN_SOURCE_DIR}/src/Dataflow/GUI
  )
ENDIF(WIN32)
# Packages enabled when building core

# Replace comma's from LoadPackage to be semi-colons to make a list
STRING(REGEX REPLACE "[,]" ";" PACKAGE_LIST ${LOAD_PACKAGE})
SET(INCLUDED_PACKAGE_INCLUDES FALSE)

FOREACH( pkg ${PACKAGE_LIST})
  # only do anything for packages other than SCIRun
  IF(NOT ${pkg} MATCHES "SCIRun")
    # add SCIRun/src/Packages to the include line once
    IF(NOT ${INCLUDED_PACKAGE_INCLUDES})
      INCLUDE_DIRECTORIES(${SCIRUN_SOURCE_DIR}/src/Packages)
      SET(INCLUDED_PACKAGE_INCLUDES TRUE)
    ENDIF(NOT ${INCLUDED_PACKAGE_INCLUDES})

    # Add the package subdirectory in CMake
    SUBDIRS(${SCIRUN_SOURCE_DIR}/src/Packages/${pkg})

    IF(WIN32) 
      # On Windows, we would just make a batch file to do the same thing as the createTCLIndex sh script,
      # but the << operator is not accepted.  So we must create a file per tcl index
      set (CREATE_TCL_PACKAGE ${SCIRUN_SOURCE_DIR}/src/Packages/${pkg}/Dataflow/GUI)
      CONFIGURE_FILE (
        "${SCIRUN_SOURCE_DIR}/src/scripts/cmake_win32createTclIndex.in"
        "${PROJECT_BINARY_DIR}/create${pkg}TclIndex.tcl"
      )
      set (CREATE_TCL_PACKAGE ${SCIRUN_SOURCE_DIR}/src/Dataflow/GUI)
      
      ADD_CUSTOM_TARGET(create_tclIndex_${pkg} ALL
        COMMAND ${TCLSH} create${pkg}TclIndex.tcl
      )  
    ELSE(WIN32)

      # Create the tclIndex file
      ADD_CUSTOM_TARGET(create_tclIndex_${pkg} ALL
        chmod a+x ${CREATETCLINDEX}
        COMMAND ${CREATETCLINDEX} ${SCIRUN_SOURCE_DIR}/src/Packages/${pkg}/Dataflow/GUI 
      )
    ENDIF(WIN32)
  ENDIF(NOT ${pkg} MATCHES "SCIRun")
ENDFOREACH( pkg )

# scirun executable
ADD_EXECUTABLE(scirun
${SCIRUN_SOURCE_DIR}/src/main/main.cc)

SET(SCIRUN_LIBS
  Core_Basis         
  Core_Comm          
  Core_Containers    
  Core_Exceptions    
  Core_Geom          
  Dataflow_GuiInterface  
  Core_ICom          
  Core_Init         
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

IF(NOT WIN32)
# don't add malloc to windows version
  SET(SCIRUN_LIBS ${SCIRUN_LIBS} Core_Malloc)
ENDIF(NOT WIN32)

TARGET_LINK_LIBRARIES(scirun ${SCIRUN_LIBS}
)



IF(CMAKE_SYSTEM_NAME MATCHES Darwin)
    TARGET_LINK_LIBRARIES(scirun Core_Datatypes Core_ImportExport Core_Persistent)
ENDIF(CMAKE_SYSTEM_NAME MATCHES Darwin)

