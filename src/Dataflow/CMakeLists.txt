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
SET (LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all libraries.")
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

ADD_EXECUTABLE(scirun
/home/darbyb/projects/Experimental/SCIRun/src/main/main.cc)

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




