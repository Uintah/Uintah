# This is the root ITK CMakeLists file.  
CMAKE_MINIMUM_REQUIRED(VERSION 2.0)

PROJECT(Painter)

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
ELSE(SCIRUN_FOUND)
  MESSAGE(FATAL_ERROR "Cannot build Painter without SCIRUN Core. Please set SCIRUN_DIR.")
ENDIF(SCIRUN_FOUND)

FIND_PACKAGE( ITK )
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR "Cannot build Painter without ITK. Please set ITK_DIR.")
ENDIF(ITK_FOUND)

#-----------------------------------------------------------------------------
# Glew
FIND_LIBRARY(GLEW_LIBRARY GLEW glew32
             /usr/local/lib
             /usr/lib
             /lib)
FIND_PATH(GLEW_INCLUDE glew.h
           /usr/local/include
           /usr/include
           /include)

SET(PAINTER_SRCS 
  Painter.cc 
  PainterTools.cc 
  PainterBrushTool.cc 
  PainterSignalTargets.cc 
  main.cc 
)

ADD_EXECUTABLE(painter ${PAINTER_SRCS})

TARGET_LINK_LIBRARIES(painter 
  Core_Algorithms_Visualization 
  Core_Basis 
  Core_Bundle 
  Core_Containers 
  Core_Datatypes 
  Core_Events 
  Core_Exceptions 
  Core_Geom 
  Core_Geometry 
  Core_Init 
  Core_Math 
  Core_Persistent 
  Core_Skinner 
  Core_Thread 
  Core_Util 
  Core_Volume
  Core_Malloc
  ${LAPACK_LIBRARY} 
  ${XML_LIBRARY} 
  ${M_LIBRARY} 
  ${GL_LIBRARY} 
  ${TEEM_LIBRARY}
  ${GLEW_LIBRARY}
)

IF(${HAVE_INSIGHT})   
  TARGET_LINK_LIBRARY(painter 
    Packages_Insight_Core_Datatypes
    ${INSIGHT_LIBRARY}
  )
ENDIF(${HAVE_INSIGHT})




