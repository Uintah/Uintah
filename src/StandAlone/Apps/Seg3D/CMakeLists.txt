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

# This is the root ITK CMakeLists file.  
CMAKE_MINIMUM_REQUIRED(VERSION 2.0)

PROJECT(SEG3D)

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
IF(NOT BUILD_FROM_TOP_LEVEL)
  SET (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all libraries.")
  SET (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all executables.")
  #output directory for installing all include files.")
  MARK_AS_ADVANCED(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)
ENDIF(NOT BUILD_FROM_TOP_LEVEL)


# DV Find SCIRun Core stuff
SET(CMAKE_MODULE_PATH ${SEG3D_SOURCE_DIR})
FIND_PACKAGE( SCIRUN )
IF(SCIRUN_FOUND)
  INCLUDE(${SCIRUN_USE_FILE})
  SET(SCIRUN_DIR . CACHE PATH "Path to SCIRun Core")
ELSE(SCIRUN_FOUND)
  MESSAGE(FATAL_ERROR "Cannot build Seg3D without SCIRUN Core. Please set SCIRUN_DIR.")
ENDIF(SCIRUN_FOUND)

IF(NOT ITK_DIR)
  SET(ITK_DIR ${SCIRUN_ITK_DIR})
ENDIF(NOT ITK_DIR)

FIND_PACKAGE( ITK )
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR "Cannot build Seg3D without ITK. Please set ITK_DIR.")
ENDIF(ITK_FOUND)


IF(WIN32)
  IF(NOT BUILD_CORE_LIBS_SHARED)
    ADD_DEFINITIONS(-DBUILD_CORE_STATIC)
  ENDIF(NOT BUILD_CORE_LIBS_SHARED)
#  This defines M_PI for windows
  ADD_DEFINITIONS(-D_USE_MATH_DEFINES)
ENDIF(WIN32)


SET(SEG3D_SRCS 
  AutoviewTool.cc 
  BrushTool.cc 
  CLUTLevelsTool.cc
  CropTool.cc
  FloodfillTool.cc
  ITKConfidenceConnectedImageFilterTool.cc
  ITKThresholdSegmentationLevelSetImageFilterTool.cc
  KeyToolSelectorTool.cc
  LayerButton.cc
  LayerMergeTool.cc
  NrrdToITK.cc
  NrrdVolume.cc
  Painter.cc 
  PainterSignalTargets.cc 
  PanTool.cc 
  ProbeTool.cc 
  PointerToolSelectorTool.cc
  StatisticsTool.cc
  SessionReader.cc
  SessionWriter.cc
  SliceWindow.cc
  VolumeOps.cc
  VolumeSlice.cc
  ZoomTool.cc
  main.cc 
)




IF(APPLE)
  ADD_EXECUTABLE(Seg3D MACOSX_BUNDLE ${SEG3D_SRCS})
  SET_TARGET_PROPERTIES(Seg3D PROPERTIES LINK_FLAGS "-framework Carbon")
  ADD_CUSTOM_COMMAND(TARGET Seg3D
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND}
                     ARGS  -E copy_directory
                     ${SEG3D_SOURCE_DIR}/data
                     ${EXECUTABLE_OUTPUT_PATH}/Seg3D.app/Contents/MacOS/data)
ELSE(APPLE)
  ADD_EXECUTABLE(Seg3D WIN32 ${SEG3D_SRCS})
  #ADD_EXECUTABLE(Seg3D ${SEG3D_SRCS})
  ADD_CUSTOM_COMMAND(TARGET Seg3D
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND}
                     ARGS  -E copy_directory
                     ${SEG3D_SOURCE_DIR}/data
                     ${EXECUTABLE_OUTPUT_PATH}/data)
ENDIF(APPLE)


                
TARGET_LINK_LIBRARIES(Seg3D 
  ${XML_LIBRARY} 
  ${M_LIBRARY} 
  ${GL_LIBRARY} 
  ${GLU_LIBRARY} 
  ${TEEM_LIBRARY}
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
)

IF(ITK_FOUND)   
  TARGET_LINK_LIBRARIES(Seg3D 
    ITKCommon
    ITKBasicFilters
    ITKIO)
ENDIF(ITK_FOUND)

