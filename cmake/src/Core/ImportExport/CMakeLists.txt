# Sources of Core/ImportExport classes
# NOTE: source files are all in sub directories ColorMap, Matrix, Field,
# and Nrrd
SET(Core_ImportExport_SRCS
  ColorMap/ColorMapIEPlugin.cc
  ColorMap/ColorMapExecConverter.cc
  Matrix/MatrixExecConverter.cc 
  Matrix/MatrixIEPlugin.cc
  Field/FieldExecConverter.cc 
  Field/FieldIEPlugin.cc 
  Field/TextPointCloudString_plugin.cc
  Nrrd/NrrdExecConverter.cc 
  Nrrd/NrrdIEPlugin.cc
)

ADD_LIBRARY(Core_ImportExport ${Core_ImportExport_SRCS})

TARGET_LINK_LIBRARIES(Core_ImportExport 
  Core_Persistent 
  Core_Exceptions 
  Core_Containers 
  Core_Thread 
  Core_Geometry 
  Core_Geom 
  Core_Math 
  Core_Util 
  Core_Datatypes
  ${THREAD_LIBRARY} 
  ${Z_LIBRARY} 
  ${M_LIBRARY}
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_ImportExport)
ENDIF(NOT BUILD_SHARED_LIBS)
