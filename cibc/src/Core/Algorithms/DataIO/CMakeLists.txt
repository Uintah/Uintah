# Sources of Core/Algorithms/DataIO classes
SET(Core_Algorithms_DataIO_SRCS
  DataIOAlgo.cc
)

IF(HAVE_INSIGHT)
  SET(Core_Algorithms_DataIO_SRCS ${Core_Algorithms_DatIO_SRCS}
      AnalyzeImage.cc AnalyzeReader.cc AnalyzeSliceImageIO.cc)
ENDIF(HAVE_INSIGHT)

ADD_LIBRARY(Core_Algorithms_DataIO ${Core_Algorithms_DataIO_SRCS})

TARGET_LINK_LIBRARIES(Core_Algorithms_DataIO 
  Core_Datatypes 
  Core_Containers 
  Core_Thread 
  Core_Exceptions 
  Core_Geometry 
  Core_Util 
  Core_Math 
  Core_Persistent 
  Core_Geom 
  Core_Bundle 
  Core_ImportExport 
  Core_Volume
  ${TEEM_LIBRARY}
  ${Z_LIBRARY}
  ${M_LIBRARY}
)

IF(HAVE_INSIGHT)
  TARGET_LINK_LIBRARIES(Core_Algorithms_DataIO
    ${INSIGHT_LIBRARY}
    ${GDCM_LIBRARY}
    ${GL_LIBRARY})
ENDIF(HAVE_INSIGHT)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Algorithms_DataIO)
ENDIF(NOT BUILD_SHARED_LIBS)

