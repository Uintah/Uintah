# Sources of Core/Algorithms/Util classes
SET(Core_Algorithms_Util_SRCS
  FieldInformation.cc
)

ADD_LIBRARY(Core_Algorithms_Util ${Core_Algorithms_Util_SRCS})

TARGET_LINK_LIBRARIES(Core_Algorithms_Util 
  Core_Containers 
  Core_Exceptions 
  Core_Thread 
  Core_Datatypes 
  Core_Basis 
  Core_Geom 
  Core_Util
  ${DL_LIBRARY} 
  ${THREAD_LIBRARY}
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Algorithms_Util)
ENDIF(NOT BUILD_SHARED_LIBS)

