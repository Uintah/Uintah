# Sources of Core/Containers classes
SET(Core_Containers_SRCS
 ConsecutiveRangeSet.cc 
 Sort.cc 
 StringUtil.cc 
 TrivialAllocator.cc 
 templates.cc
)

ADD_LIBRARY(Core_Containers ${Core_Containers_SRCS})
TARGET_LINK_LIBRARIES(Core_Containers Core_Thread Core_Exceptions)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Containers)
ENDIF(NOT BUILD_SHARED_LIBS)
