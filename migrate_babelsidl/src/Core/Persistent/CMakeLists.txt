# Sources of Core/Persistent classes
SET(Core_Persistent_SRCS
  Persistent.cc
  PersistentSTL.cc
  Pstreams.cc
)

ADD_LIBRARY(Core_Persistent ${Core_Persistent_SRCS})
TARGET_LINK_LIBRARIES( Core_Persistent 
  Core_Containers 
  Core_Exceptions 
  Core_Thread 
  Core_Util
)

TARGET_LINK_LIBRARIES(Core_Persistent ${TEEM_LIBRARY} ${ZLIB_LIBRARY})

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Persistent)
ENDIF(NOT BUILD_SHARED_LIBS)

