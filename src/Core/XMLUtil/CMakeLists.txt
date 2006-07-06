# Sources of Core/XMLUtil classes
SET(Core_XMLUtil_SRCS
  XMLUtil.cc
)

ADD_LIBRARY(Core_XMLUtil ${Core_XMLUtil_SRCS})

TARGET_LINK_LIBRARIES( Core_XMLUtil 
  Core_Containers 
  Core_Exceptions 
  ${XML2_LIBRARY}
  ${ZLIB_LIBRARY}
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_XMLUtil)
ENDIF(NOT BUILD_SHARED_LIBS)
