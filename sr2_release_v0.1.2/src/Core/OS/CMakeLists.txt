# Sources of OS classes
SET(Core_OS_SRCS
  Dir.cc
  sock.cc
  ProcessInfo.cc
)

ADD_LIBRARY(Core_OS ${Core_OS_SRCS})

TARGET_LINK_LIBRARIES( Core_OS Core_Exceptions ${SOCKET_LIBRARY})

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_OS)
ENDIF(NOT BUILD_SHARED_LIBS)

