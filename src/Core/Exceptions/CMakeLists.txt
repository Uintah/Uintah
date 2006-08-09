# Sources of Core/Exceptions classes
SET(Core_Exceptions_SRCS
  ArrayIndexOutOfBounds.cc 
  AssertionFailed.cc 
  DimensionMismatch.cc 
  ErrnoException.cc 
  Exception.cc 
  FileNotFound.cc 
  InvalidState.cc 
  InternalError.cc
)

IF(WIN32 AND NOT CYGWIN)
  SET(Core_Exceptions_SRCS ${Core_Exceptions_SRCS} StackWalker.cc)
ENDIF(WIN32 AND NOT CYGWIN)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Exceptions)
ENDIF(NOT BUILD_SHARED_LIBS)

ADD_LIBRARY(Core_Exceptions ${Core_Exceptions_SRCS})

TARGET_LINK_LIBRARIES(Core_Exceptions ${TRACEBACK_LIB})
