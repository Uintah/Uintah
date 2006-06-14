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

IF(WIN32)
  SET(Core_Exceptions_SRCS ${Core_Exceptions_SRCS} StackWalker.cc)
ENDIF(WIN32)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Exceptions)
ENDIF(NOT BUILD_SHARED_LIBS)

ADD_LIBRARY(Core_Exceptions ${Core_Exceptions_SRCS})

TARGET_LINK_LIBRARIES(Core_Exceptions ${TRACEBACK_LIB})

# Testing class instantiations
ADD_EXECUTABLE(testFileNotFound ${CMAKE_SOURCE_DIR}/Testing/src/Core/Exceptions/testFileNotFound.cc)

TARGET_LINK_LIBRARIES(testFileNotFound Core_Exceptions)

ADD_TEST(Core_Exceptions_FileNotFound_Test ${EXECUTABLE_OUTPUT_PATH}/testFileNotFound)