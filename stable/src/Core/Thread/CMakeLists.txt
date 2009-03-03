# Sources of Core/Thread classes
SET(Core_Thread_SRCS
  CleanupManager.cc 
  Guard.cc 
  MutexPool.cc 
  ParallelBase.cc 
  Runnable.cc 
  Thread.cc 
  ThreadError.cc 
  SimpleReducer.cc 
  ThreadLock.cc 
  ThreadGroup.cc 
  Thread_unix.cc 
  ThreadPool.cc 
  WorkQueue.cc 
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Thread)
ENDIF(NOT BUILD_SHARED_LIBS)

ADD_LIBRARY(Core_Thread ${Core_Thread_SRCS})

TARGET_LINK_LIBRARIES(Core_Thread 
  Core_Exceptions 
  ${THREAD_LIBRARY}  
  ${TRACEBACK_LIB} 
  ${SEMAPHORE_LIBRARY}
)

# Testing class instantiations
#ADD_EXECUTABLE(testThread ${CMAKE_SOURCE_DIR}/Testing/src/Core/Thread/TestThread.cc)

#TARGET_LINK_LIBRARIES(testThread Core_Thread)

#ADD_TEST(Core_Thread_Thread_Test ${EXECUTABLE_OUTPUT_PATH}/testThread)
