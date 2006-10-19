#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#

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

# DV Fix for all cases
IF(${CMAKE_SYSTEM_NAME} MATCHES IRIX)
  SET(Core_Thread_SRCS ${Core_Thread_SRCS} Thread_irix.cc Time_irix.cc)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES IRIX)

IF(CMAKE_USE_PTHREADS)
  SET(Core_Thread_SRCS ${Core_Thread_SRCS} Thread_pthreads.cc Time_unix.cc)
ENDIF(CMAKE_USE_PTHREADS)

IF(WIN32)
  SET(Core_Thread_SRCS ${Core_Thread_SRCS} Thread_win32.cc Time_win32.cc)
ENDIF(WIN32)

IF(BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Thread)
ENDIF(BUILD_SHARED_LIBS)

ADD_LIBRARY(Core_Thread ${Core_Thread_SRCS})

TARGET_LINK_LIBRARIES(Core_Thread 
  Core_Exceptions 
  ${THREAD_LIBRARY}  
  ${TRACEBACK_LIB} 
  ${SEMAPHORE_LIBRARY}
)