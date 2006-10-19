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

# CMakeLists.txt for Dataflow/TCLThread

SET(Dataflow_TCLThread_SRCS
  TCLThread.cc
)

ADD_LIBRARY(Dataflow_TCLThread ${Dataflow_TCLThread_SRCS})

TARGET_LINK_LIBRARIES(Dataflow_TCLThread
  Core_Thread
  Dataflow_GuiInterface
  Dataflow_TkExtensions
  Core_Comm
  Core_Containers
  Core_Exceptions
  Dataflow_Network
  Core_Util
  Core_Geom
  Dataflow_TkExtensions
  ${TK_LIBRARY}
  ${TCL_LIBRARY}
  ${BLT_LIBRARY}
  ${ITCL_LIBRARY}
  ${ITK_LIBRARY}
  ${X_LIBRARY}
  ${GL_LIBRARY}
)

IF(BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Dataflow_TCLThread)
ENDIF(BUILD_SHARED_LIBS)
