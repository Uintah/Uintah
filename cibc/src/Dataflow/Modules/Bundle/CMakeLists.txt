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

# CMakeLists.txt for Dataflow/Modules/Bundle

SET(Dataflow_Modules_Bundle_SRCS
  BundleSetBundle.cc
  BundleSetField.cc
  BundleSetMatrix.cc
  BundleSetNrrd.cc
  BundleSetColorMap.cc
  BundleSetColorMap2.cc
  BundleSetPath.cc
  BundleSetString.cc
  BundleInfo.cc
  BundleGetBundle.cc
  BundleGetField.cc
  BundleGetMatrix.cc
  BundleGetNrrd.cc
  BundleGetColorMap.cc
  BundleGetColorMap2.cc
  BundleGetPath.cc
  BundleGetString.cc
  BundleMerge.cc
)

ADD_LIBRARY(Dataflow_Modules_Bundle ${Dataflow_Modules_Bundle_SRCS})

TARGET_LINK_LIBRARIES(Dataflow_Modules_Bundle
  Core_Datatypes
  Dataflow_Network
  Core_Persistent
  Core_Containers
  Core_Util
  Core_Exceptions
  Core_Thread
  Dataflow_GuiInterface
  Core_Geom
  Core_Datatypes
  Core_Geometry
  Core_GeomInterface
  Dataflow_TkExtensions
  Core_Volume
  Core_Bundle
  ${TK_LIBRARY}
  ${GL_LIBRARY}
  ${M_LIBRARY}
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Dataflow_Modules_Bundle)
ENDIF(NOT BUILD_SHARED_LIBS)
