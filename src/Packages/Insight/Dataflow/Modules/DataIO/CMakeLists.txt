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

# CMakeLists.txt for Packages/Insight/Dataflow/Modules/DataIO

SET(Packages_Insight_Dataflow_Modules_DataIO_SRCS
  ImageFileWriter.cc
  ImageReaderFloat2D.cc
  ImageReaderFloat3D.cc
  ColorImageReaderFloat2D.cc
  ColorImageReaderFloat3D.cc
  ChooseImage.cc
  SliceReader.cc
)

ADD_LIBRARY(Packages_Insight_Dataflow_Modules_DataIO ${Packages_Insight_Dataflow_Modules_DataIO_SRCS})

TARGET_LINK_LIBRARIES(Packages_Insight_Dataflow_Modules_DataIO
  Core_Datatypes
  Dataflow_Network
  Packages_Insight_Dataflow_Ports
  Core_Algorithms_DataIO
  Core_Persistent
  Core_Containers
  Core_Util
  Core_Exceptions
  Core_Thread
  Dataflow_GuiInterface
  Core_Geom
  Core_Geometry
  Dataflow_TkExtensions
  Core_GeomInterface
  ITKCommon
  ITKIO
  ${GDCM_LIBRARY}
  ${TK_LIBRARY}
  ${GL_LIBRARY}
  ${M_LIBRARY}
  ${TEEM_LIBRARY}
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Packages_Insight_Dataflow_Modules_DataIO)
ENDIF(NOT BUILD_SHARED_LIBS)
