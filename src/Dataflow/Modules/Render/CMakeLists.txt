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

# CMakeLists.txt for Dataflow/Modules/Render

SET(Dataflow_Modules_Render_SRCS
  Ball.cc
  BallAux.cc
  BallMath.cc
  ShowAndEditCameraWidget.cc
  CreateAndEditCameraPath.cc
  ViewGraph.cc
  OpenGL.cc
  SynchronizeGeometry.cc
  ViewWindow.cc
  ViewSlices.cc
  ViewAndEditSlices.cc
  ViewAndEditSlicesTools.cc
  ViewAndEditSlicesBrushTool.cc
  ViewScene.cc
  glMath.cc
)

ADD_LIBRARY(Dataflow_Modules_Render ${Dataflow_Modules_Render_SRCS})

TARGET_LINK_LIBRARIES(Dataflow_Modules_Render
  Core_Algorithms_Fields
  Core_Basis
  Core_Bundle
  Core_Datatypes
  Core_Events
  Core_Exceptions
  Core_Geometry
  Core_Geom
  Core_GeomInterface
  Core_Containers
  Dataflow_GuiInterface
  Core_Persistent
  Core_Skinner
  Core_Thread
  Dataflow_TkExtensions
  Core_Util
  Core_Volume
  Dataflow_Comm
  Dataflow_Network
  Dataflow_Widgets
  ${TK_LIBRARY}
  ${GL_LIBRARY}
  ${MPEG_LIBRARY}
  ${MAGICK_LIBRARY}
  ${M_LIBRARY}
  ${TEEM_LIBRARY}
)

IF(HAVE_MPEG)
  TARGET_LINK_LIBRARIES( Dataflow_Modules_Render ${MPEG_LIBRARY} )
ENDIF(HAVE_MPEG)


IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Dataflow_Modules_Render)
ENDIF(NOT BUILD_SHARED_LIBS)
