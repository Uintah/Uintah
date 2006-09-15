#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
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

# CMakeLists.txt for Dataflow/Widgets

SET(Dataflow_Widgets_SRCS
  ArrowWidget.cc
  BaseWidget.cc
  BoxWidget.cc
  CriticalPointWidget.cc
  CrosshairWidget.cc
  FrameWidget.cc
  GaugeWidget.cc
  LightWidget.cc
  PathWidget.cc
  PointWidget.cc
  RingWidget.cc
  ViewWidget.cc
)

ADD_LIBRARY(Dataflow_Widgets ${Dataflow_Widgets_SRCS})

TARGET_LINK_LIBRARIES(Dataflow_Widgets
  Core_Datatypes
  Core_Exceptions
  Core_Geom
  Core_Geometry
  Core_GeomInterface
  Core_Containers
  Dataflow_GuiInterface
  Core_Thread
  Dataflow_Constraints
  Dataflow_Network
  ${M_LIBRARY}
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Dataflow_Widgets)
ENDIF(NOT BUILD_SHARED_LIBS)
