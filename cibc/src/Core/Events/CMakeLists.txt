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

# Sources of Core/Events classes

SET(Core_Events_SRCS 
  BaseEvent.cc		
  EventManager.cc	
  OpenGLViewer.cc
  OSXEventSpawner.cc
  SceneGraphEvent.cc
  SelectionTargetEvent.cc
  X11EventSpawner.cc	
  Trail.cc	
  Tools/AutoviewTool.cc	
  Tools/Ball.cc		
  Tools/BallAux.cc		
  Tools/BallMath.cc		
  Tools/BaseTool.cc	
  Tools/FrameBufferPickTool.cc	
  Tools/ToolManager.cc	
  Tools/SelectionSetTool.cc	
  Tools/ViewRotateTool.cc	
  Tools/ViewScaleTool.cc	
  Tools/ViewTranslateTool.cc	
)

# DV Fix should do subdirectory tools instead of adding tools files above
#SUBDIRS(Tools)

ADD_LIBRARY(Core_Events ${Core_Events_SRCS})

TARGET_LINK_LIBRARIES( Core_Events 
  Core_Exceptions
  Core_Thread
  Core_Math
  Core_Geom
  Core_Util
  Core_Geometry
  Core_Containers 
  Core_Datatypes
  Core_Algorithms_Fields
  Core_Algorithms_Visualization
  ${THREAD_LIBRARY} 
  ${GL_LIBRARY} 
  ${GLEW_LIBRARY} 
  ${PNG_LIBRARY} 
)

IF(HAVE_MPEG)
  TARGET_LINK_LIBRARIES( Core_Events ${MPEG_LIBRARY} )
ENDIF(HAVE_MPEG)

IF(WIN32)
  TARGET_LINK_LIBRARIES( Core_Events Core_OS)
ENDIF(WIN32)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Events)
ENDIF(NOT BUILD_SHARED_LIBS)

