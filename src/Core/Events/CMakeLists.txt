# Sources of Core/Events classes
SET(Core_Events_SRCS 
  BaseEvent.cc		
  EventManager.cc	
  OpenGLViewer.cc	
  SceneGraphEvent.cc
  SelectionTargetEvent.cc
  X11EventSpawner.cc	
  Tools/AutoviewTool.cc	
  Tools/Ball.cc		
  Tools/BallAux.cc		
  Tools/BallMath.cc		
  Tools/BaseTool.cc	
  Tools/FrameBufferPickTool.cc	
  Tools/ToolManager.cc	
  Tools/ViewRotateTool.cc	
  Tools/ViewScaleTool.cc	
  Tools/ViewTranslateTool.cc	
  Tools/ViewToolInterface.cc
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
  ${THREAD_LIBRARY} 
  ${MPEG_LIBRARY} 
  ${GL_LIBRARY} 
  ${PNG_LIBRARY} 
)

IF(WIN32)
  TARGET_LINK_LIBRARIES( Core_Events Core_OS)
ENDIF(WIN32)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Events)
ENDIF(NOT BUILD_SHARED_LIBS)

