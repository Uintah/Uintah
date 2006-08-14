# Sources of Core/Events classes
SET(Core_Events_SRCS 
  BaseEvent.cc		
  EventManager.cc	
  OpenGLViewer.cc	
  SceneGraphEvent.cc	
  X11EventSpawner.cc	
)

SUBDIRS(Tools)

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

