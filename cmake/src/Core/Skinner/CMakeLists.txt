# Sources of Core/Skinner classes
SET(Core_Skinner_SRCS
  Animation.cc
  Arithmetic.cc
  Arrow.cc
  Arc.cc
  Box.cc
  Collection.cc
  Color.cc
  ColorMap1D.cc
  Drawable.cc
  FocusRegion.cc
  Frame.cc
  Gradient.cc
  Graph2D.cc
  Grid.cc
  Layout.cc
  MenuManager.cc
  MenuList.cc
  Parent.cc
  RectRegion.cc
  Root.cc
  SceneGraph.cc
  Signals.cc
  Skinner.cc
  Text.cc
  TextEntry.cc
  Texture.cc
  Window.cc
  Variables.cc
  ViewSubRegion.cc
  XMLIO.cc    
)

ADD_LIBRARY(Core_Skinner ${Core_Skinner_SRCS})

TARGET_LINK_LIBRARIES(Core_Skinner 
  Core_Containers  
  Core_Datatypes 
  Core_Events 
  Core_Exceptions 
  Core_Geom  
  Core_Geometry  
  Core_Math  
  Core_Persistent  
  Core_Thread 
  Core_Util 
  Core_XMLUtil
  ${DL_LIBRARY} 
  ${THREAD_LIBRARY} 
  ${GL_LIBRARY}
  ${TEEM_LIBRARY} 
  ${XML2_LIBRARY}
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Skinner)
ENDIF(NOT BUILD_SHARED_LIBS)

