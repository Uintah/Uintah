# Sources of Core/Volume classes
SET(Core_Volume_SRCS
  CM2Shader.cc
  CM2Widget.cc
  ColorMap2.cc
  SliceRenderer.cc
  Texture.cc
  TextureBrick.cc
  TextureRenderer.cc
  VideoCardInfo.c
  VolShader.cc
  VolumeRenderer.cc
)

ADD_LIBRARY(Core_Volume ${Core_Volume_SRCS})

TARGET_LINK_LIBRARIES( Core_Volume 
  Core_Containers 
  Core_Datatypes
  Core_Exceptions
  Core_Geom
  Core_Geometry
  Core_Persistent
  Core_Thread
  Core_Util
  ${GL_LIBRARY}
  ${M_LIBRARY}
  ${TEEM_LIBRARY}
)

IF(WIN32 AND NOT CYGWIN)
  TARGET_LINK_LIBRARIES( Core_Volume Core_Math )
ENDIF(WIN32 AND NOT CYGWIN)

IF (${CMAKE_SYSTEM_NAME} MATCHES Darwin)
  #TARGET_LINK_LIBRARIES( Core_Volume "framework AGL" )
  TARGET_LINK_LIBRARIES( Core_Volume ${OPENGL_glu_LIBRARY})
ENDIF (${CMAKE_SYSTEM_NAME} MATCHES Darwin)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Volume)
ENDIF(NOT BUILD_SHARED_LIBS)
