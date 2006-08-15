# Sources of Core/Bundle classes
SET(Core_Bundle_SRCS
  Bundle.cc
)

ADD_LIBRARY(Core_Bundle ${Core_Bundle_SRCS})

TARGET_LINK_LIBRARIES( Core_Bundle 
  Core_Persitent
  Core_Exceptions
  Core_Containers 
  Core_Thread
  Core_Geometry
  Core_GuiInterface
  Core_Math
  Core_Util
  Core_Geom
  Core_Datatypes
  Core_Volume
  ${GL_LIBRARY}
  ${M_LIBRARY}
  ${BLAS_LIBRARY}
  ${F_LIBRARY}
  ${TEEM_LIBRARY}
  ${PNG_LIBRARY}
  ${Z_LIBRARY}
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Bundle)
ENDIF(NOT BUILD_SHARED_LIBS)