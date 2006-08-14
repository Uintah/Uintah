# Sources of Core/Geometry classes
SET(Core_Geometry_SRCS
  BBox.cc
  CompGeom.cc
  Grid.cc
  IntVector.cc
  Plane.cc
  Point.cc
  Polygon.cc
  Quaternion.cc 
  Ray.cc
  Tensor.cc
  Transform.cc
  Vector.cc
)

ADD_LIBRARY(Core_Geometry ${Core_Geometry_SRCS})

TARGET_LINK_LIBRARIES( Core_Geometry 
  Core_Persitent
  Core_Containers 
  Core_Exceptions
  Core_Util
  Core_Math 
  ${TEEM_LIBRARY}
  ${PNG_LIBRARY}
  ${ZLIB_LIBRARY}
  ${M_LIBRARY}
)
#  ${DEFAULT_LIBRARY}


IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Geometry)
ENDIF(NOT BUILD_SHARED_LIBS)
