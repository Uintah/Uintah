# Sources of Core/Algorithms/Visualization classes
SET(Core_Algorithms_Visualization_SRCS
  EdgeMC.cc             
  HexMC.cc		
  MarchingCubes.cc	
  mcube2.cc		
  Noise.cc
  NrrdTextureBuilderAlgo.cc	
  PrismMC.cc		
  QuadMC.cc		
  RenderField.cc	
  Sage.cc		
  TetMC.cc		
  TextureBuilderAlgo.cc	
  TriMC.cc		
  UHexMC.cc		
)

ADD_LIBRARY(Core_Algorithms_Visualization ${Core_Algorithms_Visualization_SRCS})

TARGET_LINK_LIBRARIES(Core_Algorithms_Visualization 
  Core_Basis
  Core_Containers 
  Core_Datatypes
  Core_Exceptions
  Core_Geom
  Core_Geometry
  Core_Persistent 
  Core_Thread
  Core_Util
  Core_Volume
  ${TEEM_LIBRARY}
  ${PNG_LIBRARY}
  ${Z_LIBRARY}
  ${GL_LIBRARY}
  ${M_LIBRARY}
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Algorithms_Visualization)
ENDIF(NOT BUILD_SHARED_LIBS)

