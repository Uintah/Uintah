# Sources of Core/Algorithms/Converter classes
SET(Core_Algorithms_Converter_SRCS
  ConverterAlgo.cc
  MatrixToField.cc
  NrrdToField.cc
  MatricesToDipoleField.cc
)

ADD_LIBRARY(Core_Algorithms_Converter ${Core_Algorithms_Converter_SRCS})

TARGET_LINK_LIBRARIES(Core_Algorithms_Converter 
  Core_Datatypes 
  Core_Util 
  Core_Containers 
  Core_Exceptions 
  Core_Thread 
  Core_GuiInterface             
  Core_Geom 
  Core_Geometry 
  Core_Algorithms_Util            
  Core_Persistent 
  Core_Basis 
  Core_Bundle
  ${M_LIBRARY} 
  ${TEEM_LIBRARY}
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Algorithms_Converter)
ENDIF(NOT BUILD_SHARED_LIBS)

