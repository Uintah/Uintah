# Sources of Core/Init classes
SET(Core_Init_SRCS
  init.cc 
  NrrdToField_Plugin.cc 
  NrrdToMatrix_Plugin.cc 
  SimpleTextFileToMatrix_Plugin.cc 
)

ADD_LIBRARY(Core_Init ${Core_Init_SRCS})

TARGET_LINK_LIBRARIES(Core_Init 
  Core_Algorithms_DataIO 
  Core_Algorithms_Converter 
  Core_Containers   
  Core_Datatypes    
  Core_Exceptions   
  Core_Geometry     
  Core_ImportExport 
  Core_Persistent   
  Core_Thread       
  Core_Util 
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Init)
ENDIF(NOT BUILD_SHARED_LIBS)

