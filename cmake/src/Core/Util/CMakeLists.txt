# Sources of Core/Util classes
SET(Core_Util_SRCS
  DebugStream.cc        
  DynamicLoader.cc      
  DynamicCompilation.cc 
  Endian.cc             
  Environment.cc        
  FileUtils.cc          
  ProgressReporter.cc   
  SimpleProfiler.cc     
  SizeTypeConvert.cc    
  RWS.cc                
  sci_system.cc         
  soloader.cc           
  Signals.cc            
  Timer.cc              
  TypeDescription.cc    
  ProgressiveWarning.cc
)

ADD_LIBRARY(Core_Util ${Core_Util_SRCS})

TARGET_LINK_LIBRARIES(Core_Util 
  Core_Containers 
  Core_Exceptions 
  Core_Thread
  ${DL_LIBRARY}
  ${THREAD_LIBRARY}
)

IF(WIN32)
  TARGET_LINK_LIBRARIES(Core_Util Core_OS)
ENDIF(WIN32)	

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Util)
ENDIF(NOT BUILD_SHARED_LIBS)

