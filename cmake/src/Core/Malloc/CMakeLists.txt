# Sources of Core/Malloc classes
SET(Core_Malloc_SRCS
  Allocator.cc 
  AllocOS.cc 
  malloc.cc 
  new.cc 
)

ADD_LIBRARY(Core_Malloc ${Core_Malloc_SRCS})

TARGET_LINK_LIBRARIES(Core_Malloc 
  Core/Thread       
)