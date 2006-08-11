# Sources of Core/GuiInterface classes
SET(Core_GuiInterface_SRCS
  GuiCallback.cc
  GuiContext.cc
  GuiInterface.cc
  GuiVar.cc
  MemStats.cc
  TCLInterface.cc
  TclObj.cc
  TCLTask.cc
  UIvar.cc
)

ADD_LIBRARY(Core_GuiInterface ${Core_GuiInterface_SRCS})

TARGET_LINK_LIBRARIES(Core_GuiInterface 
  Core_Containers 
  Core_Exceptions 
  Core_Thread
  Core_Util
  Core_TkExtensions
  Core_Math
  ${ITK_LIBRARY}
  ${ITCL_LIBRARY}
  ${TCL_LIBRARY}
  ${X_LIBRARY}
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_GuiInterface)
ENDIF(NOT BUILD_SHARED_LIBS)
