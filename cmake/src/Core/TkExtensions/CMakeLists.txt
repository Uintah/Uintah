# Sources of Core/TkExtensions classes
SET(Core_TkExtensions_SRCS
  tk3d2.c
  tkBevel.c
  tkCanvBLine.c
  tkCursor.c
  tkOpenGL.c
  tk3daux.c
  tkRange.c
  tkUnixRange.c
)

IF(WIN32 AND NOT CYGWIN)
  SET(Core_TkExtensions_SRCS ${Core_TkExtensions_SRC} tclUnixNotify-${TK_VERSION}.c)
ENDIF(WIN32 AND NOT CYGWIN)

ADD_LIBRARY(Core_TkExtensions ${Core_TkExtensions_SRCS})

TARGET_LINK_LIBRARIES( Core_TkExtensions 
  Core_Thread
  ${BLT_LIBRARY}
  ${PLPLOT_LIBRARY}
  ${ITK_LIBRARY}
  ${ITCL_LIBRARY}
  ${TK_LIBRARY}
  ${TCL_LIBRARY}
  ${GL_LIBRARY} 
  ${M_LIBRARY}
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_TkExtensions)
ENDIF(NOT BUILD_SHARED_LIBS)
