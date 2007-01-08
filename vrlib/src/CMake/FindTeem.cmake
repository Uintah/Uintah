#
# Find the Teem library
#

#
# This module finds the Teem include files and libraries. 
# It also determines what the name of the library is. 
# This code sets the following variables:
#
# TEEM_LIBRARY the libraries to link against when using Teem
# TEEM_INCLUDE where to find teem/nrrd.h, etc.
#

# DV Fix finding teem library in thirdparty
FIND_LIBRARY(TEEM_LIBRARY teem
  PATHS ${SCIRUN_THIRDPARTY_DIR}/lib
  /usr/local/lib
  /usr/lib
)

FIND_PATH(TEEM_INCLUDE_DIR teem/nrrd.h
  ${SCIRUN_THIRDPARTY_DIR}/include
  /usr/local/include
  /usr/include
 )


IF (TEEM_LIBRARY)
  IF (TEEM_INCLUDE_DIR)
    SET(TEEM_FOUND "YES")
    SET(DEF_TEEM "#define HAVE_TEEM 1")	
  ENDIF(TEEM_INCLUDE_DIR)
ENDIF(TEEM_LIBRARY)

