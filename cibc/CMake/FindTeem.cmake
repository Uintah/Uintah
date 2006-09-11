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
  /usr/lib
  /usr/local/lib
  ${SCIRUN_THIRDPARTY_PATH}/lib
)

FIND_PATH(TEEM_INCLUDE_DIR teem/nrrd.h
  /usr/local/include
  /usr/include
  ${SCIRUN_THIRDPARTY_PATH}/include
 )


IF (TEEM_LIBRARY)
  IF (TEEM_INCLUDE_DIR)
    SET(TEEM_FOUND "YES")
    SET(DEF_TEEM "#define HAVE_TEEM 1")	
  ENDIF(TEEM_INCLUDE_DIR)
ENDIF(TEEM_LIBRARY)


