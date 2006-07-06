# Sources of Core/Math classes
SET(Core_Math_SRCS
  CubicPWI.cc 
  Gaussian.cc
  LinAlg.c
  LinearPWI.cc
  Mat.c
  MiscMath.cc
  MusilRNG.cc
  PiecewiseInterp.cc
  TrigTable.cc
  sci_lapack.cc
  fft.c	
  ssmult.c
)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Math)
ENDIF(NOT BUILD_SHARED_LIBS)

ADD_LIBRARY(Core_Math ${Core_Math_SRCS} )

# DV Fix defining below libraries
TARGET_LINK_LIBRARIES(Core_Math 
  Core_Exceptions 
  Core_Containers
  ${M_LIBRARY}
  ${DL_LIBRARY}
  ${LAPACK_LIBRARY}
  ${BLAS_LIBRARY}
  ${F_LIBRARY}
)

# Testing class instantiations
ADD_EXECUTABLE(testMiscMath ${CMAKE_SOURCE_DIR}/Testing/src/Core/Math/testMiscMath.cc)

TARGET_LINK_LIBRARIES(testMiscMath Core_Math)

ADD_TEST(Core_Math_MiscMath_Test ${EXECUTABLE_OUTPUT_PATH}/testMiscMath)