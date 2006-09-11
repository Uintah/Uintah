# Sources of Core/Basis classes
SET(Core_Basis_SRCS
  CrvLinearLgn.cc
  CrvQuadraticLgn.cc
  HexTrilinearLgn.cc
  HexTriquadraticLgn.cc
  Locate.cc
  PrismLinearLgn.cc
  PrismQuadraticLgn.cc
  QuadBilinearLgn.cc
  QuadBiquadraticLgn.cc
  TetLinearLgn.cc
  TetQuadraticLgn.cc
  TriLinearLgn.cc
  TriQuadraticLgn.cc
  TriCubicHmt.cc
)

ADD_LIBRARY(Core_Basis ${Core_Basis_SRCS})

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Basis)
ENDIF(NOT BUILD_SHARED_LIBS)
