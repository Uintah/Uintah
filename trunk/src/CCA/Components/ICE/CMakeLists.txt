#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#

# CMakeLists.txt for Packages/Uintah/CCA/Components/ICE

SET(Packages_Uintah_CCA_Components_ICE_SRCS
  ICE.cc
  AMRICE.cc
  ICERF.cc
  ICEDebug.cc
  ICEMaterial.cc
  Diffusion.cc
  BoundaryCond.cc
  Turbulence.cc
  SmagorinskyModel.cc
  DynamicModel.cc
  TurbulenceFactory.cc
  impICE.cc
  impAMRICE.cc
  customInitialize.cc
  ExchangeCoefficients.cc

  EOS/EquationOfState.cc
  EOS/EquationOfStateFactory.cc
  EOS/IdealGas.cc
  EOS/TST.cc
  EOS/JWL.cc
  EOS/JWLC.cc
  EOS/Gruneisen.cc
  EOS/Tillotson.cc
  EOS/Murnahan.cc

  Advection/Advector.cc
  Advection/FirstOrderAdvector.cc
  Advection/SecondOrderAdvector.cc
  Advection/SecondOrderBase.cc
  Advection/AdvectionFactory.cc

  CustomBCs/C_BC_driver.cc
  CustomBCs/microSlipBCs.cc
  CustomBCs/LODI2.cc
  CustomBCs/MMS_BCs.cc

)

ADD_LIBRARY(Packages_Uintah_CCA_Components_ICE ${Packages_Uintah_CCA_Components_ICE_SRCS})

TARGET_LINK_LIBRARIES(Packages_Uintah_CCA_Components_ICE
  Packages_Uintah_CCA_Components_MPM
  Packages_Uintah_CCA_Components_OnTheFlyAnalysis
  Packages_Uintah_CCA_Ports
  Packages_Uintah_Core_Grid
  Packages_Uintah_Core_Util
  Packages_Uintah_Core_Math
  Packages_Uintah_Core_Disclosure
  Packages_Uintah_Core_GeometryPiece
  Packages_Uintah_Core_ProblemSpec
  Packages_Uintah_Core_Parallel
  Packages_Uintah_Core_Exceptions
  Packages_Uintah_Core_Math
  Packages_Uintah_Core_Labels
  Core_Exceptions
  Core_Geometry
  Core_Thread
  Core_Util
  Core_OS
  ${XML2_LIBRARY}
  ${MPI_LIBRARY}
  ${M_LIBRARY}
)

IF(BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Packages_Uintah_CCA_Components_ICE)
ENDIF(BUILD_SHARED_LIBS)
SUBDIRS(EOS Advection CustomBCs)

