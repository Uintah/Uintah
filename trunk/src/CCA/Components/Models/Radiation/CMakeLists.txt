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

# CMakeLists.txt for Packages/Uintah/CCA/Components/Models/Radiation


FORTRAN_DEPENDENCY(fortran/m_cellg.F CCA_Components_Models_Radiation)
FORTRAN_DEPENDENCY(fortran/m_rordr.F CCA_Components_Models_Radiation)
FORTRAN_DEPENDENCY(fortran/m_rordrss.F CCA_Components_Models_Radiation)
FORTRAN_DEPENDENCY(fortran/m_rordrtn.F CCA_Components_Models_Radiation)
FORTRAN_DEPENDENCY(fortran/m_radarray.F CCA_Components_Models_Radiation)
FORTRAN_DEPENDENCY(fortran/m_radcal.F CCA_Components_Models_Radiation)
FORTRAN_DEPENDENCY(fortran/m_radcoef.F CCA_Components_Models_Radiation)
FORTRAN_DEPENDENCY(fortran/m_radwsgg.F CCA_Components_Models_Radiation)
FORTRAN_DEPENDENCY(fortran/m_rdombc.F CCA_Components_Models_Radiation)
FORTRAN_DEPENDENCY(fortran/m_rdomsolve.F CCA_Components_Models_Radiation)
FORTRAN_DEPENDENCY(fortran/m_rdomsrc.F CCA_Components_Models_Radiation)
FORTRAN_DEPENDENCY(fortran/m_rdomflux.F CCA_Components_Models_Radiation)
FORTRAN_DEPENDENCY(fortran/m_rdombmcalc.F CCA_Components_Models_Radiation)
FORTRAN_DEPENDENCY(fortran/m_rdomvolq.F CCA_Components_Models_Radiation)
FORTRAN_DEPENDENCY(fortran/m_rshsolve.F CCA_Components_Models_Radiation)
FORTRAN_DEPENDENCY(fortran/m_rshresults.F CCA_Components_Models_Radiation)

SET(Packages_Uintah_CCA_Components_Models_Radiation_SRCS
  ${CCA_Components_Models_Radiation_FORTRAN_SRCS}
  Models_CellInformation.cc
  Models_DORadiationModel.cc
  RadiationConstVariables.cc
  RadiationDriver.cc
  Models_RadiationModel.cc
  Models_RadiationSolver.cc
  RadiationVariables.cc
)

ADD_LIBRARY(Packages_Uintah_CCA_Components_Models_Radiation ${Packages_Uintah_CCA_Components_Models_Radiation_SRCS})
SET_TARGET_PROPERTIES(Packages_Uintah_CCA_Components_Models_Radiation PROPERTIES LINKER_LANGUAGE CXX)

TARGET_LINK_LIBRARIES(Packages_Uintah_CCA_Components_Models_Radiation

)

IF(BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Packages_Uintah_CCA_Components_Models_Radiation)
ENDIF(BUILD_SHARED_LIBS)
