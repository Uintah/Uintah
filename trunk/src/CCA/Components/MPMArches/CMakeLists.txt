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

# CMakeLists.txt for Packages/Uintah/CCA/Components/MPMArches

FORTRAN_DEPENDENCY(fortran/collect_drag_cc.F CCA_Components_MPMArches)
FORTRAN_DEPENDENCY(fortran/collect_scalar_fctocc.F CCA_Components_MPMArches)
FORTRAN_DEPENDENCY(fortran/energy_exchange_term.F CCA_Components_MPMArches)
FORTRAN_DEPENDENCY(fortran/interp_centertoface.F CCA_Components_MPMArches)
FORTRAN_DEPENDENCY(fortran/momentum_exchange_term_continuous_cc.F CCA_Components_MPMArches)
FORTRAN_DEPENDENCY(fortran/pressure_force.F CCA_Components_MPMArches)
FORTRAN_DEPENDENCY(fortran/read_complex_geometry.F CCA_Components_MPMArches)
FORTRAN_DEPENDENCY(fortran/read_complex_geometry_walls.F CCA_Components_MPMArches)
FORTRAN_DEPENDENCY(fortran/taucal_cc.F CCA_Components_MPMArches)
FORTRAN_DEPENDENCY(fortran/walmom_cc.F CCA_Components_MPMArches)

SET(Packages_Uintah_CCA_Components_MPMArches_SRCS
  ${CCA_Components_MPMArches_FORTRAN_SRCS}
  MPMArches.cc
  MPMArchesLabel.cc
  CutCellInfo.cc

  # fortran files without fspec
  fortran/arrass.F
)

ADD_LIBRARY(Packages_Uintah_CCA_Components_MPMArches ${Packages_Uintah_CCA_Components_MPMArches_SRCS})
SET_TARGET_PROPERTIES(Packages_Uintah_CCA_Components_MPMArches PROPERTIES LINKER_LANGUAGE CXX)

TARGET_LINK_LIBRARIES(Packages_Uintah_CCA_Components_MPMArches
  Packages_Uintah_CCA_Ports
  Packages_Uintah_Core_Grid
  Packages_Uintah_Core_Util
  Packages_Uintah_Core_Labels
  Packages_Uintah_Core_Disclosure
  Packages_Uintah_Core_Parallel
  Packages_Uintah_Core_ProblemSpec
  Packages_Uintah_Core_Exceptions
  Packages_Uintah_Core_Math
  Packages_Uintah_CCA_Components_MPM
  Packages_Uintah_CCA_Components_Arches
  Packages_Uintah_CCA_Components_Arches_Mixing
  Core_Exceptions
  Core_Util
  Core_Thread
  Core_Geometry
  ${XML2_LIBRARY}
  ${PETSC_LIBRARY}
  ${M_LIBRARY}
  ${F_LIBRARY}
)

IF(BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Packages_Uintah_CCA_Components_MPMArches)
ENDIF(BUILD_SHARED_LIBS)

