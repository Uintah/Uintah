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

# CMakeLists.txt for Packages/Uintah/CCA/Components/Arches

FORTRAN_DEPENDENCY(fortran/areain.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/inlpresbcinout.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/bcpress.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/bcscalar.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/bcuvel.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/bcvvel.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/bcwvel.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/celltypeInit.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/inlbcs.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/intrusion_computevel.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/mm_computevel.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/mm_explicit.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/mm_explicit_oldvalue.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/mm_explicit_vel.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/mmbcvelocity.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/mmbcvelocity_momex.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/mmbcenthalpy_energyex.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/mmcelltypeinit.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/mmenthalpywallbc.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/mmscalarwallbc.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/mmwallbc.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/mmwallbc_trans.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/profscalar.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/profv.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/cellg.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/apcal_all.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/mm_modify_prescoef.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/prescoef.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/prescoef_var.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/scalcoef.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/uvelcoef.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/vvelcoef.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/wvelcoef.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/add_hydrostatic_term_topressure.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/normpress.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/explicit_scalar.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/explicit_vel.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/computeVel.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/scalarvarmodel.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/smagmodel.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/add_mm_enth_src.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/enthalpyradthinsrc.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/mascal.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/mascal_scalar.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/mmmomsrc.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/pressrcpred.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/pressrcpred_var.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/scalsrc.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/uvelsrc.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/vvelsrc.F CCA_Components_Arches)
FORTRAN_DEPENDENCY(fortran/wvelsrc.F CCA_Components_Arches)

SET(Packages_Uintah_CCA_Components_Arches_SRCS
  ${CCA_Components_Arches_FORTRAN_SRCS}
  Arches.cc
  ArchesLabel.cc
  ArchesMaterial.cc
  ArchesVariables.cc
  ArchesConstVariables.cc
  BoundaryCondition.cc
  CellInformation.cc
  Discretization.cc
  EnthalpySolver.cc
  ExplicitSolver.cc
  LinearSolver.cc
  MomentumSolver.cc
  NonlinearSolver.cc
  PhysicalConstants.cc
  PicardNonlinearSolver.cc
  PressureSolver.cc
  Properties.cc
  ReactiveScalarSolver.cc
  RHSSolver.cc
  ScalarSolver.cc
  SmagorinskyModel.cc
  ScaleSimilarityModel.cc
  IncDynamicProcedure.cc
  CompDynamicProcedure.cc
  CompLocalDynamicProcedure.cc
  OdtClosure.cc
  OdtData.cc
  Source.cc
  TurbulenceModel.cc
)

SET(LIBS 
  Packages_Uintah_Core_ProblemSpec
  Packages_Uintah_Core_GeometryPiece
  Packages_Uintah_Core_Grid
  Packages_Uintah_Core_Util
  Packages_Uintah_Core_Disclosure
  Packages_Uintah_Core_Exceptions
  Packages_Uintah_CCA_Components_Arches_Mixing
  Packages_Uintah_CCA_Components_Arches_Radiation
  Packages_Uintah_CCA_Components_OnTheFlyAnalysis
  Packages_Uintah_CCA_Ports
  Packages_Uintah_Core_Parallel
  Core_Util
  Core_Thread
  Core_Exceptions
  Core_Geometry
  Core_Containers
  ${LIBS}
  ${XML2_LIBRARY}
  ${F_LIBRARY}
  ${MPI_LIBRARY}
  ${M_LIBRARY}
)

IF(PETSC_LIBRARY)
  SET(Packages_Uintah_CCA_Components_Arches_SRCS ${Packages_Uintah_CCA_Components_Arches_SRCS} PetscSolver.cc Filter.cc)
  SET(LIBS ${LIBS} ${PETSC_LIBRARIES} 
ELSE(PETSC_LIBRARY)
  SET(Packages_Uintah_CCA_Components_Arches_SRCS ${Packages_Uintah_CCA_Components_Arches_SRCS} FakePetscSolver.cc)
ENDIF(PETSC_LIBRARY)

IF(HYPRE_LIBRARY)
  SET(Packages_Uintah_CCA_Components_Arches_SRCS ${Packages_Uintah_CCA_Components_Arches_SRCS} HypreSolver.cc)
  SET(LIBS ${LIBS} ${HYPRE_LIBRARY} 
ENDIF(HYPRE_LIBRARY)


ADD_LIBRARY(Packages_Uintah_CCA_Components_Arches ${Packages_Uintah_CCA_Components_Arches_SRCS})
SET_TARGET_PROPERTIES(Packages_Uintah_CCA_Components_Arches PROPERTIES LINKER_LANGUAGE CXX)

TARGET_LINK_LIBRARIES(Packages_Uintah_CCA_Components_Arches ${LIBS})

ADD_SUBDIRECTORY(Mixing)
ADD_SUBDIRECTORY(Radiation)

IF(BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Packages_Uintah_CCA_Components_Arches)
ENDIF(BUILD_SHARED_LIBS)
