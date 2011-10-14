# 
# 
# The MIT License
# 
# Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
# Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
# University of Utah.
# 
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
# 
# 
# 
# 
# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := CCA/Components/Arches

SRCS += $(SRCDIR)/Arches.cc                    \
        $(SRCDIR)/ArchesConstVariables.cc      \
        $(SRCDIR)/ArchesLabel.cc               \
        $(SRCDIR)/ArchesMaterial.cc            \
        $(SRCDIR)/ArchesVariables.cc           \
        $(SRCDIR)/BoundaryCondition.cc         \
        $(SRCDIR)/BoundaryCond_new.cc          \
        $(SRCDIR)/CellInformation.cc           \
        $(SRCDIR)/CompDynamicProcedure.cc      \
        $(SRCDIR)/CompLocalDynamicProcedure.cc \
        $(SRCDIR)/Discretization.cc            \
        $(SRCDIR)/DQMOM.cc                     \
        $(SRCDIR)/EnthalpySolver.cc            \
        $(SRCDIR)/ExplicitSolver.cc            \
        $(SRCDIR)/ExplicitTimeInt.cc           \
        $(SRCDIR)/IncDynamicProcedure.cc       \
        $(SRCDIR)/IntrusionBC.cc               \
        $(SRCDIR)/LinearSolver.cc              \
        $(SRCDIR)/LU.cc                        \
        $(SRCDIR)/MomentumSolver.cc            \
        $(SRCDIR)/NonlinearSolver.cc           \
        $(SRCDIR)/OdtClosure.cc                \
        $(SRCDIR)/OdtData.cc                   \
        $(SRCDIR)/PhysicalConstants.cc         \
        $(SRCDIR)/PicardNonlinearSolver.cc     \
        $(SRCDIR)/PressureSolverV2.cc          \
        $(SRCDIR)/Properties.cc                \
        $(SRCDIR)/RHSSolver.cc                 \
        $(SRCDIR)/ScalarSolver.cc              \
        $(SRCDIR)/ScaleSimilarityModel.cc      \
        $(SRCDIR)/SmagorinskyModel.cc          \
        $(SRCDIR)/Source.cc                    \
        $(SRCDIR)/OperatorSplitChem.cc         \
        $(SRCDIR)/TurbulenceModel.cc           

ifeq ($(HAVE_PETSC),yes)
  SRCS += $(SRCDIR)/PetscCommon.cc \
          $(SRCDIR)/PetscSolver.cc \
          $(SRCDIR)/Filter.cc      
else
  SRCS += $(SRCDIR)/FakePetscSolver.cc
endif

ifeq ($(HAVE_HYPRE),yes)
  SRCS += $(SRCDIR)/HypreSolverV2.cc
endif

PSELIBS := \
        CCA/Components/Arches/fortran   \
        CCA/Components/Arches/Mixing    \
        CCA/Components/Models           \
        CCA/Components/OnTheFlyAnalysis \
        CCA/Ports          \
        Core/Containers    \
        Core/Datatypes     \
        Core/Disclosure    \
        Core/Exceptions    \
        Core/Geometry      \
        Core/GeometryPiece \
        Core/IO            \
        Core/Grid          \
        Core/Math          \
        Core/Parallel      \
        Core/ProblemSpec   \
        Core/Thread        \
        Core/Util

ifneq ($(HAVE_PETSC),)
  LIBS := $(LIBS) $(PETSC_LIBRARY) 
endif

ifneq ($(HAVE_HYPRE),)
  LIBS := $(LIBS) $(HYPRE_LIBRARY) 
endif

LIBS := $(LIBS) $(XML2_LIBRARY) $(F_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) \
        $(LAPACK_LIBRARY) $(BLAS_LIBRARY) $(THREAD_LIBRARY) $(TABPROPS_LIBRARY) \
        $(BOOST_LIBRARY)

INCLUDES := $(INCLUDES) $(BOOST_INCLUDE) $(TABPROPS_INCLUDE)

#### Handle subdirs (These files are just 'included' into the build of libCCA_Components_Arches.so)
SUBDIRS := $(SRCDIR)/ChemMix            \
           $(SRCDIR)/CoalModels         \
           $(SRCDIR)/CoalModels/fortran \
           $(SRCDIR)/PropertyModels     \
           $(SRCDIR)/Radiation          \
           $(SRCDIR)/Radiation/fortran  \
           $(SRCDIR)/SourceTerms        \
           $(SRCDIR)/TransportEqns      

include $(SCIRUN_SCRIPTS)/recurse.mk
#### End handle subdirs

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

#### Handle subdirs that are their OWN SHARED LIBRARIES
# I don't know of any reason that these are actually made into separate libraries...
# perhaps just for historical reasons.  It would be just as easy to fold them into
# libArches if there ever was a reason to...
SUBDIRS := $(SRCDIR)/fortran           \
           $(SRCDIR)/Mixing            

include $(SCIRUN_SCRIPTS)/recurse.mk
#### End handle subdirs

$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/areain_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/inlpresbcinout_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/bcscalar_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/bcuvel_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/bcvvel_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/bcwvel_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/celltypeInit_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/inlbcs_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/intrusion_computevel_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/mm_computevel_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/mm_explicit_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/mm_explicit_oldvalue_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/mm_explicit_vel_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/mmbcvelocity_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/mmbcvelocity_momex_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/mmbcenthalpy_energyex_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/mmcelltypeinit_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/mmwallbc_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/mmwallbc_trans_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/profscalar_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/profv_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/get_ramping_factor_fort.h
$(SRCDIR)/CellInformation.$(OBJEXT): $(SRCDIR)/fortran/cellg_fort.h
$(SRCDIR)/Discretization.$(OBJEXT): $(SRCDIR)/fortran/apcal_all_fort.h
$(SRCDIR)/Discretization.$(OBJEXT): $(SRCDIR)/fortran/prescoef_var_fort.h
$(SRCDIR)/Discretization.$(OBJEXT): $(SRCDIR)/fortran/scalcoef_fort.h
$(SRCDIR)/Discretization.$(OBJEXT): $(SRCDIR)/fortran/uvelcoef_fort.h
$(SRCDIR)/Discretization.$(OBJEXT): $(SRCDIR)/fortran/vvelcoef_fort.h
$(SRCDIR)/Discretization.$(OBJEXT): $(SRCDIR)/fortran/wvelcoef_fort.h
$(SRCDIR)/RHSSolver.$(OBJEXT): $(SRCDIR)/fortran/explicit_scalar_fort.h
$(SRCDIR)/SmagorinskyModel.$(OBJEXT): $(SRCDIR)/fortran/scalarvarmodel_fort.h
$(SRCDIR)/SmagorinskyModel.$(OBJEXT): $(SRCDIR)/fortran/smagmodel_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/add_mm_enth_src_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/enthalpyradthinsrc_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/mascal_scalar_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/pressrcpred_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/pressrcpred_var_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/scalsrc_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/uvelsrc_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/vvelsrc_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/wvelsrc_fort.h
