# 
# 
# The MIT License
# 
# Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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

SRCS += $(SRCDIR)/Arches.cc \
        $(SRCDIR)/ArchesLabel.cc \
        $(SRCDIR)/ArchesMaterial.cc \
        $(SRCDIR)/ArchesVariables.cc \
        $(SRCDIR)/ArchesConstVariables.cc \
        $(SRCDIR)/BoundaryCondition.cc \
        $(SRCDIR)/CellInformation.cc \
        $(SRCDIR)/Discretization.cc \
        $(SRCDIR)/EnthalpySolver.cc \
        $(SRCDIR)/ExplicitSolver.cc \
        $(SRCDIR)/LinearSolver.cc \
        $(SRCDIR)/MomentumSolver.cc \
        $(SRCDIR)/NonlinearSolver.cc \
        $(SRCDIR)/PhysicalConstants.cc \
        $(SRCDIR)/PicardNonlinearSolver.cc \
        $(SRCDIR)/PressureSolver.cc \
        $(SRCDIR)/Properties.cc \
        $(SRCDIR)/ReactiveScalarSolver.cc \
        $(SRCDIR)/RHSSolver.cc \
        $(SRCDIR)/ScalarSolver.cc \
        $(SRCDIR)/ExtraScalarSolver.cc \
        $(SRCDIR)/ExtraScalarSrc.cc \
        $(SRCDIR)/ExtraScalarSrcFactory.cc \
        $(SRCDIR)/ZeroExtraScalarSrc.cc \
        $(SRCDIR)/CO2RateSrc.cc \
        $(SRCDIR)/SO2RateSrc.cc \
        $(SRCDIR)/SmagorinskyModel.cc \
        $(SRCDIR)/ScaleSimilarityModel.cc \
        $(SRCDIR)/IncDynamicProcedure.cc \
        $(SRCDIR)/CompDynamicProcedure.cc \
        $(SRCDIR)/CompLocalDynamicProcedure.cc \
        $(SRCDIR)/OdtClosure.cc \
        $(SRCDIR)/OdtData.cc \
        $(SRCDIR)/Source.cc \
        $(SRCDIR)/TurbulenceModel.cc \
	$(SRCDIR)/DQMOM.cc \
	$(SRCDIR)/LU.cc \
	$(SRCDIR)/BoundaryCond_new.cc \
	$(SRCDIR)/ExplicitTimeInt.cc 

ifeq ($(HAVE_PETSC),yes)
  SRCS += $(SRCDIR)/PetscSolver.cc $(SRCDIR)/Filter.cc
else
  SRCS += $(SRCDIR)/FakePetscSolver.cc
endif

ifeq ($(HAVE_HYPRE),yes)
  SRCS += $(SRCDIR)/HypreSolver.cc
endif

SUBDIRS := $(SRCDIR)/CoalModels $(SRCDIR)/SourceTerms $(SRCDIR)/TransportEqns $(SRCDIR)/ChemMix
include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
        Core/ProblemSpec   \
        Core/GeometryPiece \
        Core/Grid          \
        Core/Datatypes     \
        Core/Math          \
        Core/Util          \
        Core/Disclosure    \
        Core/Exceptions    \
        CCA/Components/Arches/fortran   \
        CCA/Components/Arches/Mixing    \
        CCA/Components/Arches/MCRT/ArchesRMCRT  \
        CCA/Components/Arches/Radiation \
        CCA/Components/Arches/ChemMix/TabProps  \
        CCA/Components/OnTheFlyAnalysis \
        CCA/Ports     \
        Core/Parallel \
        Core/Util       \
        Core/Thread     \
        Core/Exceptions \
        Core/Geometry   \
        Core/Containers \
	\
	Core/Math

ifneq ($(HAVE_PETSC),)
  LIBS := $(LIBS) $(PETSC_LIBRARY) 
endif

ifneq ($(HAVE_HYPRE),)
  LIBS := $(LIBS) $(HYPRE_LIBRARY) 
endif

LIBS := $(LIBS) $(XML2_LIBRARY) $(F_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) \
        $(LAPACK_LIBRARY) $(BLAS_LIBRARY) $(THREAD_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

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
