# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Arches

SRCS     += $(SRCDIR)/Arches.cc \
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
        $(SRCDIR)/SmagorinskyModel.cc \
        $(SRCDIR)/ScaleSimilarityModel.cc \
        $(SRCDIR)/IncDynamicProcedure.cc \
        $(SRCDIR)/CompDynamicProcedure.cc \
        $(SRCDIR)/CompLocalDynamicProcedure.cc \
        $(SRCDIR)/OdtClosure.cc \
        $(SRCDIR)/OdtData.cc \
        $(SRCDIR)/Source.cc \
        $(SRCDIR)/TurbulenceModel.cc

ifeq ($(HAVE_PETSC),yes)
  SRCS += $(SRCDIR)/PetscSolver.cc $(SRCDIR)/Filter.cc
else
  SRCS += $(SRCDIR)/FakePetscSolver.cc
endif

ifeq ($(HAVE_HYPRE),yes)
  SRCS += $(SRCDIR)/HypreSolver.cc
endif

# SUBDIRS := $(SRCDIR)/fortran 
# include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
        Packages/Uintah/Core/ProblemSpec   \
        Packages/Uintah/Core/GeometryPiece \
        Packages/Uintah/Core/Grid          \
        Packages/Uintah/Core/Util          \
        Packages/Uintah/Core/Disclosure    \
        Packages/Uintah/Core/Exceptions    \
        Packages/Uintah/CCA/Components/Arches/fortran   \
        Packages/Uintah/CCA/Components/Arches/Mixing    \
        Packages/Uintah/CCA/Components/Arches/Radiation \
        Packages/Uintah/CCA/Components/OnTheFlyAnalysis \
        Packages/Uintah/CCA/Ports     \
        Packages/Uintah/Core/Parallel \
        Core/Util       \
        Core/Thread     \
        Core/Exceptions \
        Core/Geometry   \
        Core/Containers

ifneq ($(HAVE_PETSC),)
  LIBS := $(LIBS) $(PETSC_LIBRARY) 
endif

ifneq ($(HAVE_HYPRE),)
  LIBS := $(LIBS) $(HYPRE_LIBRARY) 
endif

LIBS := $(LIBS) $(XML2_LIBRARY) $(F_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/areain_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/bcenthalpy_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/inlpresbcinout_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/bcpress_fort.h
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
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/mmenthalpywallbc_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/mmscalarwallbc_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/mmwallbc_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/mmwallbc_trans_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/profscalar_fort.h
$(SRCDIR)/BoundaryCondition.$(OBJEXT): $(SRCDIR)/fortran/profv_fort.h
$(SRCDIR)/CellInformation.$(OBJEXT): $(SRCDIR)/fortran/cellg_fort.h
$(SRCDIR)/Discretization.$(OBJEXT): $(SRCDIR)/fortran/apcal_fort.h
$(SRCDIR)/Discretization.$(OBJEXT): $(SRCDIR)/fortran/apcal_vel_fort.h
$(SRCDIR)/Discretization.$(OBJEXT): $(SRCDIR)/fortran/mm_modify_prescoef_fort.h
$(SRCDIR)/Discretization.$(OBJEXT): $(SRCDIR)/fortran/prescoef_fort.h
$(SRCDIR)/Discretization.$(OBJEXT): $(SRCDIR)/fortran/prescoef_var_fort.h
$(SRCDIR)/Discretization.$(OBJEXT): $(SRCDIR)/fortran/scalcoef_fort.h
$(SRCDIR)/Discretization.$(OBJEXT): $(SRCDIR)/fortran/uvelcoef_fort.h
$(SRCDIR)/Discretization.$(OBJEXT): $(SRCDIR)/fortran/vvelcoef_fort.h
$(SRCDIR)/Discretization.$(OBJEXT): $(SRCDIR)/fortran/wvelcoef_fort.h
$(SRCDIR)/IncDynamicProcedure.$(OBJEXT): $(SRCDIR)/fortran/inc_dynamic_1loop_fort.h
$(SRCDIR)/IncDynamicProcedure.$(OBJEXT): $(SRCDIR)/fortran/inc_dynamic_2loop_fort.h
$(SRCDIR)/IncDynamicProcedure.$(OBJEXT): $(SRCDIR)/fortran/inc_dynamic_3loop_fort.h
$(SRCDIR)/CompDynamicProcedure.$(OBJEXT): $(SRCDIR)/fortran/comp_dynamic_1loop_fort.h
$(SRCDIR)/CompDynamicProcedure.$(OBJEXT): $(SRCDIR)/fortran/comp_dynamic_2loop_fort.h
$(SRCDIR)/CompDynamicProcedure.$(OBJEXT): $(SRCDIR)/fortran/comp_dynamic_3loop_fort.h
$(SRCDIR)/CompDynamicProcedure.$(OBJEXT): $(SRCDIR)/fortran/comp_dynamic_4loop_fort.h
$(SRCDIR)/CompDynamicProcedure.$(OBJEXT): $(SRCDIR)/fortran/comp_dynamic_5loop_fort.h
$(SRCDIR)/CompDynamicProcedure.$(OBJEXT): $(SRCDIR)/fortran/comp_dynamic_6loop_fort.h
$(SRCDIR)/CompDynamicProcedure.$(OBJEXT): $(SRCDIR)/fortran/comp_dynamic_7loop_fort.h
$(SRCDIR)/CompDynamicProcedure.$(OBJEXT): $(SRCDIR)/fortran/comp_dynamic_8loop_fort.h
$(SRCDIR)/PressureSolver.$(OBJEXT): $(SRCDIR)/fortran/add_hydrostatic_term_topressure_fort.h
$(SRCDIR)/PressureSolver.$(OBJEXT): $(SRCDIR)/fortran/normpress_fort.h
$(SRCDIR)/RHSSolver.$(OBJEXT): $(SRCDIR)/fortran/explicit_fort.h
$(SRCDIR)/RHSSolver.$(OBJEXT): $(SRCDIR)/fortran/explicit_vel_fort.h
$(SRCDIR)/SmagorinskyModel.$(OBJEXT): $(SRCDIR)/fortran/scalarvarmodel_fort.h
$(SRCDIR)/SmagorinskyModel.$(OBJEXT): $(SRCDIR)/fortran/smagmodel_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/add_mm_enth_src_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/computeVel_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/enthalpyradthinsrc_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/mascal_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/mascal_scalar_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/mmmomsrc_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/pressrcpred_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/pressrcpred_var_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/scalsrc_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/uvelsrc_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/vvelsrc_fort.h
$(SRCDIR)/Source.$(OBJEXT): $(SRCDIR)/fortran/wvelsrc_fort.h
$(SRCDIR)/MomentumSolver.$(OBJEXT): $(SRCDIR)/fortran/computeVel_fort.h
