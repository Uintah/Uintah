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
	$(SRCDIR)/RBGSSolver.cc \
	$(SRCDIR)/ScalarSolver.cc \
	$(SRCDIR)/SmagorinskyModel.cc \
	$(SRCDIR)/ScaleSimilarityModel.cc \
	$(SRCDIR)/DynamicProcedure.cc \
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
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/CCA/Components/Arches/fortran \
	Packages/Uintah/CCA/Components/Arches/Mixing \
        Packages/Uintah/CCA/Components/Arches/Radiation \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/Core/Parallel \
        Core/Util \
        Core/Thread \
	Core/Exceptions \
	Core/Geometry

ifneq ($(HAVE_PETSC),)
LIBS := $(LIBS) $(PETSC_LIBRARY) 
endif

ifneq ($(HAVE_HYPRE),)
LIBS := $(LIBS) $(HYPRE_LIBRARY) 
endif

LIBS := $(LIBS) $(XML_LIBRARY) $(F_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

$(SRCDIR)/Arches.o: $(SRCDIR)/fortran/init_fort.h
$(SRCDIR)/Arches.o: $(SRCDIR)/fortran/initScal_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/addpressuregrad_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/areain_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/bcenthalpy_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/bcinout_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/inlpresbcinout_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/bcpress_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/bcscalar_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/bcuvel_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/bcvvel_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/bcwvel_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/calpbc_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/celltypeInit_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/denaccum_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/enthalpyradwallbc_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/hatvelcalpbc_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/inlbcs_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/intrusion_computevel_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/mm_computevel_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/mm_explicit_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/mm_explicit_oldvalue_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/mm_explicit_vel_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/mmbcvelocity_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/mmbcvelocity_momex_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/mmbcenthalpy_energyex_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/mmcelltypeinit_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/mmwallbc_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/mmwallbc_trans_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/outarea_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/outletbc_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/outletbcenth_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/outletbcrscal_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/profscalar_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/profv_fort.h
$(SRCDIR)/CellInformation.o: $(SRCDIR)/fortran/cellg_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/apcal_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/apcal_vel_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/explicit_vel_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/mm_modify_prescoef_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/prescoef_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/prescoef_var_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/scalcoef_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/uvelcoef_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/vvelcoef_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/wvelcoef_fort.h
$(SRCDIR)/DynamicProcedure.o: $(SRCDIR)/fortran/dynamic_1loop_fort.h
$(SRCDIR)/DynamicProcedure.o: $(SRCDIR)/fortran/dynamic_2loop_fort.h
$(SRCDIR)/DynamicProcedure.o: $(SRCDIR)/fortran/dynamic_3loop_fort.h
$(SRCDIR)/HypreSolver.o: $(SRCDIR)/fortran/rescal_fort.h
$(SRCDIR)/HypreSolver.o: $(SRCDIR)/fortran/underelax_fort.h
$(SRCDIR)/PetscSolver.o: $(SRCDIR)/fortran/rescal_fort.h
$(SRCDIR)/PetscSolver.o: $(SRCDIR)/fortran/underelax_fort.h
$(SRCDIR)/PressureSolver.o: $(SRCDIR)/fortran/add_hydrostatic_term_topressure_fort.h
$(SRCDIR)/PressureSolver.o: $(SRCDIR)/fortran/normpress_fort.h
$(SRCDIR)/RBGSSolver.o: $(SRCDIR)/fortran/explicit_fort.h
$(SRCDIR)/RBGSSolver.o: $(SRCDIR)/fortran/explicit_velocity_fort.h
$(SRCDIR)/RBGSSolver.o: $(SRCDIR)/fortran/linegs_fort.h
$(SRCDIR)/RBGSSolver.o: $(SRCDIR)/fortran/rescal_fort.h
$(SRCDIR)/RBGSSolver.o: $(SRCDIR)/fortran/underelax_fort.h
$(SRCDIR)/SmagorinskyModel.o: $(SRCDIR)/fortran/scalarvarmodel_fort.h
$(SRCDIR)/SmagorinskyModel.o: $(SRCDIR)/fortran/smagmodel_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/add_mm_enth_src_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/addpressgrad_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/addtranssrc_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/calcpressgrad_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/computeVel_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/enthalpyradflux_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/enthalpyradsrc_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/enthalpyradthinsrc_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/mascal_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/mascal_scalar_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/mmmomsrc_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/pressrccorr_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/pressrcpred_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/pressrcpred_var_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/pressrc_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/scalsrc_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/uvelsrc_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/vvelsrc_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/wvelsrc_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/uvelcoeffupdate_fort.h
$(SRCDIR)/MomentumSolver.o: $(SRCDIR)/fortran/computeVel_fort.h
