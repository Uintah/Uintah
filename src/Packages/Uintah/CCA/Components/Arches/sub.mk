# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Arches

SRCS     += $(SRCDIR)/Arches.cc \
	$(SRCDIR)/ArchesLabel.cc \
	$(SRCDIR)/ArchesMaterial.cc \
	$(SRCDIR)/ArchesVariables.cc \
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
	$(SRCDIR)/Source.cc \
	$(SRCDIR)/TurbulenceModel.cc

ifneq ($(PETSC_DIR),)
SRCS +=	$(SRCDIR)/PetscSolver.cc
else
SRCS +=	$(SRCDIR)/FakePetscSolver.cc
endif

SUBDIRS := $(SRCDIR)/fortran $(SRCDIR)/Mixing
include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Exceptions  \
	Core/Thread 			 \
	Core/Geometry                    \
	Core/Exceptions

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) -lm
ifneq ($(PETSC_DIR),)
LIBS := $(LIBS) $(PETSC_LIBS) -lpetscsles -lpetscdm -lpetscmat -lpetscvec -lpetsc -lblas
endif
LIBS := $(LIBS) $(FLIBS) 
ifneq ($(PETSC_DIR),)
CFLAGS +=	-DHAVE_PETSC
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

$(SRCDIR)/Arches.o: $(SRCDIR)/fortran/init_fort.h
$(SRCDIR)/Arches.o: $(SRCDIR)/fortran/initScal_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/addpressuregrad_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/areain_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/bcenthalpy_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/bcinout_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/bcpress_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/bcscalar_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/bcuvel_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/bcvvel_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/bcwvel_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/calpbc_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/celltypeInit_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/denaccum_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/enthalpyradwallbc_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/inlbcs_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/mmbcvelocity_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/mmcelltypeinit_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/mmwallbc_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/mm_computevel_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/outarea_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/outletbc_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/outletbcenth_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/profscalar_fort.h
$(SRCDIR)/BoundaryCondition.o: $(SRCDIR)/fortran/profv_fort.h
$(SRCDIR)/CellInformation.o: $(SRCDIR)/fortran/cellg_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/apcal_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/apcal_vel_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/explicit_vel_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/mm_modify_prescoef_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/prescoef_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/scalcoef_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/uvelcoef_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/vvelcoef_fort.h
$(SRCDIR)/Discretization.o: $(SRCDIR)/fortran/wvelcoef_fort.h
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
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/pressrc_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/scalsrc_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/uvelsrc_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/vvelsrc_fort.h
$(SRCDIR)/Source.o: $(SRCDIR)/fortran/wvelsrc_fort.h
