# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Arches

SRCS     += $(SRCDIR)/Arches.cc $(SRCDIR)/BoundaryCondition.cc \
	$(SRCDIR)/NonlinearSolver.cc $(SRCDIR)/PhysicalConstants.cc \
	$(SRCDIR)/PicardNonlinearSolver.cc $(SRCDIR)/ExplicitSolver.cc \
	$(SRCDIR)/Properties.cc $(SRCDIR)/SmagorinskyModel.cc \
	$(SRCDIR)/TurbulenceModel.cc $(SRCDIR)/Discretization.cc \
	$(SRCDIR)/LinearSolver.cc \
	$(SRCDIR)/PressureSolver.cc $(SRCDIR)/MomentumSolver.cc \
	$(SRCDIR)/ScalarSolver.cc $(SRCDIR)/RBGSSolver.cc \
	$(SRCDIR)/Source.cc $(SRCDIR)/CellInformation.cc \
	$(SRCDIR)/ArchesLabel.cc $(SRCDIR)/ArchesVariables.cc \
	$(SRCDIR)/ArchesMaterial.cc

ifneq ($(PETSC_DIR),)
SRCS +=	$(SRCDIR)/PetscSolver.cc
else
SRCS +=	$(SRCDIR)/FakePetscSolver.cc
endif


ifneq ($(CC_DEPEND_REGEN),-MD)
# The fortran code doesn't work under g++ yet
SUBDIRS := $(SRCDIR)/fortran $(SRCDIR)/Mixing

include $(SCIRUN_SCRIPTS)/recurse.mk
FLIB := -lftn
endif

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

LIBS := $(XML_LIBRARY) $(FLIB) $(MPI_LIBRARY) -lm
ifneq ($(PETSC_DIR),)
LIBS := $(LIBS) $(PETSC_LIBS) -lpetscsles -lpetscdm -lpetscmat -lpetscvec -lpetsc -lblas
endif
#CFLAGS += -DARCHES_PETSC_DEBUG
#CFLAGS += -g -DARCHES_VEL_DEBUG
#CFLAGS += -g -DARCHES_DEBUG -DARCHES_GEOM_DEBUG -DARCHES_BC_DEBUG -DARCHES_COEF_DEBUG 
CFLAGS += 
#CFLAGS += -DARCHES_SRC_DEBUG -DARCHES_PRES_DEBUG -DARCHES_VEL_DEBUG
ifneq ($(PETSC_DIR),)
CFLAGS +=	-DHAVE_PETSC
endif
#LIBS += -lblas

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

