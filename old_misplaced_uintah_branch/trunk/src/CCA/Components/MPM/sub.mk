# Makefile fragment for this subdirectory
#

#SRCDIR	:= Packages/Uintah/CCA/Components/MPM
#SUBDIRS	:= $(SRCDIR)/ConstitutiveModel
#include $(SCIRUN_SCRIPTS)/recurse.mk


include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR	:= Packages/Uintah/CCA/Components/MPM

SRCS     += $(SRCDIR)/SerialMPM.cc \
	$(SRCDIR)/RigidMPM.cc \
	$(SRCDIR)/MPMCommon.cc \
	$(SRCDIR)/FractureMPM.cc \
	$(SRCDIR)/ImpMPM.cc \
	$(SRCDIR)/ShellMPM.cc \
	$(SRCDIR)/AMRMPM.cc \
	$(SRCDIR)/SimpleSolver.cc \
	$(SRCDIR)/Solver.cc \
	$(SRCDIR)/MPMBoundCond.cc \
	$(SRCDIR)/MPMFlags.cc	\
	$(SRCDIR)/ImpMPMFlags.cc

ifeq ($(HAVE_PETSC),yes)
  SRCS += $(SRCDIR)/PetscSolver.cc 
else
  SRCS += $(SRCDIR)/FakePetscSolver.cc
endif

SUBDIRS := \
	$(SRCDIR)/ConstitutiveModel \
	$(SRCDIR)/Contact \
	$(SRCDIR)/ThermalContact \
	$(SRCDIR)/PhysicalBC \
	$(SRCDIR)/ParticleCreator \
	$(SRCDIR)/Crack		\
	$(SRCDIR)/HeatConduction

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Util        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Labels      \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/GeometryPiece  \
	Packages/Uintah/Core/Math        \
	Core/Exceptions Core/Thread      \
	Core/Geometry Core/Util          \
	Core/Math


LIBS := $(XML2_LIBRARY) $(VT_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

ifeq ($(HAVE_PETSC),yes)
  LIBS := $(LIBS) $(PETSC_LIBRARY) 
endif


include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

