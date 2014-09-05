# Makefile fragment for this subdirectory
#

SRCDIR	:= Packages/Uintah/CCA/Components/MPM
SUBDIRS	:= $(SRCDIR)/ConstitutiveModel
include $(SCIRUN_SCRIPTS)/recurse.mk


include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR	:= Packages/Uintah/CCA/Components/MPM

SRCS     += $(SRCDIR)/SerialMPM.cc \
	$(SRCDIR)/RigidMPM.cc \
	$(SRCDIR)/FractureMPM.cc \
	$(SRCDIR)/ImpMPM.cc \
	$(SRCDIR)/ShellMPM.cc \
	$(SRCDIR)/MPMLabel.cc \
	$(SRCDIR)/PetscSolver.cc \
	$(SRCDIR)/SimpleSolver.cc \
	$(SRCDIR)/MPMBoundCond.cc
#	$(SRCDIR)/MPMAlgorithm.cc \
#	$(SRCDIR)/MPMDriver.cc \
#	$(SRCDIR)/Implicit.cc \
#	$(SRCDIR)/Explicit.cc 

SUBDIRS := \
	$(SRCDIR)/Contact \
	$(SRCDIR)/ThermalContact \
	$(SRCDIR)/GeometrySpecification \
	$(SRCDIR)/PhysicalBC \
	$(SRCDIR)/ParticleCreator \
	$(SRCDIR)/Crack			
	
include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
	Packages/Uintah/CCA/Components/MPM/ConstitutiveModel \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Math        \
	Core/Exceptions Core/Thread      \
	Core/Geometry Core/Util


LIBS := $(XML_LIBRARY) $(VT_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

