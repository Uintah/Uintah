# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPM

SRCS     += $(SRCDIR)/SerialMPM.cc \
	$(SRCDIR)/RigidMPM.cc \
	$(SRCDIR)/FractureMPM.cc \
	$(SRCDIR)/ImpMPM.cc \
	$(SRCDIR)/ShellMPM.cc \
	$(SRCDIR)/MPMLabel.cc \
	$(SRCDIR)/Solver.cc 	\
	$(SRCDIR)/PetscSolver.cc \
	$(SRCDIR)/SimpleSolver.cc 
#	$(SRCDIR)/MPMAlgorithm.cc \
#	$(SRCDIR)/MPMDriver.cc \
#	$(SRCDIR)/Implicit.cc \
#	$(SRCDIR)/Explicit.cc 

SUBDIRS := $(SRCDIR)/ConstitutiveModel $(SRCDIR)/Contact \
	$(SRCDIR)/ThermalContact \
	$(SRCDIR)/GeometrySpecification \
	$(SRCDIR)/PhysicalBC \
	$(SRCDIR)/ParticleCreator \
	$(SRCDIR)/Crack			
	
include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Math        \
	Core/Exceptions Core/Thread      \
	Core/Geometry Core/Util		 \
	Packages/Uintah/CCA/Components/HETransformation


LIBS := $(XML_LIBRARY) $(VT_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

