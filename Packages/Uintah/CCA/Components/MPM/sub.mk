# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPM

SRCS     += $(SRCDIR)/SerialMPM.cc \
	$(SRCDIR)/BoundCond.cc \
	$(SRCDIR)/MPMLabel.cc \
	$(SRCDIR)/MPMPhysicalModules.cc

SUBDIRS := $(SRCDIR)/ConstitutiveModel $(SRCDIR)/Contact \
	$(SRCDIR)/Fracture \
	$(SRCDIR)/ThermalContact \
	$(SRCDIR)/GeometrySpecification \
	$(SRCDIR)/PhysicalBC \
	$(SRCDIR)/Util

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Parallel \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/Math \
	Core/Exceptions Core/Thread \
	Core/Geometry Dataflow/XMLUtil \
	Packages/Uintah/CCA/Components/ICE


LIBS := $(XML_LIBRARY) $(VT_LIBRARY) $(MPI_LIBRARY) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

