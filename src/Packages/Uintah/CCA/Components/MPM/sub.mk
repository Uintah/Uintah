# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPM

SRCS     += $(SRCDIR)/SerialMPM.cc \
	$(SRCDIR)/BoundaryCond.cc \
	$(SRCDIR)/MPMLabel.cc 

SUBDIRS := $(SRCDIR)/ConstitutiveModel $(SRCDIR)/Contact \
	$(SRCDIR)/Fracture \
	$(SRCDIR)/ThermalContact \
	$(SRCDIR)/GeometrySpecification \
	$(SRCDIR)/PhysicalBC \

#	$(SRCDIR)/Util <- Not used any more... 

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
	Core/Disclosure                  \
	Core/Geometry Dataflow/XMLUtil   \
	Packages/Uintah/CCA/Components/HETransformation


LIBS := $(XML_LIBRARY) $(VT_LIBRARY) $(MPI_LIBRARY) -lm

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

