# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Ports

SRCS     += $(SRCDIR)/SimulationInterface.cc $(SRCDIR)/DataArchive.cc \
	$(SRCDIR)/DataWarehouse.cc $(SRCDIR)/LoadBalancer.cc \
	$(SRCDIR)/Output.cc \
	$(SRCDIR)/ProblemSpecInterface.cc \
	$(SRCDIR)/Scheduler.cc

PSELIBS := \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Exceptions  \
	Dataflow/XMLUtil                 \
	Core/Thread                      \
	Core/Exceptions                  \
	Core/Geometry                    \
	Core/Containers                  \
	Core/Util

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

