# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Ports

SRCS += $(SRCDIR)/SimulationInterface.cc \
	$(SRCDIR)/DataWarehouse.cc \
	$(SRCDIR)/LoadBalancer.cc \
	$(SRCDIR)/ModelInterface.cc \
	$(SRCDIR)/ModelMaker.cc \
	$(SRCDIR)/Output.cc \
	$(SRCDIR)/ProblemSpecInterface.cc \
	$(SRCDIR)/Scheduler.cc \
	$(SRCDIR)/SolverInterface.cc

PSELIBS := \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/ProblemSpec \
	Core/Thread                      \
	Core/Exceptions                  \
	Core/Geometry                    \
	Core/Containers                  \
	Core/Util

LIBS := $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

