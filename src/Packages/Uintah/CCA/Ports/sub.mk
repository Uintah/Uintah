# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Ports

SRCS     += $(SRCDIR)/CFDInterface.cc $(SRCDIR)/DataArchive.cc \
	$(SRCDIR)/DataWarehouse.cc $(SRCDIR)/LoadBalancer.cc \
	$(SRCDIR)/MPMInterface.cc $(SRCDIR)/Output.cc \
	$(SRCDIR)/ProblemSpecInterface.cc \
	$(SRCDIR)/Scheduler.cc $(SRCDIR)/MPMCFDInterface.cc \
	$(SRCDIR)/MDInterface.cc $(SRCDIR)/Analyze.cc

PSELIBS := \
	Packages/Uintah/Core/Parallel \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Exceptions \
	Dataflow/XMLUtil \
	Core/Thread \
	Core/Exceptions \
	Core/Geometry \
	Core/Containers \
	Core/Util

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

