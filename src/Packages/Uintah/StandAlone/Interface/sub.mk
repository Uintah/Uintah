# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Interface

SRCS     += $(SRCDIR)/CFDInterface.cc $(SRCDIR)/DataArchive.cc \
	$(SRCDIR)/DataWarehouse.cc $(SRCDIR)/LoadBalancer.cc \
	$(SRCDIR)/MPMInterface.cc $(SRCDIR)/Output.cc \
	$(SRCDIR)/ProblemSpec.cc $(SRCDIR)/ProblemSpecInterface.cc \
	$(SRCDIR)/Scheduler.cc $(SRCDIR)/MPMCFDInterface.cc \
	$(SRCDIR)/MDInterface.cc $(SRCDIR)/Analyze.cc

PSELIBS := Uintah/Parallel Uintah/Grid Uintah/Exceptions Dataflow/XMLUtil \
	Core/Thread Core/Exceptions Core/Geometry Core/Util
LIBS := $(XML_LIBRARY) -lmpi

include $(SRCTOP)/scripts/smallso_epilogue.mk

