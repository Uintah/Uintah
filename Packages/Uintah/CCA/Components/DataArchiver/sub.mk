# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/DataArchiver

SRCS     += $(SRCDIR)/DataArchiver.cc

PSELIBS := \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/Core/Parallel \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/ProblemSpec \
	Core/OS \
	Core/Exceptions \
	Core/Containers \
	Dataflow/XMLUtil \
	Core/Util

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) -lm

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

