# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Regridder

SRCS     += $(SRCDIR)/RegridderCommon.cc \
	    $(SRCDIR)/RegridderFactory.cc \
	    $(SRCDIR)/HierarchicalRegridder.cc \
	    $(SRCDIR)/BNRRegridder.cc

PSELIBS := \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Util        \
	Packages/Uintah/Core/Exceptions  \
	Core/Geometry			 \
	Core/Exceptions			 \
	Core/Thread			 \
	Core/Util


LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

