# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/PatchCombiner

SRCS     += $(SRCDIR)/PatchCombiner.cc

PSELIBS := \
	Packages/Uintah/CCA/Components/Schedulers        \
	Packages/Uintah/Core/DataArchive \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/ProblemSpec \
	Core/OS          \
	Core/Exceptions  \
	Core/Containers  \
	Core/Thread      \
	Core/Util

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

