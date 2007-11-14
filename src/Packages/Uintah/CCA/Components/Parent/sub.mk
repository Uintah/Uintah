# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR  := Packages/Uintah/CCA/Components/Parent
COMPONENTS := Packages/Uintah/CCA/Components

SRCS    := $(SRCDIR)/Switcher.cc \
	   $(SRCDIR)/ComponentFactory.cc 

# ARCHES et. al. should have been seen by CCA/Components/sub.mk
PSELIBS := \
	Core/Containers \
	Core/Exceptions \
	Core/Util \
	Core/Geometry \
        Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Disclosure  \
        Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Math	 \
        Packages/Uintah/Core/Parallel    \
        Packages/Uintah/Core/ProblemSpec \
        Packages/Uintah/Core/Util        \
        $(ARCHES) \
        $(MPMARCHES) \
        $(ICE)    \
        $(MPM)    \
        $(MPMICE) \
        $(COMPONENTS)/Examples             \
	$(COMPONENTS)/PatchCombiner        \
	$(COMPONENTS)/ProblemSpecification \
	$(COMPONENTS)/Solvers              \
	$(COMPONENTS)/SwitchingCriteria

LIBS    := $(XML2_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
