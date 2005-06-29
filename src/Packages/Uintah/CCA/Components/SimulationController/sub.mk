# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/SimulationController

SRCS     += $(SRCDIR)/SimulationController.cc \
            $(SRCDIR)/AMRSimulationController.cc 

PSELIBS := \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/DataArchive \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Util        \
	Packages/Uintah/CCA/Components/DataArchiver    \
	Packages/Uintah/CCA/Components/PatchCombiner   \
	Packages/Uintah/CCA/Components/Regridder       \
	Packages/Uintah/CCA/Components/Switcher   \
	Packages/Uintah/CCA/Components/DataArchiver \
	Core/OS       \
	Core/Geometry \
	Core/Thread   \
	Core/Util     \
	Core/Exceptions

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

