# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/SimulationController

SRCS     += $(SRCDIR)/SimulationController.cc \
            $(SRCDIR)/AMRSimulationController.cc \
            $(SRCDIR)/SimpleSimulationController.cc

PSELIBS := \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/DataArchive \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Grid        \
	Core/OS       \
	Core/Geometry \
	Core/Thread   \
	Core/Util   \
	Core/Exceptions

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

