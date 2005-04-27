# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/SimulationController

SRCS     += $(SRCDIR)/SimulationController.cc \
            $(SRCDIR)/AMRSimulationController.cc

PSELIBS := \
	Packages/Uintah/Core		\
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/CCA/Components/PatchCombiner   \
	Packages/Uintah/CCA/Components/Regridder   \
	Core/OS       \
	Core/Geometry \
	Core/Thread   \
	Core/Util   \
	Core/Exceptions

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

