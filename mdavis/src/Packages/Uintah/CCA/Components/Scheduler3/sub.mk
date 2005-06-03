# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Scheduler3

SRCS += \
	$(SRCDIR)/MPIScheduler3.cc \
	$(SRCDIR)/PatchBasedDataWarehouse3.cc \
	$(SRCDIR)/Scheduler3Common.cc \
	$(SRCDIR)/SingleProcessorScheduler3.cc \
	$(SRCDIR)/TaskGraph3.cc \
	$(SRCDIR)/DetailedTasks3.cc


PSELIBS := \
	Packages/Uintah/CCA/Components/ProblemSpecification \
	Packages/Uintah/CCA/Components/Schedulers  \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/Util        \
	Packages/Uintah/Core/Exceptions  \
	Core/Geometry                    \
	Core/Containers                  \
	Core/Thread                      \
	Core/Exceptions                  \
	Core/Util

LIBS := $(XML_LIBRARY) $(TAU_LIBRARY) $(MPI_LIBRARY) $(VAMPIR_LIBRARY) $(PERFEX_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

