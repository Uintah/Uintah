# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/SwitchingCriteria

SRCS     += \
	$(SRCDIR)/SwitchingCriteriaFactory.cc \
	$(SRCDIR)/None.cc                     \
	$(SRCDIR)/TimestepNumber.cc           \
	$(SRCDIR)/SimpleBurn.cc               \
	$(SRCDIR)/SteadyBurn.cc               \
	$(SRCDIR)/SteadyState.cc

PSELIBS := \
	Core/Exceptions                  \
	Core/Geometry                    \
	Core/Util                        \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/Labels      \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/Util        \
	Packages/Uintah/Core/ProblemSpec 

LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

