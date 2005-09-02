# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/SwitchingCriteria

SRCS     += \
	$(SRCDIR)/SwitchingCriteriaFactory.cc \
	$(SRCDIR)/None.cc                     \
	$(SRCDIR)/TimestepNumber.cc           \
	$(SRCDIR)/PBXTemperature.cc 	      \
	$(SRCDIR)/SteadyState.cc

PSELIBS := \
	Core/Exceptions                  \
	Core/Util                        \
	Packages/Uintah/CCA/Components/MPM \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Labels      \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/Util        \
	Packages/Uintah/Core/ProblemSpec 

LIBS := $(XML_LIBRARY) 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

