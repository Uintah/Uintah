# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/SwitchingCriteria

SRCS     += \
	$(SRCDIR)/SwitchingCriteriaFactory.cc \
	$(SRCDIR)/None.cc   \
	$(SRCDIR)/TimestepNumber.cc

PSELIBS := \
	Core/Exceptions                  \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/Util    \
	Packages/Uintah/Core/ProblemSpec 

LIBS := $(XML_LIBRARY) 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

