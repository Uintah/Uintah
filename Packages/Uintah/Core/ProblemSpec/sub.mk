# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/ProblemSpec

SRCS     += \
	$(SRCDIR)/ProblemSpec.cc 

PSELIBS := \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/Util \
	Core/Exceptions \
	Core/Thread

LIBS := $(XML_LIBRARY) 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

