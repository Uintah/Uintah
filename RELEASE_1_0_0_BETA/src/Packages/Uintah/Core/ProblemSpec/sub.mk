# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/ProblemSpec

SRCS     += \
	$(SRCDIR)/ProblemSpec.cc \
	$(SRCDIR)/RefCounted.cc

PSELIBS := \
	Packages/Uintah/Core/Exceptions \
	Core/Exceptions \
	Dataflow/XMLUtil \
	Core/Thread


LIBS := $(XML_LIBRARY) 

include $(SRCTOP)/scripts/smallso_epilogue.mk

